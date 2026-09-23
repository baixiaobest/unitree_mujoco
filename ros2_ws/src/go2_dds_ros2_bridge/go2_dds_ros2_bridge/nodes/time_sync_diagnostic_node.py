#!/usr/bin/env python3

"""Measure Unitree LiDAR and IMU timestamp relationships without altering data.

The diagnostic subscribes directly to the Unitree DDS topics so that the cloud
header, LowState tick, and optional LiDAR-IMU header are observed before any of
the bridge timestamp reconstruction is applied.  Every callback is paired with
both CLOCK_REALTIME and CLOCK_MONOTONIC timestamps and written to one CSV file.

Recorded diagnostic findings (2026-09-21)
===========================================

Capture: /tmp/go2_motion_time_sync.csv (60 seconds; stationary, then moving).

Measured facts
--------------
* The raw LiDAR cloud header is the start of a rolling scan. The ``time`` point
  field is relative to that header in seconds. Scan duration was 62.822 ms
  median (p5-p95: 62.285-67.320 ms) at 15.256 Hz.
* The DDS callback is received after scan completion. The host-receive minus
  raw scan-end offset was 2,636,981.724 ms median, with a 3.385 ms p5-p95
  spread. The large absolute value is simply the host and LiDAR clocks having
  different epochs; it is not a transport delay.
* ``LowState.tick`` is an unsigned 32-bit millisecond counter. It advanced by
  2 ms median between the 499.623 Hz body-IMU messages. In contrast, DDS
  callback intervals had p1/median/p99 values of 0.077/1.981/5.360 ms, so
  callback time cannot safely define individual IMU sample stamps.
* The raw native LiDAR IMU and body IMU gyro magnitudes agree strongly during
  motion (peak correlation about 0.998). After independently mapping each
  sensor clock to host time, their residual phase was consistently 1-2 ms.
  Clock-rate mismatch was below 0.5 ppm relative over this capture.

Confirmed defects and corresponding bridge fixes
-------------------------------------------------
* The former cloud bridge used ``host_receive - raw_scan_start`` as its offset,
  making the output cloud header equal to callback receive time. Per-point
  offsets then placed the scan end about 62.8 ms in the future. The bridge now
  estimates offset from raw scan end, then applies that offset to raw scan
  start; recorded-data replay leaves scan end 0-3.3 ms before receipt.
* The former body-IMU bridge reconstructed time from callback spacing and
  repeatedly re-anchored to callback time. It now uses ``LowState.tick`` and a
  rolling low-delay host-clock map.

These results establish timestamp handling as a serious FAST-LIO input defect,
but do not by themselves prove it is the only possible cause of localization
divergence. Validate the corrected bridges with a safe fast-motion test near
structure before changing FAST-LIO tuning or calibration.
"""

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import sys
import threading
from time import monotonic_ns, time_ns

import numpy as np
import rclpy
from rclpy.node import Node

from go2_dds_ros2_bridge.dds_runtime import add_dds_runtime_arguments, resolve_runtime_arguments


DEFAULT_LIDAR_DDS_TOPIC = "rt/utlidar/cloud"
DEFAULT_LOWSTATE_DDS_TOPIC = "rt/lowstate"
DEFAULT_LIDAR_IMU_DDS_TOPIC = "rt/utlidar/imu"
DEFAULT_DURATION_SEC = 60.0
DEFAULT_SUMMARY_PERIOD_SEC = 5.0
LOWSTATE_TICK_WRAPAROUND = 1 << 32
TIMESTAMP_FIELD_NAMES = ("t", "timestamp", "timestamps", "time", "time_stamp")
POINT_TIME_UNIT_SCALES = {
    "seconds": 1.0,
    "milliseconds": 1e-3,
    "microseconds": 1e-6,
    "nanoseconds": 1e-9,
}
POINT_FIELD_DTYPES = {
    1: np.dtype(np.int8),
    2: np.dtype(np.uint8),
    3: np.dtype(np.int16),
    4: np.dtype(np.uint16),
    5: np.dtype(np.int32),
    6: np.dtype(np.uint32),
    7: np.dtype(np.float32),
    8: np.dtype(np.float64),
}
CSV_FIELDS = (
    "event",
    "sequence",
    "host_wall_ns",
    "host_monotonic_ns",
    "sensor_stamp_ns",
    "scan_end_ns",
    "point_count",
    "timestamp_field",
    "point_time_unit",
    "point_time_min_raw",
    "point_time_max_raw",
    "scan_duration_sec",
    "receive_minus_sensor_sec",
    "receive_minus_scan_end_sec",
    "lowstate_tick_raw",
    "lowstate_tick_unwrapped",
    "tick_delta",
    "receive_interval_sec",
    "seconds_per_tick",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "accel_x",
    "accel_y",
    "accel_z",
)


@dataclass(frozen=True)
class DiagnosticConfig:
    lidar_dds_topic: str
    lowstate_dds_topic: str
    lidar_imu_dds_topic: str
    subscribe_lidar_imu: bool
    point_time_unit: str
    duration_sec: float
    summary_period_sec: float
    csv_path: Path
    dds_domain_id: int
    dds_interface: str


def default_csv_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("/tmp") / f"go2_time_sync_{timestamp}.csv"


def parse_args() -> DiagnosticConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Record raw Unitree DDS LiDAR scan timestamps, LowState.tick, and optional LiDAR-IMU "
            "timestamps against the same host clocks. This node does not republish or modify sensor data."
        )
    )
    add_dds_runtime_arguments(parser)
    parser.add_argument("--lidar-dds-topic", default=DEFAULT_LIDAR_DDS_TOPIC)
    parser.add_argument("--lowstate-dds-topic", default=DEFAULT_LOWSTATE_DDS_TOPIC)
    parser.add_argument("--lidar-imu-dds-topic", default=DEFAULT_LIDAR_IMU_DDS_TOPIC)
    parser.add_argument(
        "--subscribe-lidar-imu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also probe the native LiDAR IMU DDS topic. Disable if the topic is unavailable.",
    )
    parser.add_argument(
        "--point-time-unit",
        choices=("auto", *POINT_TIME_UNIT_SCALES),
        default="auto",
        help="Unit of the per-point time field. Auto compares its span with consecutive cloud headers.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=DEFAULT_DURATION_SEC,
        help="Measurement duration. Use 0 or a negative value to run until Ctrl-C.",
    )
    parser.add_argument(
        "--summary-period-sec",
        type=float,
        default=DEFAULT_SUMMARY_PERIOD_SEC,
        help="Period for cumulative console summaries.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=default_csv_path(),
        help="Output CSV path. The default is a unique timestamped file under /tmp.",
    )
    non_ros_args = rclpy.utilities.remove_ros_args(args=sys.argv)[1:]
    args = parser.parse_args(non_ros_args)
    runtime = resolve_runtime_arguments(args)
    return DiagnosticConfig(
        lidar_dds_topic=str(args.lidar_dds_topic),
        lowstate_dds_topic=str(args.lowstate_dds_topic),
        lidar_imu_dds_topic=str(args.lidar_imu_dds_topic),
        subscribe_lidar_imu=bool(args.subscribe_lidar_imu),
        point_time_unit=str(args.point_time_unit),
        duration_sec=float(args.duration_sec),
        summary_period_sec=max(float(args.summary_period_sec), 0.5),
        csv_path=Path(args.csv_path).expanduser().resolve(),
        dds_domain_id=runtime.domain_id,
        dds_interface=runtime.interface,
    )


def import_raw_dds_dependencies():
    try:
        from cyclonedds import idl
        from cyclonedds.idl import annotations as annotate
        from cyclonedds.idl import types
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Quaternion_, Vector3_
        from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import Header_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
    except ModuleNotFoundError as error:
        raise SystemExit(
            "The diagnostic requires cyclonedds and unitree_sdk2py in the same Python environment as ROS2."
        ) from error

    # unitree_sdk2py currently omits the generated sensor_msgs/Imu Python class,
    # although Unitree ships its matching C++ IDL.  Define that exact DDS type
    # locally so the native rt/utlidar/imu topic can still be inspected.
    @dataclass
    @annotate.final
    @annotate.autoid("sequential")
    class DdsImu(idl.IdlStruct, typename="sensor_msgs.msg.dds_.Imu_"):
        header: Header_
        orientation: Quaternion_
        orientation_covariance: types.array[types.float64, 9]
        angular_velocity: Vector3_
        angular_velocity_covariance: types.array[types.float64, 9]
        linear_acceleration: Vector3_
        linear_acceleration_covariance: types.array[types.float64, 9]

    return ChannelFactoryInitialize, ChannelSubscriber, PointCloud2_, LowState_, DdsImu


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def point_cloud_dtype(fields, point_step: int, is_bigendian: bool) -> np.dtype:
    names: list[str] = []
    formats: list[np.dtype] = []
    offsets: list[int] = []
    byte_order = ">" if is_bigendian else "<"
    for index, field in enumerate(fields):
        base_dtype = POINT_FIELD_DTYPES.get(int(field.datatype))
        if base_dtype is None:
            raise ValueError(f"unsupported PointField datatype {field.datatype}")
        field_dtype = base_dtype.newbyteorder(byte_order)
        if int(field.count) > 1:
            field_dtype = np.dtype((field_dtype, int(field.count)))
        names.append(str(field.name) or f"unnamed_{index}")
        formats.append(field_dtype)
        offsets.append(int(field.offset))
    return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": point_step})


def point_time_range(msg) -> tuple[str | None, float, float]:
    dtype = point_cloud_dtype(msg.fields, int(msg.point_step), bool(msg.is_bigendian))
    field_names = dtype.names or ()
    timestamp_field = next((name for name in TIMESTAMP_FIELD_NAMES if name in field_names), None)
    if timestamp_field is None:
        return None, float("nan"), float("nan")

    width = int(msg.width)
    height = int(msg.height)
    point_step = int(msg.point_step)
    packed_row_step = width * point_step
    row_step = int(msg.row_step) if int(msg.row_step) > 0 else packed_row_step
    if row_step < packed_row_step:
        raise ValueError(f"row_step={row_step} is smaller than width*point_step={packed_row_step}")
    expected_size = row_step * height
    if len(msg.data) < expected_size:
        raise ValueError(f"cloud data has {len(msg.data)} bytes; expected at least {expected_size}")

    cloud = np.ndarray(
        shape=(height, width),
        dtype=dtype,
        buffer=bytes(msg.data),
        strides=(row_step, point_step),
    )
    values = np.asarray(cloud[timestamp_field], dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return timestamp_field, float("nan"), float("nan")
    return timestamp_field, float(np.min(values)), float(np.max(values))


def percentile(values: list[float], quantile: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, quantile)) if array.size else float("nan")


def median(values: list[float]) -> float:
    return percentile(values, 50.0)


def spread_ms(values: list[float]) -> float:
    return (percentile(values, 95.0) - percentile(values, 5.0)) * 1e3


class TimeSyncDiagnostic(Node):
    def __init__(
        self,
        config: DiagnosticConfig,
        channel_subscriber_cls,
        point_cloud_type,
        lowstate_type,
        lidar_imu_type,
    ) -> None:
        super().__init__("go2_time_sync_diagnostic")
        self._config = config
        self._lock = threading.Lock()
        self._start_monotonic_ns = monotonic_ns()
        self.done = False
        self._finalized = False
        self._accept_samples = True
        self._csv_rows_since_flush = 0
        self._sequence = Counter()
        self._last_cloud_header_ns: int | None = None
        self._last_cloud_receive_ns: int | None = None
        self._last_tick: int | None = None
        self._unwrapped_tick: int | None = None
        self._last_lowstate_receive_ns: int | None = None
        self._last_lidar_imu_receive_ns: int | None = None
        self._tick_reset_count = 0
        self._cloud_decode_error_count = 0
        self._cloud_schema_logged = False

        self._cloud_header_intervals: list[float] = []
        self._cloud_receive_intervals: list[float] = []
        self._cloud_scan_durations: list[float] = []
        self._cloud_receive_minus_start: list[float] = []
        self._cloud_receive_minus_end: list[float] = []
        self._point_time_units: Counter[str] = Counter()
        self._tick_deltas: list[int] = []
        self._seconds_per_tick: list[float] = []
        self._tick_fit_values: list[float] = []
        self._tick_fit_receive_sec: list[float] = []
        self._tick_offsets_assuming_ms: list[float] = []
        self._lidar_imu_intervals: list[float] = []
        self._lidar_imu_receive_offsets: list[float] = []

        self._config.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = self._config.csv_path.open("x", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=CSV_FIELDS)
        self._csv_writer.writeheader()
        self._csv_file.flush()

        self._cloud_subscriber = channel_subscriber_cls(self._config.lidar_dds_topic, point_cloud_type)
        self._cloud_subscriber.Init(self._cloud_callback, 10)
        self._lowstate_subscriber = channel_subscriber_cls(self._config.lowstate_dds_topic, lowstate_type)
        self._lowstate_subscriber.Init(self._lowstate_callback, 100)
        self._lidar_imu_subscriber = None
        if self._config.subscribe_lidar_imu:
            self._lidar_imu_subscriber = channel_subscriber_cls(
                self._config.lidar_imu_dds_topic, lidar_imu_type
            )
            self._lidar_imu_subscriber.Init(self._lidar_imu_callback, 100)

        self._summary_timer = self.create_timer(self._config.summary_period_sec, self._timer_callback)
        self.get_logger().info(
            "Measuring raw DDS timing on domain=%d interface=%s for %s. cloud='%s', lowstate='%s', "
            "lidar_imu='%s' (%s), point_time_unit=%s. CSV: %s"
            % (
                self._config.dds_domain_id,
                self._config.dds_interface,
                "until Ctrl-C" if self._config.duration_sec <= 0.0 else f"{self._config.duration_sec:.1f}s",
                self._config.lidar_dds_topic,
                self._config.lowstate_dds_topic,
                self._config.lidar_imu_dds_topic,
                "enabled" if self._config.subscribe_lidar_imu else "disabled",
                self._config.point_time_unit,
                self._config.csv_path,
            )
        )

    def _next_sequence(self, event: str) -> int:
        self._sequence[event] += 1
        return self._sequence[event]

    def _write_row(self, values: dict) -> None:
        row = {field: "" for field in CSV_FIELDS}
        row.update(values)
        self._csv_writer.writerow(row)
        self._csv_rows_since_flush += 1
        if self._csv_rows_since_flush >= 100:
            self._csv_file.flush()
            self._csv_rows_since_flush = 0

    def _infer_point_time_unit(self, raw_span: float, reference_interval_sec: float) -> str | None:
        if self._config.point_time_unit != "auto":
            return self._config.point_time_unit
        if not math.isfinite(raw_span) or raw_span <= 0.0:
            return None
        if not math.isfinite(reference_interval_sec) or reference_interval_sec <= 0.0:
            return None
        return min(
            POINT_TIME_UNIT_SCALES,
            key=lambda unit: abs(math.log10(raw_span * POINT_TIME_UNIT_SCALES[unit] / reference_interval_sec)),
        )

    def _cloud_callback(self, msg) -> None:
        host_wall_ns = time_ns()
        host_monotonic_ns = monotonic_ns()
        sensor_stamp_ns = stamp_to_ns(msg.header.stamp)
        try:
            timestamp_field, point_time_min_raw, point_time_max_raw = point_time_range(msg)
        except (TypeError, ValueError) as error:
            with self._lock:
                self._cloud_decode_error_count += 1
                error_count = self._cloud_decode_error_count
            if error_count <= 3:
                self.get_logger().error(f"Could not decode cloud point timing: {error}")
            timestamp_field = None
            point_time_min_raw = float("nan")
            point_time_max_raw = float("nan")

        with self._lock:
            if not self._accept_samples:
                return
            header_interval_sec = float("nan")
            if self._last_cloud_header_ns is not None and sensor_stamp_ns > self._last_cloud_header_ns:
                header_interval_sec = (sensor_stamp_ns - self._last_cloud_header_ns) * 1e-9
                self._cloud_header_intervals.append(header_interval_sec)
            receive_interval_sec = float("nan")
            if self._last_cloud_receive_ns is not None:
                receive_interval_sec = (host_monotonic_ns - self._last_cloud_receive_ns) * 1e-9
                if receive_interval_sec > 0.0:
                    self._cloud_receive_intervals.append(receive_interval_sec)
            self._last_cloud_header_ns = sensor_stamp_ns
            self._last_cloud_receive_ns = host_monotonic_ns

            reference_interval_sec = header_interval_sec
            if not math.isfinite(reference_interval_sec) or reference_interval_sec <= 0.0:
                reference_interval_sec = receive_interval_sec
            raw_span = point_time_max_raw - point_time_min_raw
            point_time_unit = self._infer_point_time_unit(raw_span, reference_interval_sec)
            scan_duration_sec = float("nan")
            scan_end_ns = 0
            receive_minus_end_sec = float("nan")
            if point_time_unit is not None and math.isfinite(point_time_max_raw):
                scan_duration_sec = max(point_time_max_raw, 0.0) * POINT_TIME_UNIT_SCALES[point_time_unit]
                scan_end_ns = sensor_stamp_ns + int(round(scan_duration_sec * 1e9))
                receive_minus_end_sec = (host_wall_ns - scan_end_ns) * 1e-9
                self._cloud_scan_durations.append(scan_duration_sec)
                self._cloud_receive_minus_end.append(receive_minus_end_sec)
                self._point_time_units[point_time_unit] += 1
            receive_minus_start_sec = (host_wall_ns - sensor_stamp_ns) * 1e-9
            self._cloud_receive_minus_start.append(receive_minus_start_sec)

            self._write_row(
                {
                    "event": "lidar_cloud",
                    "sequence": self._next_sequence("lidar_cloud"),
                    "host_wall_ns": host_wall_ns,
                    "host_monotonic_ns": host_monotonic_ns,
                    "sensor_stamp_ns": sensor_stamp_ns,
                    "scan_end_ns": scan_end_ns or "",
                    "point_count": int(msg.width) * int(msg.height),
                    "timestamp_field": timestamp_field or "",
                    "point_time_unit": point_time_unit or "unknown",
                    "point_time_min_raw": point_time_min_raw,
                    "point_time_max_raw": point_time_max_raw,
                    "scan_duration_sec": scan_duration_sec,
                    "receive_minus_sensor_sec": receive_minus_start_sec,
                    "receive_minus_scan_end_sec": receive_minus_end_sec,
                    "receive_interval_sec": receive_interval_sec,
                }
            )

        if not self._cloud_schema_logged:
            self._cloud_schema_logged = True
            schema = ", ".join(
                f"{field.name}:{int(field.datatype)}[{int(field.count)}]@{int(field.offset)}"
                for field in msg.fields
            )
            self.get_logger().info(
                "Raw cloud schema: frame='%s', %dx%d, point_step=%d, row_step=%d, fields=[%s]"
                % (
                    msg.header.frame_id,
                    int(msg.width),
                    int(msg.height),
                    int(msg.point_step),
                    int(msg.row_step),
                    schema,
                )
            )

    def _lowstate_callback(self, msg) -> None:
        host_wall_ns = time_ns()
        host_monotonic_ns = monotonic_ns()
        tick = int(msg.tick) % LOWSTATE_TICK_WRAPAROUND
        imu = msg.imu_state
        with self._lock:
            if not self._accept_samples:
                return
            tick_delta: int | None = None
            receive_interval_sec = float("nan")
            seconds_per_tick = float("nan")
            if self._last_tick is None or self._unwrapped_tick is None:
                self._unwrapped_tick = tick
            else:
                if tick >= self._last_tick:
                    candidate_delta = tick - self._last_tick
                elif self._last_tick > 0xF0000000 and tick < 0x0FFFFFFF:
                    candidate_delta = (LOWSTATE_TICK_WRAPAROUND - self._last_tick) + tick
                else:
                    candidate_delta = -1
                    self._tick_reset_count += 1

                if candidate_delta >= 0:
                    tick_delta = candidate_delta
                    self._unwrapped_tick += candidate_delta
                else:
                    self._unwrapped_tick = tick

                if self._last_lowstate_receive_ns is not None:
                    receive_interval_sec = (host_monotonic_ns - self._last_lowstate_receive_ns) * 1e-9
                if tick_delta is not None and tick_delta > 0 and receive_interval_sec > 0.0:
                    seconds_per_tick = receive_interval_sec / tick_delta
                    self._tick_deltas.append(tick_delta)
                    self._seconds_per_tick.append(seconds_per_tick)

            self._last_tick = tick
            self._last_lowstate_receive_ns = host_monotonic_ns
            self._tick_fit_values.append(float(self._unwrapped_tick))
            self._tick_fit_receive_sec.append(host_monotonic_ns * 1e-9)
            tick_offset_assuming_ms = host_wall_ns * 1e-9 - self._unwrapped_tick * 1e-3
            self._tick_offsets_assuming_ms.append(tick_offset_assuming_ms)
            self._write_row(
                {
                    "event": "body_imu",
                    "sequence": self._next_sequence("body_imu"),
                    "host_wall_ns": host_wall_ns,
                    "host_monotonic_ns": host_monotonic_ns,
                    "lowstate_tick_raw": tick,
                    "lowstate_tick_unwrapped": self._unwrapped_tick,
                    "tick_delta": "" if tick_delta is None else tick_delta,
                    "receive_interval_sec": receive_interval_sec,
                    "seconds_per_tick": seconds_per_tick,
                    "gyro_x": float(imu.gyroscope[0]),
                    "gyro_y": float(imu.gyroscope[1]),
                    "gyro_z": float(imu.gyroscope[2]),
                    "accel_x": float(imu.accelerometer[0]),
                    "accel_y": float(imu.accelerometer[1]),
                    "accel_z": float(imu.accelerometer[2]),
                }
            )

    def _lidar_imu_callback(self, msg) -> None:
        host_wall_ns = time_ns()
        host_monotonic_ns = monotonic_ns()
        sensor_stamp_ns = stamp_to_ns(msg.header.stamp)
        with self._lock:
            if not self._accept_samples:
                return
            receive_interval_sec = float("nan")
            if self._last_lidar_imu_receive_ns is not None:
                receive_interval_sec = (host_monotonic_ns - self._last_lidar_imu_receive_ns) * 1e-9
                if receive_interval_sec > 0.0:
                    self._lidar_imu_intervals.append(receive_interval_sec)
            self._last_lidar_imu_receive_ns = host_monotonic_ns
            receive_offset_sec = (host_wall_ns - sensor_stamp_ns) * 1e-9
            self._lidar_imu_receive_offsets.append(receive_offset_sec)
            self._write_row(
                {
                    "event": "lidar_imu",
                    "sequence": self._next_sequence("lidar_imu"),
                    "host_wall_ns": host_wall_ns,
                    "host_monotonic_ns": host_monotonic_ns,
                    "sensor_stamp_ns": sensor_stamp_ns,
                    "receive_minus_sensor_sec": receive_offset_sec,
                    "receive_interval_sec": receive_interval_sec,
                    "gyro_x": float(msg.angular_velocity.x),
                    "gyro_y": float(msg.angular_velocity.y),
                    "gyro_z": float(msg.angular_velocity.z),
                    "accel_x": float(msg.linear_acceleration.x),
                    "accel_y": float(msg.linear_acceleration.y),
                    "accel_z": float(msg.linear_acceleration.z),
                }
            )

    @staticmethod
    def _rate_hz(intervals: list[float]) -> float:
        interval = median(intervals)
        return 1.0 / interval if math.isfinite(interval) and interval > 0.0 else float("nan")

    @staticmethod
    def _linear_slope(x_values: list[float], y_values: list[float]) -> float:
        if len(x_values) < 2 or len(y_values) != len(x_values):
            return float("nan")
        x = np.asarray(x_values, dtype=np.float64)
        y = np.asarray(y_values, dtype=np.float64)
        x -= x[0]
        y -= y[0]
        if not np.any(x != 0.0):
            return float("nan")
        return float(np.polyfit(x, y, 1)[0])

    def _timer_callback(self) -> None:
        self.log_summary(final=False)
        if self._config.duration_sec > 0.0:
            elapsed_sec = (monotonic_ns() - self._start_monotonic_ns) * 1e-9
            if elapsed_sec >= self._config.duration_sec:
                self.done = True

    def log_summary(self, *, final: bool) -> None:
        with self._lock:
            elapsed_sec = (monotonic_ns() - self._start_monotonic_ns) * 1e-9
            counts = dict(self._sequence)
            units = dict(self._point_time_units)
            cloud_rate_hz = self._rate_hz(self._cloud_receive_intervals)
            scan_duration_ms = median(self._cloud_scan_durations) * 1e3
            start_offset_sec = median(self._cloud_receive_minus_start)
            start_spread_ms = spread_ms(self._cloud_receive_minus_start)
            end_offset_sec = median(self._cloud_receive_minus_end)
            end_spread_ms = spread_ms(self._cloud_receive_minus_end)
            tick_delta = median([float(value) for value in self._tick_deltas])
            seconds_per_tick = median(self._seconds_per_tick)
            fitted_seconds_per_tick = self._linear_slope(
                self._tick_fit_values, self._tick_fit_receive_sec
            )
            tick_offset_spread_ms = spread_ms(self._tick_offsets_assuming_ms)
            lidar_imu_rate_hz = self._rate_hz(self._lidar_imu_intervals)
            lidar_imu_offset_sec = median(self._lidar_imu_receive_offsets)
            lidar_imu_offset_spread_ms = spread_ms(self._lidar_imu_receive_offsets)
            self._csv_file.flush()
            self._csv_rows_since_flush = 0

        label = "FINAL" if final else "TIMING"
        self.get_logger().info(
            "%s elapsed=%.1fs samples(cloud/body_imu/lidar_imu)=%d/%d/%d; "
            "cloud=%.2fHz point_units=%s scan_duration_p50=%.3fms "
            "receive-start_p50/spread=%.6fs/%.3fms receive-end_p50/spread=%.6fs/%.3fms; "
            "tick_delta_p50=%.3f arrival_ratio_p50=%.3fus/tick fitted_scale=%.3fus/tick "
            "ms_tick_offset_spread=%.3fms resets=%d; "
            "lidar_imu=%.2fHz receive-stamp_p50/spread=%.6fs/%.3fms decode_errors=%d"
            % (
                label,
                elapsed_sec,
                counts.get("lidar_cloud", 0),
                counts.get("body_imu", 0),
                counts.get("lidar_imu", 0),
                cloud_rate_hz,
                units,
                scan_duration_ms,
                start_offset_sec,
                start_spread_ms,
                end_offset_sec,
                end_spread_ms,
                tick_delta,
                seconds_per_tick * 1e6,
                fitted_seconds_per_tick * 1e6,
                tick_offset_spread_ms,
                self._tick_reset_count,
                lidar_imu_rate_hz,
                lidar_imu_offset_sec,
                lidar_imu_offset_spread_ms,
                self._cloud_decode_error_count,
            )
        )

    def finalize(self) -> None:
        with self._lock:
            if self._finalized:
                return
            self._finalized = True
            self._accept_samples = False

        subscribers = [self._cloud_subscriber, self._lowstate_subscriber, self._lidar_imu_subscriber]
        for subscriber in subscribers:
            if subscriber is None:
                continue
            try:
                subscriber.Close()
            except Exception as error:
                self.get_logger().warning(f"Could not close a DDS diagnostic subscriber cleanly: {error}")

        self.log_summary(final=True)
        with self._lock:
            self._csv_file.flush()
            self._csv_file.close()
        self.get_logger().info(f"Timing measurement saved to {self._config.csv_path}")


def main() -> None:
    config = parse_args()
    ChannelFactoryInitialize, subscriber_cls, cloud_type, lowstate_type, lidar_imu_type = (
        import_raw_dds_dependencies()
    )
    ChannelFactoryInitialize(config.dds_domain_id, config.dds_interface)
    rclpy.init(args=None)
    node = TimeSyncDiagnostic(config, subscriber_cls, cloud_type, lowstate_type, lidar_imu_type)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node.finalize()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
