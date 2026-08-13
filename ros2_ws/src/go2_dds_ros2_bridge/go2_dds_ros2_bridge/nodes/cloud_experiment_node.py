#!/usr/bin/env python3

"""Inspect the raw lidar cloud timing and angular sampling pattern.

Measured hardware result (``/utlidar/time_corrected/cloud``, no range or Z
filter; stable 20-message windows):

    +--------------------------------------+----------------------+------------------------------------------+
    | Quantity                             | Measured value       | Interpretation                           |
    +======================================+======================+==========================================+
    | Raw PointCloud2 update rate          | 15.36 Hz (65.1 ms)   | One partial cloud event.                 |
    | Per-cloud point-time span            | 62.7--62.8 ms        | The cloud is a rolling, not instantaneous|
    |                                      |                      | acquisition.                             |
    | Opposing-fan signature (harmonic 1/2)| 0.10--0.14 / 0.95    | Two near-balanced fans 180 deg apart.    |
    | Fan centre sweep per cloud           | about 98 deg         | The opposing pair moves during a cloud.  |
    | Fan returned span per 7.8 ms slice   | about 18 deg         | Returned-point footprint, not beam spec. |
    | Front 180 deg return support/cloud   | about 120 deg        | One cloud is not a dense front scan.     |
    | 95% front return support             | 2 clouds, about 130 ms| Dense virtual front-scan cadence: 7.68 Hz|
    +--------------------------------------+----------------------+------------------------------------------+

Measured ground-relative obstacle-height envelope (stationary, level robot;
240 raw clouds; obstacle-height band z=0.10--1.50 m above ground; lidar origin
height 0.30 m; cap plot clipped at 10 m):

    +--------------------------------------+----------------------+------------------------------------------+
    | Base azimuth                         | Useful range cap     | Interpretation                           |
    +======================================+======================+==========================================+
    | Front, about -75 to +90 deg          | >= 10 m (clipped)    | Broad long-range front lobe.             |
    | Right side, -90 deg                  | about 6.3 m          | Useful lateral coverage.                 |
    | Left side, +90 deg                   | >= 10 m (clipped)    | Useful lateral coverage.                 |
    | Rear-side, +/-105 deg                | about 1.5--1.8 m     | Coverage falls rapidly behind the sides. |
    | Rear-side, +/-120 deg                | about 0.8 m          | Limited rear-side obstacle coverage.     |
    | Rear-side, +/-135 deg                | about 0.5--0.6 m     | Near-rear only.                          |
    | Direct rear, +/-180 deg              | about 0.4--0.5 m     | No meaningful long-range rear coverage.  |
    +--------------------------------------+----------------------+------------------------------------------+

The cap is a function of azimuth, not one constant rear range.  In Cartesian
ground-plane coordinates it is a broad front half-disc plus short rear-side
extensions.  Use an azimuth lookup/interpolation when reproducing it in
simulation.  Values reported as >=10 m only establish coverage to the chosen
plot cap; they are not a measured maximum sensor range.

The 7.68 Hz figure is an observed two-cloud *return-coverage assembly* rate,
not the lidar's raw output rate.  Empty angular bins contain no returned point;
they must not automatically be interpreted as free-space rays.

For the existing TemporalLidarScan policy, assemble two timestamped clouds,
deskew their points to one reference pose, then height-filter and bin the front
180 degrees.  The resulting virtual scan is acquired over roughly 130 ms.
Alternatively, retain every 15.36 Hz partial cloud only with a validity mask
and a policy trained for that partial representation.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from time import monotonic_ns

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField

from go2_dds_ros2_bridge.tf_utils import (
    DEFAULT_LIDAR_TF_RPY_DEG,
    DEFAULT_LIDAR_TF_XYZ,
    rotation_matrix_from_rpy_degrees,
)


DEFAULT_INPUT_TOPIC = "/utlidar/time_corrected/cloud"
DEFAULT_LOG_EVERY_MESSAGES = 20
DEFAULT_AZIMUTH_BINS = 256
DEFAULT_FOV_DEG = 180.0
DEFAULT_COVERAGE_FRACTION = 0.95
DEFAULT_MIN_RANGE_M = 0.0
DEFAULT_TIMESTAMP_SLICES = 8
DEFAULT_HEIGHT_ENVELOPE_MIN_GROUND_Z_M = 0.10
DEFAULT_HEIGHT_ENVELOPE_MAX_GROUND_Z_M = 1.50
DEFAULT_LIDAR_HEIGHT_ABOVE_GROUND_M = 0.30
DEFAULT_HEIGHT_ENVELOPE_MAX_RANGE_M = 10.0
DEFAULT_HEIGHT_ENVELOPE_AZIMUTH_BINS = 72
DEFAULT_HEIGHT_ENVELOPE_ACCUMULATION_MESSAGES = 120
DEFAULT_HEIGHT_ENVELOPE_PRINT_STEP_DEG = 15.0
DEFAULT_HEIGHT_ENVELOPE_PLOT_PATH = ""
DEFAULT_HEIGHT_ENVELOPE_RETURN_PLOT_SAMPLES_PER_CLOUD = 2000
TIMESTAMP_FIELD_NAMES = ("t", "timestamp", "timestamps", "time", "time_stamp")
POINT_FIELD_DTYPES = {
    PointField.INT8: np.dtype(np.int8),
    PointField.UINT8: np.dtype(np.uint8),
    PointField.INT16: np.dtype(np.int16),
    PointField.UINT16: np.dtype(np.uint16),
    PointField.INT32: np.dtype(np.int32),
    PointField.UINT32: np.dtype(np.uint32),
    PointField.FLOAT32: np.dtype(np.float32),
    PointField.FLOAT64: np.dtype(np.float64),
}
POINT_FIELD_TYPE_NAMES = {
    PointField.INT8: "int8",
    PointField.UINT8: "uint8",
    PointField.INT16: "int16",
    PointField.UINT16: "uint16",
    PointField.INT32: "int32",
    PointField.UINT32: "uint32",
    PointField.FLOAT32: "float32",
    PointField.FLOAT64: "float64",
}
TIMESTAMP_UNIT_SCALES_TO_SECONDS = {
    "seconds": 1.0,
    "milliseconds": 1e-3,
    "microseconds": 1e-6,
    "nanoseconds": 1e-9,
}


def _default_front_center_deg() -> float:
    """Express base_link's forward direction as an azimuth in the default lidar frame."""
    rotation_base_from_lidar = rotation_matrix_from_rpy_degrees(*DEFAULT_LIDAR_TF_RPY_DEG)
    forward_in_lidar = rotation_base_from_lidar.T @ np.array((1.0, 0.0, 0.0), dtype=np.float64)
    return math.degrees(math.atan2(float(forward_in_lidar[1]), float(forward_in_lidar[0])))


DEFAULT_FRONT_CENTER_DEG = _default_front_center_deg()


@dataclass(frozen=True)
class ExperimentConfig:
    input_topic: str
    log_every_messages: int
    azimuth_bins: int
    fov_deg: float
    front_center_deg: float
    coverage_fraction: float
    min_range_m: float
    timestamp_slices: int
    min_z_m: float
    max_z_m: float
    measure_height_envelope: bool
    height_envelope_min_ground_z_m: float
    height_envelope_max_ground_z_m: float
    lidar_height_above_ground_m: float
    height_envelope_max_range_m: float
    height_envelope_azimuth_bins: int
    height_envelope_accumulation_messages: int
    height_envelope_print_step_deg: float
    height_envelope_plot_path: str
    height_envelope_return_plot_samples_per_cloud: int


@dataclass(frozen=True)
class CloudInspection:
    point_count: int
    filtered_point_count: int
    azimuth_min_deg: float
    azimuth_max_deg: float
    front_coverage_mask: np.ndarray
    time_slice_first_harmonic: float
    time_slice_second_harmonic: float
    time_slice_folded_span_deg: float
    time_slice_fan_center_sweep_deg: float
    time_slice_count: int
    timestamp_field_name: str | None
    timestamp_type_name: str | None
    timestamp_stats: tuple[float, float, float] | None
    height_envelope_azimuth_rad: np.ndarray
    height_envelope_far_range_m: np.ndarray
    height_envelope_return_xyz_m: np.ndarray

    @property
    def front_bin_count(self) -> int:
        return int(np.count_nonzero(self.front_coverage_mask))

    @property
    def front_coverage_fraction(self) -> float:
        return float(self.front_bin_count / self.front_coverage_mask.size)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect PointCloud2 layout, point statistics, and timestamp-field behavior for a cloud topic."
        )
    )
    parser.add_argument(
        "--input-topic",
        type=str,
        default=DEFAULT_INPUT_TOPIC,
        help="PointCloud2 topic to inspect.",
    )
    parser.add_argument(
        "--log-every-messages",
        type=int,
        default=DEFAULT_LOG_EVERY_MESSAGES,
        help="Emit one summary log after this many received clouds.",
    )
    parser.add_argument(
        "--azimuth-bins",
        type=int,
        default=DEFAULT_AZIMUTH_BINS,
        help="Number of equal angular bins across --fov-deg used to measure per-packet coverage.",
    )
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=DEFAULT_FOV_DEG,
        help="Front field-of-view width, centred at --front-center-deg in the cloud frame.",
    )
    parser.add_argument(
        "--front-center-deg",
        type=float,
        default=DEFAULT_FRONT_CENTER_DEG,
        help=(
            "Azimuth of base_link forward expressed in the input cloud frame. The default (%.2f deg) "
            "is derived from this repository's default base_link <- utlidar_lidar rotation; override it "
            "when the deployed lidar TF differs."
            % DEFAULT_FRONT_CENTER_DEG
        ),
    )
    parser.add_argument(
        "--coverage-fraction",
        type=float,
        default=DEFAULT_COVERAGE_FRACTION,
        help="Fraction of front-FOV bins required to count a sequence of partial packets as one full scan.",
    )
    parser.add_argument(
        "--min-range-m",
        type=float,
        default=DEFAULT_MIN_RANGE_M,
        help="Ignore finite points closer than this XY range before measuring azimuth coverage.",
    )
    parser.add_argument(
        "--timestamp-slices",
        type=int,
        default=DEFAULT_TIMESTAMP_SLICES,
        help=(
            "Number of equal per-point-timestamp windows used for the single-fan versus "
            "antipodal-fan diagnostic."
        ),
    )
    parser.add_argument(
        "--min-z-m",
        type=float,
        default=-float("inf"),
        help="Ignore points below this Z value in the cloud frame; default keeps all heights.",
    )
    parser.add_argument(
        "--max-z-m",
        type=float,
        default=float("inf"),
        help="Ignore points above this Z value in the cloud frame; default keeps all heights.",
    )
    parser.add_argument(
        "--measure-height-envelope",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Estimate the base-frame obstacle-height visibility envelope from returned beam directions. "
            "This needs a stationary, level robot and a calibration scene with returns around the robot."
        ),
    )
    parser.add_argument(
        "--height-envelope-min-ground-z-m",
        type=float,
        default=DEFAULT_HEIGHT_ENVELOPE_MIN_GROUND_Z_M,
        help="Lower obstacle height above level ground used by the visibility-envelope measurement.",
    )
    parser.add_argument(
        "--height-envelope-max-ground-z-m",
        type=float,
        default=DEFAULT_HEIGHT_ENVELOPE_MAX_GROUND_Z_M,
        help="Upper obstacle height above level ground used by the visibility-envelope measurement.",
    )
    parser.add_argument(
        "--lidar-height-above-ground-m",
        type=float,
        default=DEFAULT_LIDAR_HEIGHT_ABOVE_GROUND_M,
        help=(
            "Measured vertical lidar-origin height above level ground while standing. The default is 0.30 m. "
            "It is combined with the configured base_link <- lidar extrinsic to derive the base-frame slab."
        ),
    )
    parser.add_argument(
        "--height-envelope-max-range-m",
        type=float,
        default=DEFAULT_HEIGHT_ENVELOPE_MAX_RANGE_M,
        help="Clip the reported base-frame visibility envelope at this horizontal range.",
    )
    parser.add_argument(
        "--height-envelope-azimuth-bins",
        type=int,
        default=DEFAULT_HEIGHT_ENVELOPE_AZIMUTH_BINS,
        help="Number of base-frame azimuth bins for the height-envelope measurement.",
    )
    parser.add_argument(
        "--height-envelope-accumulation-messages",
        type=int,
        default=DEFAULT_HEIGHT_ENVELOPE_ACCUMULATION_MESSAGES,
        help="Partial-cloud messages accumulated before emitting one height-envelope result.",
    )
    parser.add_argument(
        "--height-envelope-print-step-deg",
        type=float,
        default=DEFAULT_HEIGHT_ENVELOPE_PRINT_STEP_DEG,
        help="Angular spacing of the compact r_cap(azimuth) list in the height-envelope log.",
    )
    parser.add_argument(
        "--height-envelope-plot-path",
        type=str,
        default=DEFAULT_HEIGHT_ENVELOPE_PLOT_PATH,
        help=(
            "Write the accumulated base-frame coverage plot to this PNG/PDF/SVG path after each envelope "
            "window. Empty (the default) disables plotting and avoids requiring matplotlib."
        ),
    )
    parser.add_argument(
        "--height-envelope-return-plot-samples-per-cloud",
        type=int,
        default=DEFAULT_HEIGHT_ENVELOPE_RETURN_PLOT_SAMPLES_PER_CLOUD,
        help=(
            "Maximum height-filtered, base-frame returned points retained from each partial cloud for the "
            "coverage plot. Set to 0 to omit the measured-return layer."
        ),
    )
    non_ros_args = rclpy.utilities.remove_ros_args(args=sys.argv)[1:]
    args = parser.parse_args(non_ros_args)
    if args.azimuth_bins < 2:
        raise SystemExit("--azimuth-bins must be at least 2")
    if args.fov_deg <= 0.0 or args.fov_deg > 360.0:
        raise SystemExit("--fov-deg must be in (0, 360]")
    if args.coverage_fraction <= 0.0 or args.coverage_fraction > 1.0:
        raise SystemExit("--coverage-fraction must be in (0, 1]")
    if args.min_range_m < 0.0:
        raise SystemExit("--min-range-m must be non-negative")
    if args.timestamp_slices < 2:
        raise SystemExit("--timestamp-slices must be at least 2")
    if args.min_z_m > args.max_z_m:
        raise SystemExit("--min-z-m must not exceed --max-z-m")
    if args.height_envelope_min_ground_z_m >= args.height_envelope_max_ground_z_m:
        raise SystemExit("--height-envelope-min-ground-z-m must be smaller than --height-envelope-max-ground-z-m")
    if args.lidar_height_above_ground_m <= 0.0:
        raise SystemExit("--lidar-height-above-ground-m must be positive")
    if args.height_envelope_max_range_m <= 0.0:
        raise SystemExit("--height-envelope-max-range-m must be positive")
    if args.height_envelope_azimuth_bins < 8:
        raise SystemExit("--height-envelope-azimuth-bins must be at least 8")
    if args.height_envelope_accumulation_messages < 1:
        raise SystemExit("--height-envelope-accumulation-messages must be positive")
    if args.height_envelope_print_step_deg <= 0.0 or args.height_envelope_print_step_deg > 180.0:
        raise SystemExit("--height-envelope-print-step-deg must be in (0, 180]")
    if args.height_envelope_return_plot_samples_per_cloud < 0:
        raise SystemExit("--height-envelope-return-plot-samples-per-cloud must be non-negative")
    return ExperimentConfig(
        input_topic=args.input_topic,
        log_every_messages=max(args.log_every_messages, 1),
        azimuth_bins=int(args.azimuth_bins),
        fov_deg=float(args.fov_deg),
        front_center_deg=float(args.front_center_deg),
        coverage_fraction=float(args.coverage_fraction),
        min_range_m=float(args.min_range_m),
        timestamp_slices=int(args.timestamp_slices),
        min_z_m=float(args.min_z_m),
        max_z_m=float(args.max_z_m),
        measure_height_envelope=bool(args.measure_height_envelope),
        height_envelope_min_ground_z_m=float(args.height_envelope_min_ground_z_m),
        height_envelope_max_ground_z_m=float(args.height_envelope_max_ground_z_m),
        lidar_height_above_ground_m=float(args.lidar_height_above_ground_m),
        height_envelope_max_range_m=float(args.height_envelope_max_range_m),
        height_envelope_azimuth_bins=int(args.height_envelope_azimuth_bins),
        height_envelope_accumulation_messages=int(args.height_envelope_accumulation_messages),
        height_envelope_print_step_deg=float(args.height_envelope_print_step_deg),
        height_envelope_plot_path=str(args.height_envelope_plot_path),
        height_envelope_return_plot_samples_per_cloud=int(args.height_envelope_return_plot_samples_per_cloud),
    )


def dtype_from_fields(fields: list[PointField], point_step: int, is_bigendian: bool) -> np.dtype:
    names: list[str] = []
    formats: list[np.dtype] = []
    offsets: list[int] = []
    byte_order = ">" if is_bigendian else "<"

    for index, field in enumerate(fields):
        base_dtype = POINT_FIELD_DTYPES.get(field.datatype)
        if base_dtype is None:
            raise ValueError(f"Unsupported PointField datatype: {field.datatype}")
        field_dtype = base_dtype.newbyteorder(byte_order)
        if int(field.count) > 1:
            field_dtype = np.dtype((field_dtype, int(field.count)))
        names.append(field.name or f"unnamed_field_{index}")
        formats.append(field_dtype)
        offsets.append(int(field.offset))

    return np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": int(point_step),
        }
    )


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def describe_fields(fields: list[PointField]) -> str:
    return ", ".join(
        f"{field.name}:{POINT_FIELD_TYPE_NAMES.get(field.datatype, str(field.datatype))}"
        f"[{int(field.count)}]@{int(field.offset)}"
        for field in fields
    )


class CloudExperimentNode(Node):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__("go2_cloud_experiment")
        self._config = config
        self._subscription = self.create_subscription(
            PointCloud2,
            self._config.input_topic,
            self._cloud_callback,
            qos_profile_sensor_data,
        )
        self._message_count = 0
        self._schema_logged = False
        self._first_receive_ns: int | None = None
        self._last_receive_ns: int | None = None
        self._last_header_stamp_ns: int | None = None
        self._receive_intervals_sec: list[float] = []
        self._header_intervals_sec: list[float] = []
        self._point_counts: list[int] = []
        self._timestamp_field_name: str | None = None
        self._timestamp_min_values: list[float] = []
        self._timestamp_max_values: list[float] = []
        self._timestamp_spans: list[float] = []
        self._timestamp_sample_count = 0
        self._timestamp_field_type_name: str | None = None
        self._packet_azimuth_min_deg: list[float] = []
        self._packet_azimuth_max_deg: list[float] = []
        self._packet_front_coverage_fraction: list[float] = []
        self._packet_front_bin_counts: list[int] = []
        self._packet_time_slice_first_harmonic: list[float] = []
        self._packet_time_slice_second_harmonic: list[float] = []
        self._packet_time_slice_folded_spans_deg: list[float] = []
        self._packet_time_slice_fan_center_sweeps_deg: list[float] = []
        self._packet_time_slice_counts: list[int] = []
        self._height_filtered_point_counts: list[int] = []
        self._full_scan_spans_sec: list[float] = []
        self._full_scan_packet_counts: list[int] = []
        self._full_scan_periods_sec: list[float] = []
        self._active_coverage_mask: np.ndarray | None = None
        self._active_scan_start_ns: int | None = None
        self._active_scan_packet_count = 0
        self._last_full_scan_end_ns: int | None = None
        self._full_scan_count = 0
        self._coverage_target_bins = int(math.ceil(self._config.coverage_fraction * self._config.azimuth_bins))
        self._rotation_base_from_lidar = rotation_matrix_from_rpy_degrees(*DEFAULT_LIDAR_TF_RPY_DEG)
        self._lidar_origin_in_base_m = np.asarray(DEFAULT_LIDAR_TF_XYZ, dtype=np.float64)
        self._base_height_above_ground_m = (
            self._config.lidar_height_above_ground_m - float(self._lidar_origin_in_base_m[2])
        )
        self._height_envelope_min_base_z_m = (
            self._config.height_envelope_min_ground_z_m - self._base_height_above_ground_m
        )
        self._height_envelope_max_base_z_m = (
            self._config.height_envelope_max_ground_z_m - self._base_height_above_ground_m
        )
        self._height_envelope_far_range_m = np.full(
            self._config.height_envelope_azimuth_bins, np.nan, dtype=np.float64
        )
        self._height_envelope_ray_counts = np.zeros(self._config.height_envelope_azimuth_bins, dtype=np.int64)
        self._height_envelope_return_xyz_chunks: list[np.ndarray] = []
        self._height_envelope_message_count = 0

        self.get_logger().info(
            "Inspecting PointCloud2 topic '%s'. A summary will be logged every %d messages. "
            "Measuring %d front bins over %.1f deg centred at %.2f deg in cloud coordinates; "
            "a full scan needs %d bins (%.1f%%). "
            "The fan diagnostic uses %d per-point-time windows. "
            "Filtering in cloud frame: xy_range>=%.3fm, z=[%.3f, %.3f]m."
            % (
                self._config.input_topic,
                self._config.log_every_messages,
                self._config.azimuth_bins,
                self._config.fov_deg,
                self._config.front_center_deg,
                self._coverage_target_bins,
                100.0 * self._config.coverage_fraction,
                self._config.timestamp_slices,
                self._config.min_range_m,
                self._config.min_z_m,
                self._config.max_z_m,
            )
        )
        if self._config.measure_height_envelope:
            self.get_logger().info(
                "Height-envelope measurement enabled: transforming returned beam directions with the configured "
                "base_link <- utlidar_lidar extrinsic, then finding the farthest horizontal point of each beam "
                "within ground-relative z=[%.2f, %.2f]m (base_link z=[%.2f, %.2f]m using lidar height %.2fm). "
                "Accumulating %d clouds into %d azimuth bins, clipped at %.1fm; "
                "retaining up to %d actual returns/cloud for the optional plot."
                % (
                    self._config.height_envelope_min_ground_z_m,
                    self._config.height_envelope_max_ground_z_m,
                    self._height_envelope_min_base_z_m,
                    self._height_envelope_max_base_z_m,
                    self._config.lidar_height_above_ground_m,
                    self._config.height_envelope_accumulation_messages,
                    self._config.height_envelope_azimuth_bins,
                    self._config.height_envelope_max_range_m,
                    self._config.height_envelope_return_plot_samples_per_cloud,
                )
            )

    def _cloud_callback(self, msg: PointCloud2) -> None:
        receive_ns = monotonic_ns()
        header_stamp_ns = stamp_to_ns(msg.header.stamp)

        if self._first_receive_ns is None:
            self._first_receive_ns = receive_ns
        if self._last_receive_ns is not None:
            self._receive_intervals_sec.append((receive_ns - self._last_receive_ns) * 1e-9)
        if self._last_header_stamp_ns is not None and header_stamp_ns > self._last_header_stamp_ns:
            self._header_intervals_sec.append((header_stamp_ns - self._last_header_stamp_ns) * 1e-9)
        self._last_receive_ns = receive_ns
        self._last_header_stamp_ns = header_stamp_ns

        self._message_count += 1

        if not self._schema_logged:
            self._schema_logged = True
            self.get_logger().info(
                "Cloud schema: frame_id='%s', width=%d, height=%d, point_step=%d, row_step=%d, "
                "is_bigendian=%s, fields=[%s]"
                % (
                    msg.header.frame_id,
                    int(msg.width),
                    int(msg.height),
                    int(msg.point_step),
                    int(msg.row_step),
                    bool(msg.is_bigendian),
                    describe_fields(msg.fields),
                )
            )

        try:
            inspection = self._inspect_cloud(msg)
        except ValueError as error:
            self.get_logger().error(f"Failed to inspect PointCloud2 message: {error}")
            return

        self._point_counts.append(inspection.point_count)
        self._height_filtered_point_counts.append(inspection.filtered_point_count)
        self._packet_azimuth_min_deg.append(inspection.azimuth_min_deg)
        self._packet_azimuth_max_deg.append(inspection.azimuth_max_deg)
        self._packet_front_bin_counts.append(inspection.front_bin_count)
        self._packet_front_coverage_fraction.append(inspection.front_coverage_fraction)
        self._packet_time_slice_first_harmonic.append(inspection.time_slice_first_harmonic)
        self._packet_time_slice_second_harmonic.append(inspection.time_slice_second_harmonic)
        self._packet_time_slice_folded_spans_deg.append(inspection.time_slice_folded_span_deg)
        self._packet_time_slice_fan_center_sweeps_deg.append(inspection.time_slice_fan_center_sweep_deg)
        self._packet_time_slice_counts.append(inspection.time_slice_count)
        sample_time_ns = header_stamp_ns if header_stamp_ns > 0 else receive_ns
        self._update_full_scan_measurement(inspection.front_coverage_mask, sample_time_ns)
        self._update_height_envelope(
            inspection.height_envelope_azimuth_rad,
            inspection.height_envelope_far_range_m,
            inspection.height_envelope_return_xyz_m,
        )

        timestamp_field_name = inspection.timestamp_field_name
        timestamp_type_name = inspection.timestamp_type_name
        timestamp_stats = inspection.timestamp_stats
        if timestamp_field_name is not None and timestamp_stats is not None:
            self._timestamp_field_name = timestamp_field_name
            self._timestamp_field_type_name = timestamp_type_name
            min_timestamp, max_timestamp, span_timestamp = timestamp_stats
            self._timestamp_min_values.append(min_timestamp)
            self._timestamp_max_values.append(max_timestamp)
            self._timestamp_spans.append(span_timestamp)
            self._timestamp_sample_count += 1

        if self._message_count % self._config.log_every_messages == 0:
            self._log_summary(msg.header.frame_id)
            self._reset_window()

    def _inspect_cloud(self, msg: PointCloud2) -> CloudInspection:
        cloud_dtype = dtype_from_fields(msg.fields, int(msg.point_step), bool(msg.is_bigendian))
        width = int(msg.width)
        height = int(msg.height)
        point_step = int(msg.point_step)
        row_step = int(msg.row_step) if int(msg.row_step) > 0 else width * point_step
        packed_row_step = width * point_step

        if row_step < packed_row_step:
            raise ValueError(f"row_step={row_step} is smaller than width * point_step={packed_row_step}")

        expected_buffer_size = row_step * height
        if len(msg.data) < expected_buffer_size:
            raise ValueError(
                f"PointCloud2 data buffer is too small: len(data)={len(msg.data)} expected>={expected_buffer_size}"
            )

        cloud = np.ndarray(
            shape=(height, width),
            dtype=cloud_dtype,
            buffer=msg.data,
            strides=(row_step, point_step),
        )
        field_names = cloud.dtype.names or ()
        for field_name in ("x", "y", "z"):
            if field_name not in field_names:
                raise ValueError(f"Point cloud is missing required '{field_name}' field")

        points = np.column_stack(
            (
                np.asarray(cloud["x"], dtype=np.float64).reshape(-1),
                np.asarray(cloud["y"], dtype=np.float64).reshape(-1),
                np.asarray(cloud["z"], dtype=np.float64).reshape(-1),
            )
        )
        mask = np.all(np.isfinite(points), axis=1)
        point_count = int(np.count_nonzero(mask))

        xy_range = np.hypot(points[:, 0], points[:, 1])
        selected_mask = (
            mask
            & (xy_range >= self._config.min_range_m)
            & (points[:, 2] >= self._config.min_z_m)
            & (points[:, 2] <= self._config.max_z_m)
        )
        selected_points = points[selected_mask]
        finite_points = points[mask]
        height_envelope_azimuth_rad, height_envelope_far_range_m = self._height_envelope_samples(finite_points)
        height_envelope_return_xyz_m = self._height_envelope_return_samples(finite_points)
        azimuth_min_deg = float("nan")
        azimuth_max_deg = float("nan")
        front_coverage_mask = np.zeros(self._config.azimuth_bins, dtype=bool)
        if selected_points.size > 0:
            azimuth_rad = np.arctan2(selected_points[:, 1], selected_points[:, 0])
            azimuth_deg = np.degrees(azimuth_rad)
            azimuth_min_deg = float(np.min(azimuth_deg))
            azimuth_max_deg = float(np.max(azimuth_deg))
            half_fov_rad = math.radians(self._config.fov_deg) * 0.5
            front_center_rad = math.radians(self._config.front_center_deg)
            relative_front_azimuth = (azimuth_rad - front_center_rad + math.pi) % (2.0 * math.pi) - math.pi
            front_mask = (relative_front_azimuth >= -half_fov_rad) & (relative_front_azimuth <= half_fov_rad)
            if np.any(front_mask):
                front_positions = (relative_front_azimuth[front_mask] + half_fov_rad) / (2.0 * half_fov_rad)
                front_bin_indices = np.minimum(
                    (front_positions * self._config.azimuth_bins).astype(np.int64), self._config.azimuth_bins - 1
                )
                front_coverage_mask[np.unique(front_bin_indices)] = True

        timestamp_field_name = next((name for name in TIMESTAMP_FIELD_NAMES if name in field_names), None)
        if timestamp_field_name is None:
            return CloudInspection(
                point_count=point_count,
                filtered_point_count=int(selected_points.shape[0]),
                azimuth_min_deg=azimuth_min_deg,
                azimuth_max_deg=azimuth_max_deg,
                front_coverage_mask=front_coverage_mask,
                time_slice_first_harmonic=float("nan"),
                time_slice_second_harmonic=float("nan"),
                time_slice_folded_span_deg=float("nan"),
                time_slice_fan_center_sweep_deg=float("nan"),
                time_slice_count=0,
                timestamp_field_name=None,
                timestamp_type_name=None,
                timestamp_stats=None,
                height_envelope_azimuth_rad=height_envelope_azimuth_rad,
                height_envelope_far_range_m=height_envelope_far_range_m,
                height_envelope_return_xyz_m=height_envelope_return_xyz_m,
            )

        raw_timestamps = np.asarray(cloud[timestamp_field_name], dtype=np.float64).reshape(-1)
        timestamp_mask = mask & np.isfinite(raw_timestamps)
        timestamps = raw_timestamps[timestamp_mask]
        if timestamps.size == 0:
            timestamp_stats = None
        else:
            timestamp_stats = (
                float(np.min(timestamps)),
                float(np.max(timestamps)),
                float(np.max(timestamps) - np.min(timestamps)),
            )

        selected_timestamps = raw_timestamps[selected_mask]
        selected_timestamp_mask = np.isfinite(selected_timestamps)
        if selected_points.size > 0 and np.any(selected_timestamp_mask):
            selected_azimuth_rad = np.arctan2(selected_points[:, 1], selected_points[:, 0])
            (
                first_harmonic,
                second_harmonic,
                folded_span_deg,
                fan_center_sweep_deg,
                time_slice_count,
            ) = self._time_slice_azimuth_signature(
                selected_azimuth_rad[selected_timestamp_mask],
                selected_timestamps[selected_timestamp_mask],
            )
        else:
            first_harmonic = float("nan")
            second_harmonic = float("nan")
            folded_span_deg = float("nan")
            fan_center_sweep_deg = float("nan")
            time_slice_count = 0
        return CloudInspection(
            point_count=point_count,
            filtered_point_count=int(selected_points.shape[0]),
            azimuth_min_deg=azimuth_min_deg,
            azimuth_max_deg=azimuth_max_deg,
            front_coverage_mask=front_coverage_mask,
            time_slice_first_harmonic=first_harmonic,
            time_slice_second_harmonic=second_harmonic,
            time_slice_folded_span_deg=folded_span_deg,
            time_slice_fan_center_sweep_deg=fan_center_sweep_deg,
            time_slice_count=time_slice_count,
            timestamp_field_name=timestamp_field_name,
            timestamp_type_name=self._field_type_name(msg, timestamp_field_name),
            timestamp_stats=timestamp_stats,
            height_envelope_azimuth_rad=height_envelope_azimuth_rad,
            height_envelope_far_range_m=height_envelope_far_range_m,
            height_envelope_return_xyz_m=height_envelope_return_xyz_m,
        )

    def _height_envelope_samples(self, points_lidar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return body azimuth and far useful range for each observed beam direction.

        A returned point gives us a sampled ray direction even when its return was
        from the ground.  With the static lidar extrinsic, its height at horizontal
        distance ``rho`` from the lidar is ``origin_z + dz / hypot(dx, dy) * rho``.
        Intersecting that line with the configured obstacle-height slab gives the
        distance interval at which that ray could observe an obstacle represented
        by the policy.  This describes the tilted sensor geometry rather than the
        particular scene's obstacle layout.
        """
        if not self._config.measure_height_envelope or points_lidar.size == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

        directions_base = points_lidar @ self._rotation_base_from_lidar.T
        direction_norms = np.linalg.norm(directions_base, axis=1)
        valid_directions = direction_norms > 1e-6
        if not np.any(valid_directions):
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        directions_base = directions_base[valid_directions] / direction_norms[valid_directions, None]

        horizontal_norms = np.hypot(directions_base[:, 0], directions_base[:, 1])
        valid_horizontal = horizontal_norms > 1e-6
        if not np.any(valid_horizontal):
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        directions_base = directions_base[valid_horizontal]
        horizontal_norms = horizontal_norms[valid_horizontal]

        slopes = directions_base[:, 2] / horizontal_norms
        origin_z_m = float(self._lidar_origin_in_base_m[2])
        lower_z_m = self._height_envelope_min_base_z_m
        upper_z_m = self._height_envelope_max_base_z_m
        max_range_m = self._config.height_envelope_max_range_m
        range_low_m = np.empty_like(slopes)
        range_high_m = np.empty_like(slopes)
        nearly_level = np.abs(slopes) < 1e-8
        within_slab = (origin_z_m >= lower_z_m) & (origin_z_m <= upper_z_m)
        range_low_m[nearly_level] = 0.0 if within_slab else max_range_m + 1.0
        range_high_m[nearly_level] = max_range_m if within_slab else -1.0
        nonlevel = ~nearly_level
        lower_crossing_m = (lower_z_m - origin_z_m) / slopes[nonlevel]
        upper_crossing_m = (upper_z_m - origin_z_m) / slopes[nonlevel]
        range_low_m[nonlevel] = np.minimum(lower_crossing_m, upper_crossing_m)
        range_high_m[nonlevel] = np.maximum(lower_crossing_m, upper_crossing_m)
        range_low_m = np.maximum(range_low_m, 0.0)
        range_high_m = np.minimum(range_high_m, max_range_m)
        valid_interval = range_high_m >= range_low_m
        if not np.any(valid_interval):
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

        # Attribute the far endpoint to the azimuth around base_link, not around
        # the offset lidar.  The difference matters for short rear-side coverage.
        horizontal_direction = directions_base[valid_interval, :2] / horizontal_norms[valid_interval, None]
        far_xy_m = self._lidar_origin_in_base_m[:2] + horizontal_direction * range_high_m[valid_interval, None]
        far_base_range_m = np.hypot(far_xy_m[:, 0], far_xy_m[:, 1])
        far_base_azimuth_rad = np.arctan2(far_xy_m[:, 1], far_xy_m[:, 0])
        return far_base_azimuth_rad, far_base_range_m

    def _height_envelope_return_samples(self, points_lidar: np.ndarray) -> np.ndarray:
        """Return actual points for the plot, including ground-level returns.

        This is deliberately separate from ``_height_envelope_samples``. The
        latter answers where *a beam direction could be useful*; this function
        answers where an object actually produced a return in this recording.
        It intentionally does *not* apply the obstacle-height slab: that makes
        the plot agree with RViz and exposes whether an apparent rear return is
        a ground hit or an obstacle-height hit.
        """
        max_samples = self._config.height_envelope_return_plot_samples_per_cloud
        if not self._config.measure_height_envelope or max_samples == 0 or points_lidar.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        points_base = points_lidar @ self._rotation_base_from_lidar.T + self._lidar_origin_in_base_m
        xy_range_m = np.hypot(points_base[:, 0], points_base[:, 1])
        return_xyz_m = points_base[xy_range_m <= self._config.height_envelope_max_range_m].copy()
        # The XY coordinates remain in base_link; the plotted colour is made
        # ground-relative so it can be directly compared with obstacle height.
        return_xyz_m[:, 2] += self._base_height_above_ground_m
        if return_xyz_m.shape[0] <= max_samples:
            return return_xyz_m
        sample_indices = np.linspace(0, return_xyz_m.shape[0] - 1, num=max_samples, dtype=np.int64)
        return return_xyz_m[sample_indices]

    def _time_slice_azimuth_signature(
        self, azimuth_rad: np.ndarray, timestamps: np.ndarray
    ) -> tuple[float, float, float, float, int]:
        """Measure whether simultaneous returns form one fan or opposing fans.

        In each short time slice, the first circular harmonic is near one for one
        narrow fan.  Two similarly populated fans 180 degrees apart cancel that
        harmonic but retain a high second harmonic.  This remains a diagnostic:
        scene geometry and return dropouts can also reduce either value.
        """
        timestamp_min = float(np.min(timestamps))
        timestamp_max = float(np.max(timestamps))
        timestamp_span = timestamp_max - timestamp_min
        if timestamp_span <= 0.0:
            return float("nan"), float("nan"), float("nan"), float("nan"), 0

        normalized_time = (timestamps - timestamp_min) / timestamp_span
        slice_indices = np.minimum(
            (normalized_time * self._config.timestamp_slices).astype(np.int64),
            self._config.timestamp_slices - 1,
        )
        first_harmonics: list[float] = []
        second_harmonics: list[float] = []
        folded_spans_deg: list[float] = []
        slice_time_centers: list[float] = []
        double_angle_centers: list[float] = []
        for slice_index in range(self._config.timestamp_slices):
            slice_azimuth = azimuth_rad[slice_indices == slice_index]
            if slice_azimuth.size < 20:
                continue
            first_harmonics.append(float(np.abs(np.mean(np.exp(1j * slice_azimuth)))))
            double_angle_mean = np.mean(np.exp(2j * slice_azimuth))
            second_harmonics.append(float(np.abs(double_angle_mean)))
            # Folding theta modulo pi places two opposing fans on top of one
            # another.  The robust 5--95% span then estimates one fan's width.
            fan_center_rad = 0.5 * float(np.angle(double_angle_mean))
            folded_delta_rad = 0.5 * np.angle(np.exp(2j * (slice_azimuth - fan_center_rad)))
            folded_spans_deg.append(
                float(np.degrees(np.percentile(folded_delta_rad, 95.0) - np.percentile(folded_delta_rad, 5.0)))
            )
            slice_time_centers.append(float(np.mean(normalized_time[slice_indices == slice_index])))
            double_angle_centers.append(float(np.angle(double_angle_mean)))

        if not first_harmonics:
            return float("nan"), float("nan"), float("nan"), float("nan"), 0
        if len(double_angle_centers) < 2:
            fan_center_sweep_deg = float("nan")
        else:
            # A paired fan orientation is periodic over pi, hence the doubled
            # angle.  Unwrap it over time and fit its sweep across one cloud.
            unwrapped_double_angles = np.unwrap(np.asarray(double_angle_centers))
            slope_rad_per_cloud = float(np.polyfit(slice_time_centers, unwrapped_double_angles, deg=1)[0])
            fan_center_sweep_deg = abs(math.degrees(0.5 * slope_rad_per_cloud))
        return (
            float(np.median(np.asarray(first_harmonics))),
            float(np.median(np.asarray(second_harmonics))),
            float(np.median(np.asarray(folded_spans_deg))),
            fan_center_sweep_deg,
            len(first_harmonics),
        )

    def _update_full_scan_measurement(self, packet_coverage_mask: np.ndarray, sample_time_ns: int) -> None:
        if not np.any(packet_coverage_mask):
            return
        if self._active_coverage_mask is None:
            self._active_coverage_mask = packet_coverage_mask.copy()
            self._active_scan_start_ns = sample_time_ns
            self._active_scan_packet_count = 1
        else:
            self._active_coverage_mask |= packet_coverage_mask
            self._active_scan_packet_count += 1

        if int(np.count_nonzero(self._active_coverage_mask)) < self._coverage_target_bins:
            return

        start_ns = self._active_scan_start_ns
        if start_ns is not None and sample_time_ns >= start_ns:
            self._full_scan_spans_sec.append((sample_time_ns - start_ns) * 1e-9)
        self._full_scan_packet_counts.append(self._active_scan_packet_count)
        if self._last_full_scan_end_ns is not None and sample_time_ns > self._last_full_scan_end_ns:
            self._full_scan_periods_sec.append((sample_time_ns - self._last_full_scan_end_ns) * 1e-9)
        self._last_full_scan_end_ns = sample_time_ns
        self._full_scan_count += 1
        self._active_coverage_mask = None
        self._active_scan_start_ns = None
        self._active_scan_packet_count = 0

    def _update_height_envelope(
        self,
        azimuth_rad: np.ndarray,
        far_range_m: np.ndarray,
        return_xyz_m: np.ndarray,
    ) -> None:
        """Accumulate one stationary calibration's body-frame ray-height envelope."""
        if not self._config.measure_height_envelope:
            return
        self._height_envelope_message_count += 1
        if azimuth_rad.size:
            bin_positions = (azimuth_rad + math.pi) / (2.0 * math.pi)
            bin_indices = np.minimum(
                (bin_positions * self._config.height_envelope_azimuth_bins).astype(np.int64),
                self._config.height_envelope_azimuth_bins - 1,
            )
            finite = np.isfinite(far_range_m)
            bin_indices = bin_indices[finite]
            far_range_m = far_range_m[finite]
            if bin_indices.size:
                current = np.nan_to_num(self._height_envelope_far_range_m, nan=-float("inf"))
                np.maximum.at(current, bin_indices, far_range_m)
                self._height_envelope_far_range_m = np.where(np.isfinite(current), current, np.nan)
                np.add.at(self._height_envelope_ray_counts, bin_indices, 1)
        if return_xyz_m.size:
            self._height_envelope_return_xyz_chunks.append(return_xyz_m)

        if self._height_envelope_message_count < self._config.height_envelope_accumulation_messages:
            return
        self.get_logger().info(self._height_envelope_summary())
        self._save_height_envelope_plot()
        self._height_envelope_far_range_m.fill(np.nan)
        self._height_envelope_ray_counts.fill(0)
        self._height_envelope_return_xyz_chunks = []
        self._height_envelope_message_count = 0

    def _height_envelope_summary(self) -> str:
        bin_count = self._config.height_envelope_azimuth_bins
        bin_centers_rad = -math.pi + (np.arange(bin_count, dtype=np.float64) + 0.5) * (2.0 * math.pi / bin_count)
        bin_centers_deg = np.degrees(bin_centers_rad)
        valid = np.isfinite(self._height_envelope_far_range_m)

        def zone_summary(name: str, zone_mask: np.ndarray) -> str:
            values = self._height_envelope_far_range_m[zone_mask & valid]
            available_bins = int(np.count_nonzero(zone_mask & valid))
            total_bins = int(np.count_nonzero(zone_mask))
            if values.size == 0:
                return f"{name}=unobserved(0/{total_bins})"
            p05, p50, p95 = np.percentile(values, (5.0, 50.0, 95.0))
            return "%s_r_cap_p05/p50/p95=%.2f/%.2f/%.2fm(%d/%d)" % (
                name,
                p05,
                p50,
                p95,
                available_bins,
                total_bins,
            )

        # base_link convention: +x is front and +y is left.
        front = np.abs(bin_centers_deg) <= 90.0
        left_rear = (bin_centers_deg > 90.0) & (bin_centers_deg < 180.0)
        rear = np.abs(bin_centers_deg) >= 150.0
        right_rear = (bin_centers_deg < -90.0) & (bin_centers_deg > -180.0)
        samples_per_observed_bin = self._height_envelope_ray_counts[valid]
        mean_samples = self._mean_or_nan(samples_per_observed_bin.astype(np.float64).tolist())

        output_angles_deg = np.arange(-180.0, 180.0, self._config.height_envelope_print_step_deg)
        sampled_caps: list[str] = []
        for requested_angle_deg in output_angles_deg:
            circular_delta_deg = (bin_centers_deg - requested_angle_deg + 180.0) % 360.0 - 180.0
            index = int(np.argmin(np.abs(circular_delta_deg)))
            cap_m = self._height_envelope_far_range_m[index]
            value = "--" if not math.isfinite(float(cap_m)) else f"{cap_m:.2f}"
            sampled_caps.append(f"{requested_angle_deg:+.0f}:{value}")

        return (
            "Height-envelope summary: clouds=%d ground_z=[%.2f,%.2f]m base_z=[%.2f,%.2f]m "
            "lidar_height_above_ground=%.2fm max_range=%.1fm "
            "observed_azimuth_bins=%d/%d ray_samples_per_observed_bin=%.0f "
            "%s %s %s %s r_cap_m_by_base_azimuth_deg={%s}"
            % (
                self._height_envelope_message_count,
                self._config.height_envelope_min_ground_z_m,
                self._config.height_envelope_max_ground_z_m,
                self._height_envelope_min_base_z_m,
                self._height_envelope_max_base_z_m,
                self._config.lidar_height_above_ground_m,
                self._config.height_envelope_max_range_m,
                int(np.count_nonzero(valid)),
                bin_count,
                mean_samples,
                zone_summary("front", front),
                zone_summary("left_rear", left_rear),
                zone_summary("rear", rear),
                zone_summary("right_rear", right_rear),
                ",".join(sampled_caps),
            )
        )

    def _save_height_envelope_plot(self) -> None:
        """Save a top-down coverage boundary and an azimuth-versus-range view.

        Matplotlib is intentionally imported only here: timing-only uses of this
        node should not acquire an optional plotting dependency.
        """
        if not self._config.height_envelope_plot_path:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            self.get_logger().error(
                "Cannot write --height-envelope-plot-path because matplotlib is unavailable. "
                "Install python3-matplotlib or omit this option."
            )
            return

        path = Path(self._config.height_envelope_plot_path).expanduser()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self.get_logger().error(f"Cannot create height-envelope plot directory '{path.parent}': {error}")
            return

        bin_count = self._config.height_envelope_azimuth_bins
        azimuth_rad = -math.pi + (np.arange(bin_count, dtype=np.float64) + 0.5) * (2.0 * math.pi / bin_count)
        caps_m = self._height_envelope_far_range_m.copy()
        valid = np.isfinite(caps_m)
        return_xyz_m = (
            np.concatenate(self._height_envelope_return_xyz_chunks, axis=0)
            if self._height_envelope_return_xyz_chunks
            else np.empty((0, 3), dtype=np.float64)
        )
        if not np.any(valid) and not return_xyz_m.size:
            self.get_logger().warning("Height-envelope plot was not written: no valid beam directions or returns.")
            return

        figure, (top_down_axis, cap_axis) = plt.subplots(1, 2, figsize=(14.0, 6.5), constrained_layout=True)
        # Plot only contiguous observed sections; joining gaps would falsely
        # imply coverage where no returned ray direction was sampled.
        changes = np.flatnonzero(np.diff(valid.astype(np.int8)) != 0) + 1
        sections = np.split(np.arange(bin_count), changes)
        for section in sections:
            if section.size == 0 or not valid[section[0]]:
                continue
            section_angles = azimuth_rad[section]
            section_caps = caps_m[section]
            boundary_x_m = section_caps * np.cos(section_angles)
            boundary_y_m = section_caps * np.sin(section_angles)
            top_down_axis.plot(boundary_x_m, boundary_y_m, color="tab:blue", linewidth=2.5)
            top_down_axis.fill(
                np.concatenate(([0.0], boundary_x_m, [0.0])),
                np.concatenate(([0.0], boundary_y_m, [0.0])),
                color="tab:blue",
                alpha=0.18,
            )

        if return_xyz_m.size:
            return_scatter = top_down_axis.scatter(
                return_xyz_m[:, 0],
                return_xyz_m[:, 1],
                s=1.5,
                c=return_xyz_m[:, 2],
                cmap="coolwarm",
                alpha=0.18,
                rasterized=True,
                label="all actual returns",
            )
            colorbar = figure.colorbar(return_scatter, ax=top_down_axis, shrink=0.72)
            colorbar.set_label("return height above ground (m)")

        body_length_m, body_width_m = 0.70, 0.35
        top_down_axis.add_patch(
            plt.Rectangle(
                (-0.5 * body_length_m, -0.5 * body_width_m),
                body_length_m,
                body_width_m,
                facecolor="0.2",
                edgecolor="black",
                alpha=0.75,
                label="robot body",
            )
        )
        top_down_axis.plot(
            self._lidar_origin_in_base_m[0],
            self._lidar_origin_in_base_m[1],
            marker="*",
            markersize=12,
            color="tab:red",
            label="lidar origin",
        )
        top_down_axis.annotate(
            "front (+x)",
            xy=(self._config.height_envelope_max_range_m * 0.9, 0.0),
            xytext=(self._config.height_envelope_max_range_m * 0.5, 0.4),
            arrowprops={"arrowstyle": "->"},
        )
        top_down_axis.set_aspect("equal", adjustable="box")
        top_down_axis.set_xlim(-self._config.height_envelope_max_range_m, self._config.height_envelope_max_range_m)
        top_down_axis.set_ylim(-self._config.height_envelope_max_range_m, self._config.height_envelope_max_range_m)
        top_down_axis.set_xlabel("base x (m): forward")
        top_down_axis.set_ylabel("base y (m): left")
        top_down_axis.set_title(
            "Coverage envelope + all actual returns\n(top-down ground plane; colour is height, points are not free-space rays)"
        )
        top_down_axis.grid(True, alpha=0.3)
        top_down_axis.legend(loc="upper left")

        cap_axis.plot(np.degrees(azimuth_rad[valid]), caps_m[valid], marker=".", color="tab:blue")
        cap_axis.axvspan(-90.0, 90.0, color="tab:green", alpha=0.08, label="front half-plane")
        cap_axis.axvspan(-180.0, -90.0, color="tab:orange", alpha=0.08, label="right rear")
        cap_axis.axvspan(90.0, 180.0, color="tab:red", alpha=0.08, label="left rear")
        cap_axis.set_xlim(-180.0, 180.0)
        cap_axis.set_ylim(0.0, self._config.height_envelope_max_range_m * 1.05)
        cap_axis.set_xticks(np.arange(-180.0, 181.0, 45.0))
        cap_axis.set_xlabel("base azimuth (deg): 0 front, +90 left, -90 right, ±180 rear")
        cap_axis.set_ylabel("maximum useful obstacle range r_cap (m)")
        cap_axis.set_title(
            "Range cap where a sampled beam lies at ground height z=[%.2f, %.2f] m"
            % (
                self._config.height_envelope_min_ground_z_m,
                self._config.height_envelope_max_ground_z_m,
            )
        )
        cap_axis.grid(True, alpha=0.3)
        cap_axis.legend(loc="upper center")
        figure.suptitle(
            "Tilted lidar coverage; source=%d partial clouds, lidar origin=(%.3f, %.3f, %.3f) m in base_link"
            % (
                self._height_envelope_message_count,
                self._lidar_origin_in_base_m[0],
                self._lidar_origin_in_base_m[1],
                self._lidar_origin_in_base_m[2],
            )
        )
        try:
            figure.savefig(path, dpi=180)
        except OSError as error:
            self.get_logger().error(f"Failed to write height-envelope plot '{path}': {error}")
        else:
            self.get_logger().info(f"Wrote height-envelope plot: {path}")
        finally:
            plt.close(figure)

    @staticmethod
    def _field_type_name(msg: PointCloud2, field_name: str) -> str | None:
        field = next((item for item in msg.fields if item.name == field_name), None)
        if field is None:
            return None
        return POINT_FIELD_TYPE_NAMES.get(field.datatype, str(field.datatype))

    def _log_summary(self, frame_id: str) -> None:
        mean_receive_interval = self._mean_or_nan(self._receive_intervals_sec)
        mean_header_interval = self._mean_or_nan(self._header_intervals_sec)
        mean_receive_hz = self._hz_from_interval(mean_receive_interval)
        mean_header_hz = self._hz_from_interval(mean_header_interval)
        mean_point_count = self._mean_or_nan(self._point_counts)
        mean_filtered_point_count = self._mean_or_nan(self._height_filtered_point_counts)
        min_azimuth_deg = self._min_or_nan(self._packet_azimuth_min_deg)
        max_azimuth_deg = self._max_or_nan(self._packet_azimuth_max_deg)
        mean_front_bins = self._mean_or_nan(self._packet_front_bin_counts)
        median_front_coverage = self._percentile_or_nan(self._packet_front_coverage_fraction, 50.0)
        p95_front_coverage = self._percentile_or_nan(self._packet_front_coverage_fraction, 95.0)
        median_first_harmonic = self._percentile_or_nan(self._packet_time_slice_first_harmonic, 50.0)
        median_second_harmonic = self._percentile_or_nan(self._packet_time_slice_second_harmonic, 50.0)
        median_folded_span_deg = self._percentile_or_nan(self._packet_time_slice_folded_spans_deg, 50.0)
        median_fan_center_sweep_deg = self._percentile_or_nan(
            self._packet_time_slice_fan_center_sweeps_deg, 50.0
        )
        median_time_slice_count = self._percentile_or_nan(self._packet_time_slice_counts, 50.0)

        summary = (
            "Cloud experiment summary: messages=%d frame_id='%s' avg_points=%.1f filtered_points=%.1f "
            "recv_interval=%.6fs (%.2f Hz) header_interval=%.6fs (%.2f Hz) "
            "packet_azimuth_deg=[%.1f, %.1f] front_return_bins=%.1f/%d return_coverage_p50/p95=%.1f%%/%.1f%% "
            "time_slice_harmonic_1/2=%.3f/%.3f folded_fan_span_p05-p95=%.1fdeg "
            "fan_center_sweep_per_cloud=%.1fdeg slices=%.0f"
            % (
                self._message_count,
                frame_id,
                mean_point_count,
                mean_filtered_point_count,
                mean_receive_interval,
                mean_receive_hz,
                mean_header_interval,
                mean_header_hz,
                min_azimuth_deg,
                max_azimuth_deg,
                mean_front_bins,
                self._config.azimuth_bins,
                100.0 * median_front_coverage,
                100.0 * p95_front_coverage,
                median_first_harmonic,
                median_second_harmonic,
                median_folded_span_deg,
                median_fan_center_sweep_deg,
                median_time_slice_count,
            )
        )

        full_scan_summary = self._full_scan_summary()
        if full_scan_summary:
            summary += " " + full_scan_summary

        if self._timestamp_field_name is None:
            self.get_logger().info(summary + " timestamp_field=<none>")
            return

        if not self._timestamp_spans:
            self.get_logger().info(
                summary
                + " timestamp_field='%s' type=%s no_finite_timestamp_values=true"
                % (self._timestamp_field_name, self._timestamp_field_type_name)
            )
            return

        mean_min = self._mean_or_nan(self._timestamp_min_values)
        mean_max = self._mean_or_nan(self._timestamp_max_values)
        mean_span = self._mean_or_nan(self._timestamp_spans)
        inferred_unit = self._infer_timestamp_unit(mean_span, mean_header_interval)
        self.get_logger().info(
            summary
            + " timestamp_field='%s' type=%s min=%.9f max=%.9f span=%.9f inferred_unit=%s"
            % (
                self._timestamp_field_name,
                self._timestamp_field_type_name,
                mean_min,
                mean_max,
                mean_span,
                inferred_unit,
            )
        )

    @staticmethod
    def _mean_or_nan(values: list[float]) -> float:
        if not values:
            return float("nan")
        return float(np.mean(np.asarray(values, dtype=np.float64)))

    @staticmethod
    def _min_or_nan(values: list[float]) -> float:
        finite_values = np.asarray(values, dtype=np.float64)
        finite_values = finite_values[np.isfinite(finite_values)]
        return float(np.min(finite_values)) if finite_values.size > 0 else float("nan")

    @staticmethod
    def _max_or_nan(values: list[float]) -> float:
        finite_values = np.asarray(values, dtype=np.float64)
        finite_values = finite_values[np.isfinite(finite_values)]
        return float(np.max(finite_values)) if finite_values.size > 0 else float("nan")

    @staticmethod
    def _percentile_or_nan(values: list[float], percentile: float) -> float:
        finite_values = np.asarray(values, dtype=np.float64)
        finite_values = finite_values[np.isfinite(finite_values)]
        return float(np.percentile(finite_values, percentile)) if finite_values.size > 0 else float("nan")

    def _full_scan_summary(self) -> str:
        completed_count = len(self._full_scan_packet_counts)
        if completed_count == 0:
            active_bins = (
                0 if self._active_coverage_mask is None else int(np.count_nonzero(self._active_coverage_mask))
            )
            return (
                "return_coverage_assemblies=0 active_return_bins=%d/%d target=%d"
                % (active_bins, self._config.azimuth_bins, self._coverage_target_bins)
            )

        mean_span_sec = self._mean_or_nan(self._full_scan_spans_sec)
        p50_span_sec = self._percentile_or_nan(self._full_scan_spans_sec, 50.0)
        p95_span_sec = self._percentile_or_nan(self._full_scan_spans_sec, 95.0)
        mean_packets = self._mean_or_nan(self._full_scan_packet_counts)
        mean_period_sec = self._mean_or_nan(self._full_scan_periods_sec)
        effective_hz = self._hz_from_interval(mean_period_sec)
        return (
            "return_coverage_assemblies=%d total=%d packets_per_assembly=%.2f "
            "assembly_span_mean/p50/p95=%.4f/%.4f/%.4fs completion_period=%.4fs (%.2f Hz)"
            % (
                completed_count,
                self._full_scan_count,
                mean_packets,
                mean_span_sec,
                p50_span_sec,
                p95_span_sec,
                mean_period_sec,
                effective_hz,
            )
        )

    @staticmethod
    def _hz_from_interval(interval_sec: float) -> float:
        if not math.isfinite(interval_sec) or interval_sec <= 0.0:
            return float("nan")
        return 1.0 / interval_sec

    @staticmethod
    def _infer_timestamp_unit(span_value: float, header_interval_sec: float) -> str:
        if not math.isfinite(span_value) or span_value <= 0.0:
            return "unknown"
        if not math.isfinite(header_interval_sec) or header_interval_sec <= 0.0:
            return "unknown"

        best_name = "unknown"
        best_score = float("inf")
        for unit_name, scale_to_seconds in TIMESTAMP_UNIT_SCALES_TO_SECONDS.items():
            converted_span_sec = span_value * scale_to_seconds
            score = (
                abs(math.log10(converted_span_sec / header_interval_sec))
                if converted_span_sec > 0.0
                else float("inf")
            )
            if score < best_score:
                best_score = score
                best_name = unit_name

        converted_span_sec = span_value * TIMESTAMP_UNIT_SCALES_TO_SECONDS[best_name]
        return "%s (span_sec=%.6f vs header_interval_sec=%.6f)" % (
            best_name,
            converted_span_sec,
            header_interval_sec,
        )

    def _reset_window(self) -> None:
        self._message_count = 0
        self._receive_intervals_sec = []
        self._header_intervals_sec = []
        self._point_counts = []
        self._height_filtered_point_counts = []
        self._timestamp_min_values = []
        self._timestamp_max_values = []
        self._timestamp_spans = []
        self._timestamp_sample_count = 0
        self._packet_azimuth_min_deg = []
        self._packet_azimuth_max_deg = []
        self._packet_front_coverage_fraction = []
        self._packet_front_bin_counts = []
        self._packet_time_slice_first_harmonic = []
        self._packet_time_slice_second_harmonic = []
        self._packet_time_slice_folded_spans_deg = []
        self._packet_time_slice_fan_center_sweeps_deg = []
        self._packet_time_slice_counts = []
        self._full_scan_spans_sec = []
        self._full_scan_packet_counts = []
        self._full_scan_periods_sec = []


def main() -> None:
    config = parse_args()
    rclpy.init(args=None)
    node = CloudExperimentNode(config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
