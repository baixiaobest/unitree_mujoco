#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from time import monotonic_ns

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField


DEFAULT_INPUT_TOPIC = "/utlidar/time_corrected/cloud"
DEFAULT_LOG_EVERY_MESSAGES = 20
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


@dataclass(frozen=True)
class ExperimentConfig:
    input_topic: str
    log_every_messages: int


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
    non_ros_args = rclpy.utilities.remove_ros_args(args=sys.argv)[1:]
    args = parser.parse_args(non_ros_args)
    return ExperimentConfig(
        input_topic=args.input_topic,
        log_every_messages=max(args.log_every_messages, 1),
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
        f"{field.name}:{POINT_FIELD_TYPE_NAMES.get(field.datatype, str(field.datatype))}[{int(field.count)}]@{int(field.offset)}"
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

        self.get_logger().info(
            "Inspecting PointCloud2 topic '%s'. A summary will be logged every %d messages."
            % (self._config.input_topic, self._config.log_every_messages)
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
                "Cloud schema: frame_id='%s', width=%d, height=%d, point_step=%d, row_step=%d, is_bigendian=%s, fields=[%s]"
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
            point_count, timestamp_field_name, timestamp_type_name, timestamp_stats = self._inspect_cloud(msg)
        except ValueError as error:
            self.get_logger().error(f"Failed to inspect PointCloud2 message: {error}")
            return

        self._point_counts.append(point_count)
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

    def _inspect_cloud(self, msg: PointCloud2) -> tuple[int, str | None, str | None, tuple[float, float, float] | None]:
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

        timestamp_field_name = next((name for name in TIMESTAMP_FIELD_NAMES if name in field_names), None)
        if timestamp_field_name is None:
            return point_count, None, None, None

        raw_timestamps = np.asarray(cloud[timestamp_field_name], dtype=np.float64).reshape(-1)
        timestamp_mask = mask & np.isfinite(raw_timestamps)
        timestamps = raw_timestamps[timestamp_mask]
        if timestamps.size == 0:
            return point_count, timestamp_field_name, self._field_type_name(msg, timestamp_field_name), None

        return (
            point_count,
            timestamp_field_name,
            self._field_type_name(msg, timestamp_field_name),
            (float(np.min(timestamps)), float(np.max(timestamps)), float(np.max(timestamps) - np.min(timestamps))),
        )

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

        summary = (
            "Cloud experiment summary: messages=%d frame_id='%s' avg_points=%.1f "
            "recv_interval=%.6fs (%.2f Hz) header_interval=%.6fs (%.2f Hz)"
            % (
                self._message_count,
                frame_id,
                mean_point_count,
                mean_receive_interval,
                mean_receive_hz,
                mean_header_interval,
                mean_header_hz,
            )
        )

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
            score = abs(math.log10(converted_span_sec / header_interval_sec)) if converted_span_sec > 0.0 else float("inf")
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
        self._timestamp_min_values = []
        self._timestamp_max_values = []
        self._timestamp_spans = []
        self._timestamp_sample_count = 0


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