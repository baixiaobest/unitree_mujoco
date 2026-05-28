#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from time import monotonic_ns

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray, String

from go2_dds_ros2_bridge.dds_runtime import add_dds_runtime_arguments, resolve_runtime_arguments

DEFAULT_DDS_TOPIC = "rt/lowstate"
DEFAULT_TOPIC_PREFIX = "/imu_experiment"
DEFAULT_PUBLISH_PERIOD_SEC = 1.0
DEFAULT_STATIONARY_HOLD_SEC = 2.0
DEFAULT_GYRO_STATIONARY_THRESHOLD_RAD_PER_SEC = 0.05
DEFAULT_ACCEL_NORM_TOLERANCE_MPS2 = 0.5
DEFAULT_ACCEL_DELTA_THRESHOLD_MPS2 = 0.1
DEFAULT_GRAVITY_MAGNITUDE_MPS2 = 9.81
DEFAULT_MIN_STATIONARY_SAMPLES = 200
DEFAULT_DURATION_SEC = 60.0
SUMMARY_WARNING_LIMIT = 3


@dataclass(frozen=True)
class ExperimentConfig:
    dds_topic: str
    topic_prefix: str
    dds_domain_id: int
    dds_interface: str
    publish_period_sec: float
    stationary_hold_sec: float
    gyro_stationary_threshold_rad_per_sec: float
    accel_norm_tolerance_mps2: float
    accel_delta_threshold_mps2: float
    gravity_magnitude_mps2: float
    min_stationary_samples: int
    duration_sec: float
    csv_path: Path | None


class RunningScalarStats:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.mean = 0.0
        self._sum_squared_deviation = 0.0

    def add_sample(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        updated_delta = value - self.mean
        self._sum_squared_deviation += delta * updated_delta

    def snapshot(self) -> tuple[int, float, float]:
        if self.count == 0:
            return 0, float("nan"), float("nan")
        variance = self._sum_squared_deviation / self.count
        return self.count, self.mean, variance


class RunningVectorStats:
    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.mean = np.zeros(self._dimension, dtype=np.float64)
        self._sum_squared_deviation = np.zeros((self._dimension, self._dimension), dtype=np.float64)

    def add_sample(self, values: np.ndarray) -> None:
        sample = np.asarray(values, dtype=np.float64).reshape(self._dimension)
        self.count += 1
        delta = sample - self.mean
        self.mean += delta / self.count
        updated_delta = sample - self.mean
        self._sum_squared_deviation += np.outer(delta, updated_delta)

    def snapshot(self) -> tuple[int, np.ndarray, np.ndarray]:
        if self.count == 0:
            nan_vector = np.full(self._dimension, np.nan, dtype=np.float64)
            nan_matrix = np.full((self._dimension, self._dimension), np.nan, dtype=np.float64)
            return 0, nan_vector, nan_matrix
        covariance = self._sum_squared_deviation / self.count
        return self.count, self.mean.copy(), covariance.copy()


def import_raw_dds_dependencies():
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as DdsLowState
    except ModuleNotFoundError as error:
        if error.name == "cyclonedds":
            raise SystemExit(
                "The ROS2 bridge runtime cannot import 'cyclonedds'.\n"
                "Install cyclonedds into the same Python interpreter that runs ROS2.\n"
                "For ROS Humble on this machine that is typically /usr/bin/python3 (Python 3.10).\n"
                "The Unitree conda environment has cyclonedds for Python 3.12, which is not import-compatible\n"
                "with the ROS Humble Python runtime."
            ) from error
        raise SystemExit(
            "The ROS2 bridge runtime cannot import unitree_sdk2py.\n"
            "Install unitree_sdk2py into the same Python interpreter that runs ROS2, or extend PYTHONPATH\n"
            "to point at a compatible installation built for the same Python version."
        ) from error

    return ChannelFactoryInitialize, ChannelSubscriber, DdsLowState


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Characterize Unitree IMU stationary bias and white-noise covariance from raw DDS LowState samples. "
            "The node also tracks a provisional bias covariance from stationary window means and can optionally "
            "log raw samples for offline Allan-variance analysis."
        )
    )
    add_dds_runtime_arguments(parser)
    parser.add_argument(
        "--dds-topic",
        type=str,
        default=DEFAULT_DDS_TOPIC,
        help="Raw DDS lowstate topic to subscribe to.",
    )
    parser.add_argument(
        "--topic-prefix",
        type=str,
        default=DEFAULT_TOPIC_PREFIX,
        help="Prefix for the published experiment topics.",
    )
    parser.add_argument(
        "--publish-period-sec",
        type=float,
        default=DEFAULT_PUBLISH_PERIOD_SEC,
        help="Period of the published experiment summaries.",
    )
    parser.add_argument(
        "--stationary-hold-sec",
        type=float,
        default=DEFAULT_STATIONARY_HOLD_SEC,
        help="Minimum uninterrupted stationary time before samples are accepted into the estimator.",
    )
    parser.add_argument(
        "--gyro-stationary-threshold",
        type=float,
        default=DEFAULT_GYRO_STATIONARY_THRESHOLD_RAD_PER_SEC,
        help="Maximum gyro norm (rad/s) allowed for a sample to be considered stationary.",
    )
    parser.add_argument(
        "--accel-norm-tolerance",
        type=float,
        default=DEFAULT_ACCEL_NORM_TOLERANCE_MPS2,
        help="Maximum absolute deviation from gravity magnitude (m/s^2) allowed for stationarity.",
    )
    parser.add_argument(
        "--accel-delta-threshold",
        type=float,
        default=DEFAULT_ACCEL_DELTA_THRESHOLD_MPS2,
        help="Maximum change in acceleration magnitude between consecutive samples for stationarity.",
    )
    parser.add_argument(
        "--gravity-magnitude",
        type=float,
        default=DEFAULT_GRAVITY_MAGNITUDE_MPS2,
        help="Expected gravity magnitude in m/s^2 for the current environment.",
    )
    parser.add_argument(
        "--min-stationary-samples",
        type=int,
        default=DEFAULT_MIN_STATIONARY_SAMPLES,
        help="Minimum accepted stationary samples required before a window contributes reported statistics.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=DEFAULT_DURATION_SEC,
        help="Total experiment duration before auto-stop and final summary. Set <=0 to run indefinitely.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Optional CSV path for raw IMU sample logging for offline Allan-variance analysis.",
    )

    non_ros_args = rclpy.utilities.remove_ros_args(args=sys.argv)[1:]
    args = parser.parse_args(non_ros_args)
    runtime_profile = resolve_runtime_arguments(args)

    return ExperimentConfig(
        dds_topic=args.dds_topic,
        topic_prefix=args.topic_prefix.rstrip("/"),
        dds_domain_id=runtime_profile.domain_id,
        dds_interface=runtime_profile.interface,
        publish_period_sec=max(args.publish_period_sec, 0.1),
        stationary_hold_sec=max(args.stationary_hold_sec, 0.0),
        gyro_stationary_threshold_rad_per_sec=max(args.gyro_stationary_threshold, 1e-6),
        accel_norm_tolerance_mps2=max(args.accel_norm_tolerance, 1e-6),
        accel_delta_threshold_mps2=max(args.accel_delta_threshold, 0.0),
        gravity_magnitude_mps2=max(args.gravity_magnitude, 1e-6),
        min_stationary_samples=max(args.min_stationary_samples, 1),
        duration_sec=float(args.duration_sec),
        csv_path=args.csv_path,
    )


class ImuExperimentNode(Node):
    def __init__(self, config: ExperimentConfig, channel_subscriber_cls, dds_lowstate_type) -> None:
        super().__init__("go2_imu_experiment")
        self._config = config
        self._sample_lock = threading.Lock()

        self._summary_publisher = self.create_publisher(String, self._topic("summary"), 10)
        self._sample_rate_publisher = self.create_publisher(Float64, self._topic("sample_rate_hz"), 10)
        self._stationary_ratio_publisher = self.create_publisher(Float64, self._topic("stationary_ratio"), 10)
        self._gravity_error_publisher = self.create_publisher(Float64, self._topic("accel_gravity_norm_error"), 10)
        self._gyro_bias_mean_publisher = self.create_publisher(Float64MultiArray, self._topic("gyro_bias_mean"), 10)
        self._accel_bias_mean_publisher = self.create_publisher(Float64MultiArray, self._topic("accel_bias_mean"), 10)
        self._gyro_noise_cov_publisher = self.create_publisher(Float64MultiArray, self._topic("gyro_noise_cov"), 10)
        self._accel_noise_cov_publisher = self.create_publisher(Float64MultiArray, self._topic("accel_noise_cov"), 10)
        self._gyro_bias_cov_publisher = self.create_publisher(Float64MultiArray, self._topic("gyro_bias_cov"), 10)
        self._accel_bias_cov_publisher = self.create_publisher(Float64MultiArray, self._topic("accel_bias_cov"), 10)

        self._interval_stats = RunningScalarStats()
        self._gyro_stationary_stats = RunningVectorStats(3)
        self._accel_stationary_stats = RunningVectorStats(3)
        self._gyro_bias_window_stats = RunningVectorStats(3)
        self._accel_bias_window_stats = RunningVectorStats(3)
        self._window_total_samples = 0
        self._window_candidate_stationary_samples = 0
        self._window_accepted_stationary_samples = 0
        self._window_rejected_motion_samples = 0
        self._window_invalid_samples = 0
        self._window_gyro_rejection_samples = 0
        self._window_accel_norm_rejection_samples = 0
        self._window_accel_delta_rejection_samples = 0
        self._previous_receive_time_ns: int | None = None
        self._previous_accel_norm_mps2: float | None = None
        self._last_motion_time_ns: int | None = None
        self._invalid_sample_warning_count = 0
        self._stop_requested = False

        self._csv_file = None
        self._csv_writer = None
        if self._config.csv_path is not None:
            self._config.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = self._config.csv_path.open("w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(
                [
                    "receive_time_ns",
                    "tick",
                    "gyro_x",
                    "gyro_y",
                    "gyro_z",
                    "accel_x",
                    "accel_y",
                    "accel_z",
                    "gyro_norm",
                    "accel_norm",
                    "stationary_candidate",
                    "stationary_accepted",
                ]
            )

        self._dds_subscriber = channel_subscriber_cls(self._config.dds_topic, dds_lowstate_type)
        self._dds_subscriber.Init(self._dds_lowstate_handler, 10)
        self._publish_timer = self.create_timer(self._config.publish_period_sec, self._publish_statistics)
        self._stop_timer = None
        if self._config.duration_sec > 0.0:
            self._stop_timer = self.create_timer(self._config.duration_sec, self._request_stop)

        ros_domain_id = os.environ.get("ROS_DOMAIN_ID", "<unset>")
        self.get_logger().info(
            "Running IMU experiment on '%s' (domain=%d, interface=%s, ROS_DOMAIN_ID=%s). "
            "Publishing summary under '%s/*' every %.2fs. Stationary acceptance requires %.2fs of uninterrupted "
            "samples with gyro_norm<=%.4f rad/s, |accel_norm-gravity|<=%.4f m/s^2, and accel_norm delta<=%.4f m/s^2. "
            "Bias covariance is a provisional covariance of stationary window means; use longer stationary logging "
            "for offline Allan-variance analysis before tuning FAST-LIO bias random-walk terms. Auto-stop after %.2fs."
            % (
                self._config.dds_topic,
                self._config.dds_domain_id,
                self._config.dds_interface,
                ros_domain_id,
                self._config.topic_prefix,
                self._config.publish_period_sec,
                self._config.stationary_hold_sec,
                self._config.gyro_stationary_threshold_rad_per_sec,
                self._config.accel_norm_tolerance_mps2,
                self._config.accel_delta_threshold_mps2,
                self._config.duration_sec,
            )
        )

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def _topic(self, suffix: str) -> str:
        return f"{self._config.topic_prefix}/{suffix}"

    def _dds_lowstate_handler(self, msg) -> None:
        imu_state = msg.imu_state
        gyro = np.asarray([float(value) for value in imu_state.gyroscope], dtype=np.float64)
        accel = np.asarray([float(value) for value in imu_state.accelerometer], dtype=np.float64)
        if gyro.shape[0] != 3 or accel.shape[0] != 3:
            return
        if not np.all(np.isfinite(gyro)) or not np.all(np.isfinite(accel)):
            with self._sample_lock:
                self._window_invalid_samples += 1
            if self._invalid_sample_warning_count < SUMMARY_WARNING_LIMIT:
                self._invalid_sample_warning_count += 1
                self.get_logger().warning("Skipping invalid IMU sample with non-finite gyro or accelerometer values.")
            return

        receive_time_ns = monotonic_ns()
        tick = int(getattr(msg, "tick", 0))
        gyro_norm = float(np.linalg.norm(gyro))
        accel_norm = float(np.linalg.norm(accel))
        previous_accel_norm_mps2 = self._previous_accel_norm_mps2
        accel_delta = 0.0 if previous_accel_norm_mps2 is None else abs(accel_norm - previous_accel_norm_mps2)
        self._previous_accel_norm_mps2 = accel_norm

        stationary_candidate = (
            gyro_norm <= self._config.gyro_stationary_threshold_rad_per_sec
            and abs(accel_norm - self._config.gravity_magnitude_mps2) <= self._config.accel_norm_tolerance_mps2
            and accel_delta <= self._config.accel_delta_threshold_mps2
        )
        gyro_stationary = gyro_norm <= self._config.gyro_stationary_threshold_rad_per_sec
        accel_norm_stationary = abs(accel_norm - self._config.gravity_magnitude_mps2) <= self._config.accel_norm_tolerance_mps2
        accel_delta_stationary = accel_delta <= self._config.accel_delta_threshold_mps2
        hard_stationary = gyro_stationary and accel_norm_stationary

        stationary_accepted = False
        with self._sample_lock:
            self._window_total_samples += 1
            if self._previous_receive_time_ns is not None:
                interval_sec = (receive_time_ns - self._previous_receive_time_ns) * 1e-9
                if interval_sec > 0.0 and math.isfinite(interval_sec):
                    self._interval_stats.add_sample(interval_sec)
            self._previous_receive_time_ns = receive_time_ns

            if self._last_motion_time_ns is None:
                self._last_motion_time_ns = receive_time_ns

            if not gyro_stationary:
                self._window_gyro_rejection_samples += 1
            if not accel_norm_stationary:
                self._window_accel_norm_rejection_samples += 1
            if not accel_delta_stationary:
                self._window_accel_delta_rejection_samples += 1

            if stationary_candidate:
                self._window_candidate_stationary_samples += 1

            if hard_stationary:
                stationary_duration_ns = receive_time_ns - self._last_motion_time_ns
                if stationary_duration_ns >= int(round(self._config.stationary_hold_sec * 1e9)) and accel_delta_stationary:
                    stationary_accepted = True
                    self._window_accepted_stationary_samples += 1
                    self._gyro_stationary_stats.add_sample(gyro)
                    self._accel_stationary_stats.add_sample(accel)
                else:
                    self._window_rejected_motion_samples += 1
            else:
                self._window_rejected_motion_samples += 1
                self._last_motion_time_ns = receive_time_ns

        if self._csv_writer is not None:
            self._csv_writer.writerow(
                [
                    receive_time_ns,
                    tick,
                    gyro[0],
                    gyro[1],
                    gyro[2],
                    accel[0],
                    accel[1],
                    accel[2],
                    gyro_norm,
                    accel_norm,
                    int(stationary_candidate),
                    int(stationary_accepted),
                ]
            )

    def _publish_statistics(self) -> None:
        summary_text, accepted_samples, minimum_samples = self._emit_summary(log_summary=True)
        if self._stop_requested:
            self.get_logger().info(
                "IMU experiment auto-stop complete after %.2fs. Final summary: %s"
                % (self._config.duration_sec, summary_text)
            )
            if accepted_samples < minimum_samples:
                self.get_logger().warning(
                    "Final summary still had only %d accepted stationary samples (minimum=%d)."
                    % (accepted_samples, minimum_samples)
                )
            raise SystemExit(0)

    def _emit_summary(self, *, log_summary: bool) -> tuple[str, int, int]:
        with self._sample_lock:
            interval_count, mean_interval_sec, _ = self._interval_stats.snapshot()
            gyro_count, gyro_mean, gyro_cov = self._gyro_stationary_stats.snapshot()
            accel_count, accel_mean, accel_cov = self._accel_stationary_stats.snapshot()
            total_samples = self._window_total_samples
            candidate_samples = self._window_candidate_stationary_samples
            accepted_samples = self._window_accepted_stationary_samples
            rejected_samples = self._window_rejected_motion_samples
            invalid_samples = self._window_invalid_samples
            gyro_rejection_samples = self._window_gyro_rejection_samples
            accel_norm_rejection_samples = self._window_accel_norm_rejection_samples
            accel_delta_rejection_samples = self._window_accel_delta_rejection_samples
            last_motion_time_ns = self._last_motion_time_ns

            if gyro_count >= self._config.min_stationary_samples and accel_count >= self._config.min_stationary_samples:
                self._gyro_bias_window_stats.add_sample(gyro_mean)
                self._accel_bias_window_stats.add_sample(accel_mean)

            bias_window_count, _, gyro_bias_cov = self._gyro_bias_window_stats.snapshot()
            _, _, accel_bias_cov = self._accel_bias_window_stats.snapshot()

            self._interval_stats.reset()
            self._gyro_stationary_stats.reset()
            self._accel_stationary_stats.reset()
            self._window_total_samples = 0
            self._window_candidate_stationary_samples = 0
            self._window_accepted_stationary_samples = 0
            self._window_rejected_motion_samples = 0
            self._window_invalid_samples = 0
            self._window_gyro_rejection_samples = 0
            self._window_accel_norm_rejection_samples = 0
            self._window_accel_delta_rejection_samples = 0

        if self._csv_file is not None:
            self._csv_file.flush()

        sample_rate_hz = float("nan") if interval_count == 0 or mean_interval_sec <= 0.0 else 1.0 / mean_interval_sec
        stationary_ratio = 0.0 if total_samples == 0 else accepted_samples / total_samples
        gravity_norm_error = float("nan") if accel_count == 0 else abs(float(np.linalg.norm(accel_mean)) - self._config.gravity_magnitude_mps2)
        stationary_active = False
        if last_motion_time_ns is not None:
            stationary_active = (monotonic_ns() - last_motion_time_ns) >= int(round(self._config.stationary_hold_sec * 1e9))

        self._publish_scalar(self._sample_rate_publisher, sample_rate_hz)
        self._publish_scalar(self._stationary_ratio_publisher, stationary_ratio)
        self._publish_scalar(self._gravity_error_publisher, gravity_norm_error)
        self._publish_array(self._gyro_bias_mean_publisher, gyro_mean)
        self._publish_array(self._accel_bias_mean_publisher, accel_mean)
        self._publish_array(self._gyro_noise_cov_publisher, gyro_cov.reshape(-1))
        self._publish_array(self._accel_noise_cov_publisher, accel_cov.reshape(-1))
        self._publish_array(self._gyro_bias_cov_publisher, gyro_bias_cov.reshape(-1))
        self._publish_array(self._accel_bias_cov_publisher, accel_bias_cov.reshape(-1))

        summary = String()
        summary.data = (
            "stationary_active=%s total=%d candidates=%d accepted=%d rejected=%d invalid=%d sample_rate_hz=%s "
            "gyro_reject=%d accel_norm_reject=%d accel_delta_reject=%d gyro_mean=%s gyro_var_diag=%s "
            "accel_mean=%s accel_var_diag=%s gravity_norm_error=%s bias_windows=%d"
            % (
                stationary_active,
                total_samples,
                candidate_samples,
                accepted_samples,
                rejected_samples,
                invalid_samples,
                self._format_scalar(sample_rate_hz),
                gyro_rejection_samples,
                accel_norm_rejection_samples,
                accel_delta_rejection_samples,
                self._format_vector(gyro_mean),
                self._format_vector(np.diag(gyro_cov) if gyro_cov.ndim == 2 else gyro_cov),
                self._format_vector(accel_mean),
                self._format_vector(np.diag(accel_cov) if accel_cov.ndim == 2 else accel_cov),
                self._format_scalar(gravity_norm_error),
                bias_window_count,
            )
        )
        self._summary_publisher.publish(summary)

        if accepted_samples < self._config.min_stationary_samples:
            if candidate_samples > 0 and not stationary_active:
                if log_summary:
                    self.get_logger().info(
                        "IMU experiment window is still waiting for the stationary hold time: accepted=%d minimum=%d total=%d candidates=%d gyro_reject=%d accel_norm_reject=%d accel_delta_reject=%d."
                        % (
                            accepted_samples,
                            self._config.min_stationary_samples,
                            total_samples,
                            candidate_samples,
                            gyro_rejection_samples,
                            accel_norm_rejection_samples,
                            accel_delta_rejection_samples,
                        )
                    )
            else:
                if log_summary:
                    self.get_logger().warning(
                        "IMU experiment window produced only %d accepted stationary samples (minimum=%d, total=%d, candidates=%d, rejected=%d, invalid=%d, gyro_reject=%d, accel_norm_reject=%d, accel_delta_reject=%d)."
                        % (
                            accepted_samples,
                            self._config.min_stationary_samples,
                            total_samples,
                            candidate_samples,
                            rejected_samples,
                            invalid_samples,
                            gyro_rejection_samples,
                            accel_norm_rejection_samples,
                            accel_delta_rejection_samples,
                        )
                    )
            return summary.data, accepted_samples, self._config.min_stationary_samples

        if log_summary:
            self.get_logger().info(summary.data)
        return summary.data, accepted_samples, self._config.min_stationary_samples

    def _request_stop(self) -> None:
        if self._stop_requested:
            return
        self._stop_requested = True
        if self._stop_timer is not None:
            self._stop_timer.cancel()
        self.get_logger().info("IMU experiment reached requested duration; publishing final summary and stopping.")

    @staticmethod
    def _publish_scalar(publisher, value: float) -> None:
        message = Float64()
        message.data = float(value)
        publisher.publish(message)

    @staticmethod
    def _publish_array(publisher, values) -> None:
        message = Float64MultiArray()
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        message.data = [float(value) for value in array]
        publisher.publish(message)

    @staticmethod
    def _format_scalar(value: float) -> str:
        if not math.isfinite(value):
            return "nan"
        return f"{value:.6g}"

    @classmethod
    def _format_vector(cls, values) -> str:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        return "[" + ", ".join(cls._format_scalar(float(value)) for value in array) + "]"


def main() -> None:
    config = parse_args()
    ChannelFactoryInitialize, channel_subscriber_cls, dds_lowstate_type = import_raw_dds_dependencies()
    ChannelFactoryInitialize(config.dds_domain_id, config.dds_interface)
    rclpy.init(args=None)

    node = ImuExperimentNode(config, channel_subscriber_cls, dds_lowstate_type)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()