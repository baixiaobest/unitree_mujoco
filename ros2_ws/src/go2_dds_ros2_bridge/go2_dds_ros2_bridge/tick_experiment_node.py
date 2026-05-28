#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
import math
import os
import sys
import threading
from dataclasses import dataclass
from time import monotonic_ns

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String

from go2_dds_ros2_bridge.dds_runtime import add_dds_runtime_arguments, resolve_runtime_arguments

DEFAULT_DDS_TOPIC = "rt/lowstate"
DEFAULT_MEAN_TOPIC = "/lowstate_tick_experiment/mean_sec_per_tick"
DEFAULT_VARIANCE_TOPIC = "/lowstate_tick_experiment/variance_sec2_per_tick2"
DEFAULT_INTERVAL_TOPIC = "/lowstate_tick_experiment/mean_receive_interval_sec"
DEFAULT_TICK_INCREMENT_TOPIC = "/lowstate_tick_experiment/mean_tick_increment"
DEFAULT_TICK_INCREMENT_SUMMARY_TOPIC = "/lowstate_tick_experiment/tick_increment_summary"
DEFAULT_PUBLISH_PERIOD_SEC = 1.0
DEFAULT_SUMMARY_TOP_K = 5
LOWSTATE_TICK_WRAPAROUND = 1 << 32


@dataclass(frozen=True)
class BridgeConfig:
    dds_topic: str
    mean_topic: str
    variance_topic: str
    interval_topic: str
    tick_increment_topic: str
    tick_increment_summary_topic: str
    dds_domain_id: int
    dds_interface: str
    publish_period_sec: float
    summary_top_k: int


class RunningStats:
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


def parse_args() -> BridgeConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether Unitree LowState.tick behaves like microseconds-from-boot by comparing "
            "DDS receive intervals against tick increments."
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
        "--mean-topic",
        type=str,
        default=DEFAULT_MEAN_TOPIC,
        help="ROS2 topic that publishes the 1-second window mean of receive_interval_sec / tick_increment.",
    )
    parser.add_argument(
        "--variance-topic",
        type=str,
        default=DEFAULT_VARIANCE_TOPIC,
        help="ROS2 topic that publishes the 1-second window variance of receive_interval_sec / tick_increment.",
    )
    parser.add_argument(
        "--interval-topic",
        type=str,
        default=DEFAULT_INTERVAL_TOPIC,
        help="ROS2 topic that publishes the 1-second window mean receive interval in seconds.",
    )
    parser.add_argument(
        "--tick-increment-topic",
        type=str,
        default=DEFAULT_TICK_INCREMENT_TOPIC,
        help="ROS2 topic that publishes the 1-second window mean tick increment.",
    )
    parser.add_argument(
        "--tick-increment-summary-topic",
        type=str,
        default=DEFAULT_TICK_INCREMENT_SUMMARY_TOPIC,
        help="ROS2 topic that publishes a compact frequency summary of tick increments within each window.",
    )
    parser.add_argument(
        "--publish-period-sec",
        type=float,
        default=DEFAULT_PUBLISH_PERIOD_SEC,
        help="Window duration and publish period for experiment statistics.",
    )
    parser.add_argument(
        "--summary-top-k",
        type=int,
        default=DEFAULT_SUMMARY_TOP_K,
        help="Maximum number of tick increment frequencies included in the published summary string.",
    )

    non_ros_args = rclpy.utilities.remove_ros_args(args=sys.argv)[1:]
    args = parser.parse_args(non_ros_args)
    runtime_profile = resolve_runtime_arguments(args)

    return BridgeConfig(
        dds_topic=args.dds_topic,
        mean_topic=args.mean_topic,
        variance_topic=args.variance_topic,
        interval_topic=args.interval_topic,
        tick_increment_topic=args.tick_increment_topic,
        tick_increment_summary_topic=args.tick_increment_summary_topic,
        dds_domain_id=runtime_profile.domain_id,
        dds_interface=runtime_profile.interface,
        publish_period_sec=max(args.publish_period_sec, 0.1),
        summary_top_k=max(args.summary_top_k, 1),
    )


class LowStateTickExperiment(Node):
    def __init__(self, config: BridgeConfig, channel_subscriber_cls, dds_lowstate_type) -> None:
        super().__init__("go2_tick_experiment")
        self._config = config
        self._mean_publisher = self.create_publisher(Float64, self._config.mean_topic, 10)
        self._variance_publisher = self.create_publisher(Float64, self._config.variance_topic, 10)
        self._interval_publisher = self.create_publisher(Float64, self._config.interval_topic, 10)
        self._tick_increment_publisher = self.create_publisher(Float64, self._config.tick_increment_topic, 10)
        self._tick_increment_summary_publisher = self.create_publisher(String, self._config.tick_increment_summary_topic, 10)
        self._sample_lock = threading.Lock()
        self._ratio_stats = RunningStats()
        self._interval_stats = RunningStats()
        self._tick_increment_stats = RunningStats()
        self._tick_increment_counts: Counter[int] = Counter()
        self._window_skipped_samples = 0
        self._previous_receive_time_ns: int | None = None
        self._previous_tick: int | None = None
        self._missing_tick_field_reported = False

        self._dds_subscriber = channel_subscriber_cls(self._config.dds_topic, dds_lowstate_type)
        self._dds_subscriber.Init(self._dds_lowstate_handler, 10)
        self._publish_timer = self.create_timer(self._config.publish_period_sec, self._publish_statistics)

        ros_domain_id = os.environ.get("ROS_DOMAIN_ID", "<unset>")
        self.get_logger().info(
            "Running LowState tick experiment on '%s' (domain=%d, interface=%s, ROS_DOMAIN_ID=%s). "
            "Publishing sec/tick mean on '%s', variance on '%s', mean receive interval on '%s', mean tick increment on '%s', "
            "and tick increment summary on '%s' every %.2fs. If tick is microseconds-from-boot, "
            "the mean should trend toward 1e-6 sec/tick."
            % (
                self._config.dds_topic,
                self._config.dds_domain_id,
                self._config.dds_interface,
                ros_domain_id,
                self._config.mean_topic,
                self._config.variance_topic,
                self._config.interval_topic,
                self._config.tick_increment_topic,
                self._config.tick_increment_summary_topic,
                self._config.publish_period_sec,
            )
        )

    def _dds_lowstate_handler(self, msg) -> None:
        raw_tick = getattr(msg, "tick", None)
        if raw_tick is None:
            if not self._missing_tick_field_reported:
                self._missing_tick_field_reported = True
                self.get_logger().error("Received LowState message without a tick field; experiment cannot proceed.")
            return

        receive_time_ns = monotonic_ns()
        tick = int(raw_tick) % LOWSTATE_TICK_WRAPAROUND

        with self._sample_lock:
            if self._previous_receive_time_ns is None or self._previous_tick is None:
                self._previous_receive_time_ns = receive_time_ns
                self._previous_tick = tick
                return

            interval_ns = receive_time_ns - self._previous_receive_time_ns
            tick_increment = self._compute_tick_increment(self._previous_tick, tick)
            self._previous_receive_time_ns = receive_time_ns
            self._previous_tick = tick

            if interval_ns <= 0 or tick_increment <= 0:
                self._window_skipped_samples += 1
                return

            ratio_sec_per_tick = (interval_ns * 1e-9) / tick_increment
            if not math.isfinite(ratio_sec_per_tick):
                self._window_skipped_samples += 1
                return

            interval_sec = interval_ns * 1e-9
            self._ratio_stats.add_sample(ratio_sec_per_tick)
            self._interval_stats.add_sample(interval_sec)
            self._tick_increment_stats.add_sample(float(tick_increment))
            self._tick_increment_counts[tick_increment] += 1

    @staticmethod
    def _compute_tick_increment(previous_tick: int, current_tick: int) -> int:
        if current_tick >= previous_tick:
            return current_tick - previous_tick
        return (LOWSTATE_TICK_WRAPAROUND - previous_tick) + current_tick

    def _publish_statistics(self) -> None:
        with self._sample_lock:
            sample_count, mean_ratio, variance_ratio = self._ratio_stats.snapshot()
            _, mean_interval_sec, _ = self._interval_stats.snapshot()
            _, mean_tick_increment, _ = self._tick_increment_stats.snapshot()
            tick_increment_counts = dict(self._tick_increment_counts)
            skipped_samples = self._window_skipped_samples
            self._ratio_stats.reset()
            self._interval_stats.reset()
            self._tick_increment_stats.reset()
            self._tick_increment_counts.clear()
            self._window_skipped_samples = 0

        mean_msg = Float64()
        variance_msg = Float64()
        interval_msg = Float64()
        tick_increment_msg = Float64()
        summary_msg = String()
        mean_msg.data = mean_ratio
        variance_msg.data = variance_ratio
        interval_msg.data = mean_interval_sec
        tick_increment_msg.data = mean_tick_increment
        summary_msg.data = self._format_tick_increment_summary(tick_increment_counts, sample_count)
        self._mean_publisher.publish(mean_msg)
        self._variance_publisher.publish(variance_msg)
        self._interval_publisher.publish(interval_msg)
        self._tick_increment_publisher.publish(tick_increment_msg)
        self._tick_increment_summary_publisher.publish(summary_msg)

        if sample_count == 0:
            self.get_logger().warning(
                "Tick experiment window produced no valid samples over %.2fs (skipped=%d)."
                % (self._config.publish_period_sec, skipped_samples)
            )
            return

        self.get_logger().info(
            "tick_ratio window: samples=%d skipped=%d mean=%.9e sec/tick (%.3f us/tick) variance=%.9e "
            "mean_interval=%.9e sec (%.3f ms) mean_tick_increment=%.3f tick_summary=%s"
            % (
                sample_count,
                skipped_samples,
                mean_ratio,
                mean_ratio * 1e6,
                variance_ratio,
                mean_interval_sec,
                mean_interval_sec * 1e3,
                mean_tick_increment,
                summary_msg.data,
            )
        )

    def _format_tick_increment_summary(self, tick_increment_counts: dict[int, int], sample_count: int) -> str:
        if sample_count == 0 or not tick_increment_counts:
            return "no valid samples"

        ordered_counts = sorted(tick_increment_counts.items(), key=lambda item: (-item[1], item[0]))
        top_counts = ordered_counts[: self._config.summary_top_k]
        formatted_entries = [
            f"{tick_increment}:{count} ({100.0 * count / sample_count:.1f}%)"
            for tick_increment, count in top_counts
        ]
        remainder_count = sample_count - sum(count for _, count in top_counts)
        if remainder_count > 0:
            formatted_entries.append(f"other:{remainder_count} ({100.0 * remainder_count / sample_count:.1f}%)")
        return ", ".join(formatted_entries)


def main() -> None:
    config = parse_args()
    ChannelFactoryInitialize, channel_subscriber_cls, dds_lowstate_type = import_raw_dds_dependencies()
    ChannelFactoryInitialize(config.dds_domain_id, config.dds_interface)
    rclpy.init(args=None)

    node = LowStateTickExperiment(config, channel_subscriber_cls, dds_lowstate_type)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()