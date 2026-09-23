#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from collections import deque
from dataclasses import dataclass

import rclpy
from builtin_interfaces.msg import Time
from rclpy.node import Node
from sensor_msgs.msg import Imu

from go2_dds_ros2_bridge.dds_runtime import add_dds_runtime_arguments, resolve_runtime_arguments

DEFAULT_DDS_IMU_TOPIC = "rt/lowstate"
DEFAULT_ROS_IMU_TOPIC = "/imu/data_raw"
DEFAULT_IMU_FRAME_ID = "imu_link"
DEFAULT_IMU_CONFIG_FILE = "imu_bridge.yaml"
LOWSTATE_TICK_NS = 1_000_000
LOWSTATE_TICK_WRAPAROUND = 1 << 32
LOWSTATE_TICK_HALF_WRAPAROUND = LOWSTATE_TICK_WRAPAROUND // 2
CLOCK_OFFSET_WINDOW_NS = 5_000_000_000
CLOCK_OFFSET_QUANTILE = 0.05
IMU_DEFAULT_VARIANCES = {
    "orientation": 0.05,
    "angular_velocity": 0.02,
    "linear_acceleration": 0.1,
}


@dataclass(frozen=True)
class BridgeConfig:
    dds_topic: str
    ros_topic: str
    frame_id: str
    dds_domain_id: int
    dds_interface: str
    orientation_variance: float
    angular_velocity_variance: float
    linear_acceleration_variance: float
    variance_source: str


def _import_yaml_module():
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise SystemExit(
            "The ROS2 bridge runtime cannot import 'yaml'.\n"
            "Install python3-yaml into the same Python interpreter that runs ROS2."
        ) from error
    return yaml


def default_config_file() -> Path | None:
    try:
        from ament_index_python.packages import get_package_share_directory

        installed_config_file = Path(get_package_share_directory("go2_dds_ros2_bridge")) / "config" / DEFAULT_IMU_CONFIG_FILE
        if installed_config_file.is_file():
            return installed_config_file
    except Exception:
        pass

    source_config_file = Path(__file__).resolve().parents[2] / "config" / DEFAULT_IMU_CONFIG_FILE
    if source_config_file.is_file():
        return source_config_file
    return None


def load_imu_variances(config_file: Path | None) -> tuple[dict[str, float], str]:
    variances = dict(IMU_DEFAULT_VARIANCES)
    if config_file is None:
        raise SystemExit(
            "The IMU bridge requires a YAML config file. Pass --config-file or provide config/imu_bridge.yaml in the package."
        )

    yaml = _import_yaml_module()
    file_path = Path(config_file)
    loaded_data = yaml.safe_load(file_path.read_text())
    if loaded_data is None:
        return variances, str(file_path)

    section = loaded_data.get("variances", loaded_data) if isinstance(loaded_data, dict) else loaded_data
    if not isinstance(section, dict):
        raise SystemExit("IMU bridge YAML must contain a mapping of variance names to numeric values.")

    key_aliases = {
        "orientation": "orientation",
        "orientation_variance": "orientation",
        "angular_velocity": "angular_velocity",
        "angular_velocity_variance": "angular_velocity",
        "linear_acceleration": "linear_acceleration",
        "linear_acceleration_variance": "linear_acceleration",
    }
    for key, value in section.items():
        if not isinstance(key, str):
            raise SystemExit("IMU bridge YAML keys must be strings.")
        canonical_key = key_aliases.get(key)
        if canonical_key is None:
            valid_keys = ", ".join(sorted(key_aliases))
            raise SystemExit(f"Unsupported IMU bridge variance key '{key}'. Supported keys: {valid_keys}")
        if not isinstance(value, (int, float)):
            raise SystemExit(f"IMU bridge variance '{key}' must be numeric.")
        numeric_value = float(value)
        if not math.isfinite(numeric_value) or numeric_value < 0.0:
            raise SystemExit(f"IMU bridge variance '{key}' must be finite and non-negative.")
        variances[canonical_key] = numeric_value
    return variances, str(file_path)


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
        description="Bridge Unitree raw DDS LowState IMU data to a ROS2 sensor_msgs/Imu topic."
    )
    add_dds_runtime_arguments(parser)
    parser.add_argument(
        "--dds-topic",
        type=str,
        default=DEFAULT_DDS_IMU_TOPIC,
        help="Raw DDS lowstate topic to subscribe to.",
    )
    parser.add_argument(
        "--ros-topic",
        type=str,
        default=DEFAULT_ROS_IMU_TOPIC,
        help="ROS2 IMU topic to publish.",
    )
    parser.add_argument(
        "--frame-id",
        type=str,
        default=DEFAULT_IMU_FRAME_ID,
        help="frame_id used for the published IMU message.",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=default_config_file(),
        help="YAML file containing IMU bridge variances.",
    )
    non_ros_args = rclpy.utilities.remove_ros_args(args=sys.argv)[1:]
    args = parser.parse_args(non_ros_args)
    runtime_profile = resolve_runtime_arguments(args)
    configured_variances, variance_source = load_imu_variances(args.config_file)

    orientation_variance = configured_variances["orientation"]
    angular_velocity_variance = configured_variances["angular_velocity"]
    linear_acceleration_variance = configured_variances["linear_acceleration"]

    return BridgeConfig(
        dds_topic=args.dds_topic,
        ros_topic=args.ros_topic,
        frame_id=args.frame_id,
        dds_domain_id=runtime_profile.domain_id,
        dds_interface=runtime_profile.interface,
        orientation_variance=max(orientation_variance, 0.0),
        angular_velocity_variance=max(angular_velocity_variance, 0.0),
        linear_acceleration_variance=max(linear_acceleration_variance, 0.0),
        variance_source=variance_source,
    )


def diagonal_covariance(variance: float) -> list[float]:
    covariance = [0.0] * 9
    covariance[0] = variance
    covariance[4] = variance
    covariance[8] = variance
    return covariance


def stamp_from_ns(stamp_ns: int) -> Time:
    stamp = Time()
    stamp.sec = int(stamp_ns // 1_000_000_000)
    stamp.nanosec = int(stamp_ns % 1_000_000_000)
    return stamp


def low_quantile(values: deque[tuple[int, int]]) -> int:
    ordered = sorted(offset_ns for _, offset_ns in values)
    index = min(int(math.floor((len(ordered) - 1) * CLOCK_OFFSET_QUANTILE)), len(ordered) - 1)
    return ordered[index]


class DdsImuBridge(Node):
    def __init__(self, config: BridgeConfig, channel_subscriber_cls, dds_lowstate_type) -> None:
        super().__init__("go2_imu_bridge")
        self._config = config
        self._publisher = self.create_publisher(Imu, self._config.ros_topic, 10)
        self._orientation_covariance = diagonal_covariance(self._config.orientation_variance)
        self._angular_velocity_covariance = diagonal_covariance(self._config.angular_velocity_variance)
        self._linear_acceleration_covariance = diagonal_covariance(self._config.linear_acceleration_variance)
        self._last_tick_raw: int | None = None
        self._tick_epoch = 0
        self._offset_samples: deque[tuple[int, int]] = deque()
        self._last_stamp_ns: int | None = None

        self._dds_subscriber = channel_subscriber_cls(self._config.dds_topic, dds_lowstate_type)
        self._dds_subscriber.Init(self._dds_lowstate_handler, 10)

        ros_domain_id = os.environ.get("ROS_DOMAIN_ID", "<unset>")
        self.get_logger().info(
            "Bridging raw DDS lowstate '%s' (domain=%d, interface=%s) to ROS2 topic '%s' as frame '%s' (ROS_DOMAIN_ID=%s). "
            "IMU timestamps use the LowState.tick millisecond clock and a rolling low-delay host-clock map. Variances source: %s."
            % (
                self._config.dds_topic,
                self._config.dds_domain_id,
                self._config.dds_interface,
                self._config.ros_topic,
                self._config.frame_id,
                ros_domain_id,
                self._config.variance_source,
            )
        )

    def _dds_lowstate_handler(self, msg) -> None:
        imu_state = msg.imu_state
        quaternion = [float(value) for value in imu_state.quaternion]
        gyroscope = [float(value) for value in imu_state.gyroscope]
        accelerometer = [float(value) for value in imu_state.accelerometer]

        if not self._is_finite_vector(quaternion) or not self._is_finite_vector(gyroscope) or not self._is_finite_vector(accelerometer):
            return

        normalized_quaternion = self._normalize_quaternion(quaternion)
        if normalized_quaternion is None:
            return

        receive_time_ns = self.get_clock().now().nanoseconds
        stamp_ns = self._stamp_from_tick(int(msg.tick), receive_time_ns)

        ros_msg = Imu()
        ros_msg.header.stamp = stamp_from_ns(stamp_ns)
        ros_msg.header.frame_id = self._config.frame_id
        ros_msg.orientation.w = normalized_quaternion[0]
        ros_msg.orientation.x = normalized_quaternion[1]
        ros_msg.orientation.y = normalized_quaternion[2]
        ros_msg.orientation.z = normalized_quaternion[3]
        ros_msg.orientation_covariance = list(self._orientation_covariance)

        ros_msg.angular_velocity.x = gyroscope[0]
        ros_msg.angular_velocity.y = gyroscope[1]
        ros_msg.angular_velocity.z = gyroscope[2]
        ros_msg.angular_velocity_covariance = list(self._angular_velocity_covariance)

        ros_msg.linear_acceleration.x = accelerometer[0]
        ros_msg.linear_acceleration.y = accelerometer[1]
        ros_msg.linear_acceleration.z = accelerometer[2]
        ros_msg.linear_acceleration_covariance = list(self._linear_acceleration_covariance)

        self._publisher.publish(ros_msg)

    @staticmethod
    def _is_finite_vector(values: list[float]) -> bool:
        return all(math.isfinite(value) for value in values)

    @staticmethod
    def _normalize_quaternion(values: list[float]) -> tuple[float, float, float, float] | None:
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= 1e-9:
            return None
        return (
            values[0] / norm,
            values[1] / norm,
            values[2] / norm,
            values[3] / norm,
        )

    def _stamp_from_tick(self, tick_raw: int, receive_time_ns: int) -> int:
        """Map the Unitree millisecond tick to host time without callback jitter.

        DDS callback timing is observably bursty, so it must not define sample
        times. The receive time is only used to maintain a slowly moving clock
        offset. LowState.tick wraps as an unsigned 32-bit millisecond counter.
        """
        tick_raw &= LOWSTATE_TICK_WRAPAROUND - 1
        last_tick_raw = self._last_tick_raw
        if last_tick_raw is not None and tick_raw < last_tick_raw:
            if last_tick_raw - tick_raw > LOWSTATE_TICK_HALF_WRAPAROUND:
                self._tick_epoch += LOWSTATE_TICK_WRAPAROUND
            else:
                # A small backward jump indicates a producer restart, not a
                # 49-day counter wrap. Re-anchor the new clock epoch safely.
                self._tick_epoch = 0
                self._offset_samples.clear()
                self.get_logger().warning("LowState.tick moved backwards; re-anchoring the IMU clock map.")

        self._last_tick_raw = tick_raw
        tick_ns = (self._tick_epoch + tick_raw) * LOWSTATE_TICK_NS
        self._offset_samples.append((receive_time_ns, receive_time_ns - tick_ns))
        oldest_allowed_ns = receive_time_ns - CLOCK_OFFSET_WINDOW_NS
        while self._offset_samples and self._offset_samples[0][0] < oldest_allowed_ns:
            self._offset_samples.popleft()
        stamp_ns = tick_ns + low_quantile(self._offset_samples)
        # Preserve monotonic ROS timestamps even if a source delivers an old
        # sample after a newer one.
        if self._last_stamp_ns is not None and stamp_ns < self._last_stamp_ns:
            stamp_ns = self._last_stamp_ns
        self._last_stamp_ns = stamp_ns
        return stamp_ns

def main() -> None:
    config = parse_args()
    ChannelFactoryInitialize, channel_subscriber_cls, dds_lowstate_type = import_raw_dds_dependencies()
    ChannelFactoryInitialize(config.dds_domain_id, config.dds_interface)
    rclpy.init(args=None)

    node = DdsImuBridge(config, channel_subscriber_cls, dds_lowstate_type)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
