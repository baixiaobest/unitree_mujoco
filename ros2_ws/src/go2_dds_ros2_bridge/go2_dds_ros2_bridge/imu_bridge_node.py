#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import threading
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu


SIMULATION_DOMAIN_ID = 1
HARDWARE_DOMAIN_ID = 0
SIMULATION_INTERFACE = "wlo1"
HARDWARE_INTERFACE = "enp108s0"

DEFAULT_DDS_IMU_TOPIC = "rt/lowstate"
DEFAULT_ROS_IMU_TOPIC = "/imu/data_raw"
DEFAULT_IMU_FRAME_ID = "imu_link"
DEFAULT_IMU_CONFIG_FILE = "imu_bridge.yaml"
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
    publish_hz: float
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

    source_config_file = Path(__file__).resolve().parents[1] / "config" / DEFAULT_IMU_CONFIG_FILE
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
    parser.add_argument(
        "--run-mode",
        choices=("simulation", "hardware"),
        default="hardware",
        help="Select the default DDS domain/interface pair.",
    )
    parser.add_argument(
        "--dds-domain-id",
        type=int,
        default=None,
        help="Override the raw DDS domain id used to subscribe to the Unitree lowstate topic.",
    )
    parser.add_argument(
        "--dds-interface",
        type=str,
        default=None,
        help="Override the network interface used for the raw DDS subscriber.",
    )
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
    parser.add_argument(
        "--publish-hz",
        type=float,
        default=200.0,
        help="Maximum ROS2 publish rate for forwarding IMU messages.",
    )
    non_ros_args = rclpy.utilities.remove_ros_args(args=sys.argv)[1:]
    args = parser.parse_args(non_ros_args)
    default_domain = SIMULATION_DOMAIN_ID if args.run_mode == "simulation" else HARDWARE_DOMAIN_ID
    default_interface = SIMULATION_INTERFACE if args.run_mode == "simulation" else HARDWARE_INTERFACE
    configured_variances, variance_source = load_imu_variances(args.config_file)

    orientation_variance = configured_variances["orientation"]
    angular_velocity_variance = configured_variances["angular_velocity"]
    linear_acceleration_variance = configured_variances["linear_acceleration"]

    return BridgeConfig(
        dds_topic=args.dds_topic,
        ros_topic=args.ros_topic,
        frame_id=args.frame_id,
        dds_domain_id=default_domain if args.dds_domain_id is None else args.dds_domain_id,
        dds_interface=default_interface if args.dds_interface is None else args.dds_interface,
        publish_hz=max(args.publish_hz, 1.0),
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


class DdsImuBridge(Node):
    def __init__(self, config: BridgeConfig, channel_subscriber_cls, dds_lowstate_type) -> None:
        super().__init__("go2_imu_bridge")
        self._config = config
        self._publisher = self.create_publisher(Imu, self._config.ros_topic, 10)
        self._orientation_covariance = diagonal_covariance(self._config.orientation_variance)
        self._angular_velocity_covariance = diagonal_covariance(self._config.angular_velocity_variance)
        self._linear_acceleration_covariance = diagonal_covariance(self._config.linear_acceleration_variance)
        self._message_lock = threading.Lock()
        self._pending_message: Imu | None = None
        self._message_version = 0
        self._last_published_version = 0

        self._dds_subscriber = channel_subscriber_cls(self._config.dds_topic, dds_lowstate_type)
        self._dds_subscriber.Init(self._dds_lowstate_handler, 10)
        self._publish_timer = self.create_timer(1.0 / self._config.publish_hz, self._publish_pending_message)

        ros_domain_id = os.environ.get("ROS_DOMAIN_ID", "<unset>")
        self.get_logger().info(
            "Bridging raw DDS lowstate '%s' (domain=%d, interface=%s) to ROS2 topic '%s' as frame '%s' (ROS_DOMAIN_ID=%s, publish_hz=%.1f). "
            "This first implementation stamps IMU messages at ROS receive time. Variances source: %s."
            % (
                self._config.dds_topic,
                self._config.dds_domain_id,
                self._config.dds_interface,
                self._config.ros_topic,
                self._config.frame_id,
                ros_domain_id,
                self._config.publish_hz,
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

        ros_msg = Imu()
        ros_msg.header.stamp = self.get_clock().now().to_msg()
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

        with self._message_lock:
            self._pending_message = ros_msg
            self._message_version += 1

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

    def _publish_pending_message(self) -> None:
        with self._message_lock:
            if self._pending_message is None or self._message_version == self._last_published_version:
                return
            ros_msg = self._pending_message
            message_version = self._message_version

        self._publisher.publish(ros_msg)
        self._last_published_version = message_version


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