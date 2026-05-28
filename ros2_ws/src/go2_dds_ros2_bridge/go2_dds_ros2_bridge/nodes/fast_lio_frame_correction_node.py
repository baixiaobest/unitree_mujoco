#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Imu
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from go2_dds_ros2_bridge.dds_runtime import add_dds_runtime_arguments, resolve_runtime_arguments
from go2_dds_ros2_bridge.tf_utils import (
    quaternion_from_rotation_matrix,
    roll_pitch_correction_from_gravity,
    rotation_matrix_from_quaternion_xyzw,
)

DEFAULT_DDS_POSTURE_TOPIC = "rt/robot_posture"
DEFAULT_IMU_TOPIC = "/imu/data_fastlio"
DEFAULT_PARENT_FRAME = "camera_init_correct"
DEFAULT_CHILD_FRAME = "camera_init"
DEFAULT_BODY_FRAME = "body"
DEFAULT_TRIGGER_STATE = 4
DEFAULT_REPUBLISH_HZ = 10.0
DEFAULT_PROCESS_HZ = 20.0
DEFAULT_GRAVITY_SAMPLE_WINDOW_SEC = 0.5
DEFAULT_MAX_IMU_AGE_SEC = 0.5
DEFAULT_LOOKUP_TIMEOUT_SEC = 0.05
DEFAULT_MIN_GRAVITY_SAMPLES = 20
DEFAULT_STANDING_HEIGHT_M = 0.4


@dataclass(frozen=True)
class CorrectionConfig:
    dds_topic: str
    dds_domain_id: int
    dds_interface: str
    imu_topic: str
    parent_frame: str
    child_frame: str
    body_frame: str
    trigger_state: int
    republish_hz: float
    process_hz: float
    gravity_sample_window_sec: float
    max_imu_age_sec: float
    lookup_timeout_sec: float
    min_gravity_samples: int
    standing_height_m: float


@dataclass(frozen=True)
class ImuSample:
    receive_time_ns: int
    stamp: Time
    acceleration: np.ndarray


def import_raw_dds_dependencies():
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import UwbSwitch_ as DdsRobotPosture
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

    return ChannelFactoryInitialize, ChannelSubscriber, DdsRobotPosture


def parse_args() -> CorrectionConfig:
    parser = argparse.ArgumentParser(
        description="Broadcast a pitch/roll-only correction transform from camera_init_correct to camera_init."
    )
    add_dds_runtime_arguments(parser)
    parser.add_argument(
        "--dds-topic",
        type=str,
        default=DEFAULT_DDS_POSTURE_TOPIC,
        help="Raw DDS robot posture topic to subscribe to.",
    )
    parser.add_argument(
        "--imu-topic",
        type=str,
        default=DEFAULT_IMU_TOPIC,
        help="ROS2 IMU topic used to estimate gravity during stand_holding.",
    )
    parser.add_argument(
        "--parent-frame",
        type=str,
        default=DEFAULT_PARENT_FRAME,
        help="Parent frame for the correction transform.",
    )
    parser.add_argument(
        "--child-frame",
        type=str,
        default=DEFAULT_CHILD_FRAME,
        help="Child frame for the correction transform.",
    )
    parser.add_argument(
        "--body-frame",
        type=str,
        default=DEFAULT_BODY_FRAME,
        help="Frame in which the incoming IMU acceleration is expressed.",
    )
    parser.add_argument(
        "--trigger-state",
        type=int,
        default=DEFAULT_TRIGGER_STATE,
        help="Robot posture enum value that triggers recomputing the correction.",
    )
    parser.add_argument(
        "--republish-hz",
        type=float,
        default=DEFAULT_REPUBLISH_HZ,
        help="Rate used to republish the latest correction transform.",
    )
    parser.add_argument(
        "--process-hz",
        type=float,
        default=DEFAULT_PROCESS_HZ,
        help="Rate used to evaluate pending correction requests.",
    )
    parser.add_argument(
        "--gravity-sample-window-sec",
        type=float,
        default=DEFAULT_GRAVITY_SAMPLE_WINDOW_SEC,
        help="How long to average IMU acceleration after entering stand_holding before solving the correction.",
    )
    parser.add_argument(
        "--max-imu-age-sec",
        type=float,
        default=DEFAULT_MAX_IMU_AGE_SEC,
        help="Maximum allowed age of the newest IMU sample when solving the correction.",
    )
    parser.add_argument(
        "--lookup-timeout-sec",
        type=float,
        default=DEFAULT_LOOKUP_TIMEOUT_SEC,
        help="Timeout for tf lookups when resolving camera_init to body.",
    )
    parser.add_argument(
        "--min-gravity-samples",
        type=int,
        default=DEFAULT_MIN_GRAVITY_SAMPLES,
        help="Minimum number of IMU samples required to compute a correction.",
    )
    parser.add_argument(
        "--standing-height-m",
        type=float,
        default=DEFAULT_STANDING_HEIGHT_M,
        help="Expected body-frame height above the ground during stand_holding, expressed in camera_init_correct.",
    )
    non_ros_args = rclpy.utilities.remove_ros_args(args=sys.argv)[1:]
    args = parser.parse_args(non_ros_args)
    runtime_profile = resolve_runtime_arguments(args)

    if args.republish_hz <= 0.0:
        raise SystemExit("--republish-hz must be positive")
    if args.process_hz <= 0.0:
        raise SystemExit("--process-hz must be positive")
    if args.gravity_sample_window_sec <= 0.0:
        raise SystemExit("--gravity-sample-window-sec must be positive")
    if args.max_imu_age_sec <= 0.0:
        raise SystemExit("--max-imu-age-sec must be positive")
    if args.lookup_timeout_sec <= 0.0:
        raise SystemExit("--lookup-timeout-sec must be positive")
    if args.min_gravity_samples <= 0:
        raise SystemExit("--min-gravity-samples must be positive")
    if args.standing_height_m < 0.0:
        raise SystemExit("--standing-height-m must be non-negative")

    return CorrectionConfig(
        dds_topic=args.dds_topic,
        dds_domain_id=runtime_profile.domain_id,
        dds_interface=runtime_profile.interface,
        imu_topic=args.imu_topic,
        parent_frame=args.parent_frame,
        child_frame=args.child_frame,
        body_frame=args.body_frame,
        trigger_state=args.trigger_state,
        republish_hz=float(args.republish_hz),
        process_hz=float(args.process_hz),
        gravity_sample_window_sec=float(args.gravity_sample_window_sec),
        max_imu_age_sec=float(args.max_imu_age_sec),
        lookup_timeout_sec=float(args.lookup_timeout_sec),
        min_gravity_samples=int(args.min_gravity_samples),
        standing_height_m=float(args.standing_height_m),
    )


class FastLioFrameCorrectionNode(Node):
    def __init__(self, config: CorrectionConfig, channel_subscriber_cls, dds_robot_posture_type) -> None:
        super().__init__("fast_lio_frame_correction")
        self._config = config
        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = TransformBroadcaster(self)
        self._imu_subscription = self.create_subscription(Imu, self._config.imu_topic, self._imu_callback, 50)
        self._dds_subscriber = channel_subscriber_cls(self._config.dds_topic, dds_robot_posture_type)
        self._dds_subscriber.Init(self._dds_posture_handler, 10)
        self._state_lock = threading.Lock()
        self._imu_samples: deque[ImuSample] = deque(maxlen=max(self._config.min_gravity_samples * 10, 512))
        self._current_posture_state: int | None = None
        self._pending_trigger_time_ns: int | None = None
        self._latest_correction_rotation: np.ndarray | None = None
        self._latest_correction_translation: np.ndarray | None = None
        self._imu_frame_warning_emitted = False
        self._tf_warning_last_ns = 0
        self._imu_warning_last_ns = 0
        self._publish_timer = self.create_timer(1.0 / self._config.republish_hz, self._publish_cached_transform)
        self._process_timer = self.create_timer(1.0 / self._config.process_hz, self._process_pending_correction)

        ros_domain_id = os.environ.get("ROS_DOMAIN_ID", "<unset>")
        self.get_logger().info(
            "Correcting FAST-LIO frame '%s' by broadcasting '%s' -> '%s' when DDS posture '%s' enters state %d "
            "(domain=%d, interface=%s, ROS_DOMAIN_ID=%s, imu_topic=%s)."
            % (
                self._config.child_frame,
                self._config.parent_frame,
                self._config.child_frame,
                self._config.dds_topic,
                self._config.trigger_state,
                self._config.dds_domain_id,
                self._config.dds_interface,
                ros_domain_id,
                self._config.imu_topic,
            )
        )

    def _imu_callback(self, msg: Imu) -> None:
        if msg.header.frame_id and msg.header.frame_id != self._config.body_frame and not self._imu_frame_warning_emitted:
            self._imu_frame_warning_emitted = True
            self.get_logger().warning(
                "Incoming IMU frame '%s' does not match configured body frame '%s'."
                % (msg.header.frame_id, self._config.body_frame)
            )

        acceleration = np.array(
            [
                float(msg.linear_acceleration.x),
                float(msg.linear_acceleration.y),
                float(msg.linear_acceleration.z),
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(acceleration)):
            return

        receive_time_ns = self.get_clock().now().nanoseconds
        sample = ImuSample(
            receive_time_ns=receive_time_ns,
            stamp=Time.from_msg(msg.header.stamp),
            acceleration=acceleration,
        )
        retention_ns = int(max(self._config.gravity_sample_window_sec, self._config.max_imu_age_sec) * 4.0 * 1e9)
        with self._state_lock:
            self._imu_samples.append(sample)
            while self._imu_samples and receive_time_ns - self._imu_samples[0].receive_time_ns > retention_ns:
                self._imu_samples.popleft()

    def _dds_posture_handler(self, msg) -> None:
        posture_state = int(msg.enabled)
        now_ns = self.get_clock().now().nanoseconds
        with self._state_lock:
            previous_state = self._current_posture_state
            self._current_posture_state = posture_state
            if posture_state == self._config.trigger_state and previous_state != self._config.trigger_state:
                self._pending_trigger_time_ns = now_ns
            elif posture_state != self._config.trigger_state and previous_state == self._config.trigger_state:
                self._pending_trigger_time_ns = None

    def _process_pending_correction(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        sample_window_ns = int(self._config.gravity_sample_window_sec * 1e9)
        max_imu_age_ns = int(self._config.max_imu_age_sec * 1e9)

        with self._state_lock:
            trigger_time_ns = self._pending_trigger_time_ns
            posture_state = self._current_posture_state
            imu_samples = list(self._imu_samples)

        if trigger_time_ns is None:
            return
        if posture_state != self._config.trigger_state:
            return
        if now_ns - trigger_time_ns < sample_window_ns:
            return

        samples_since_trigger = [sample for sample in imu_samples if sample.receive_time_ns >= trigger_time_ns]
        if len(samples_since_trigger) < self._config.min_gravity_samples:
            self._maybe_log_imu_readiness_warning(
                now_ns,
                "Waiting for more IMU samples after entering stand_holding (%d/%d ready)."
                % (len(samples_since_trigger), self._config.min_gravity_samples),
            )
            return

        latest_sample = samples_since_trigger[-1]
        if now_ns - latest_sample.receive_time_ns > max_imu_age_ns:
            self._maybe_log_imu_readiness_warning(
                now_ns,
                "Latest IMU sample is too old to compute frame correction (age=%.3f s)."
                % ((now_ns - latest_sample.receive_time_ns) / 1e9),
            )
            return

        mean_acceleration = np.mean([sample.acceleration for sample in samples_since_trigger], axis=0)
        gravity_body = -mean_acceleration
        transform_camera_body = self._lookup_transform_components(
            target_frame=self._config.child_frame,
            source_frame=self._config.body_frame,
            stamp=latest_sample.stamp,
        )
        if transform_camera_body is None:
            return
        translation_camera_body, rotation_camera_body = transform_camera_body

        gravity_camera = rotation_camera_body @ gravity_body
        gravity_norm = float(np.linalg.norm(gravity_camera))
        if not math.isfinite(gravity_norm) or gravity_norm <= 1e-6:
            self._maybe_log_imu_readiness_warning(now_ns, "Computed gravity vector is invalid; skipping correction update.")
            return

        try:
            roll, pitch, correction_rotation = roll_pitch_correction_from_gravity(gravity_camera)
        except ValueError as error:
            self._maybe_log_imu_readiness_warning(now_ns, f"Failed to solve gravity-based correction: {error}")
            return

        body_position_in_corrected_frame = correction_rotation @ translation_camera_body
        correction_translation = np.array(
            [0.0, 0.0, self._config.standing_height_m - float(body_position_in_corrected_frame[2])],
            dtype=np.float64,
        )

        with self._state_lock:
            self._latest_correction_rotation = correction_rotation.copy()
            self._latest_correction_translation = correction_translation.copy()
            self._pending_trigger_time_ns = None

        self.get_logger().info(
            "Updated FAST-LIO level correction from %d IMU samples: roll=%.2f deg, pitch=%.2f deg, yaw=0.00 deg, z_offset=%.3f m, body_height=%.3f m."
            % (
                len(samples_since_trigger),
                math.degrees(roll),
                math.degrees(pitch),
                float(correction_translation[2]),
                float(body_position_in_corrected_frame[2]),
            )
        )
        self._publish_cached_transform()

    def _lookup_transform_components(
        self,
        *,
        target_frame: str,
        source_frame: str,
        stamp: Time,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        last_error: Exception | None = None
        for lookup_stamp in (stamp, Time()):
            try:
                transform = self._tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    lookup_stamp,
                    timeout=Duration(seconds=self._config.lookup_timeout_sec),
                )
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                return (
                    np.array(
                        [
                            float(translation.x),
                            float(translation.y),
                            float(translation.z),
                        ],
                        dtype=np.float64,
                    ),
                    rotation_matrix_from_quaternion_xyzw(
                        float(rotation.x),
                        float(rotation.y),
                        float(rotation.z),
                        float(rotation.w),
                    ),
                )
            except TransformException as error:
                last_error = error

        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._tf_warning_last_ns >= 2_000_000_000:
            self._tf_warning_last_ns = now_ns
            self.get_logger().warning(
                "Failed to resolve transform %s -> %s for frame correction: %s"
                % (target_frame, source_frame, last_error)
            )
        return None

    def _maybe_log_imu_readiness_warning(self, now_ns: int, message: str) -> None:
        if now_ns - self._imu_warning_last_ns < 2_000_000_000:
            return
        self._imu_warning_last_ns = now_ns
        self.get_logger().warning(message)

    def _publish_cached_transform(self) -> None:
        with self._state_lock:
            correction_rotation = None if self._latest_correction_rotation is None else self._latest_correction_rotation.copy()
            correction_translation = (
                None if self._latest_correction_translation is None else self._latest_correction_translation.copy()
            )

        if correction_rotation is None or correction_translation is None:
            return

        qx, qy, qz, qw = quaternion_from_rotation_matrix(correction_rotation)
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self._config.parent_frame
        transform.child_frame_id = self._config.child_frame
        transform.transform.translation.x = float(correction_translation[0])
        transform.transform.translation.y = float(correction_translation[1])
        transform.transform.translation.z = float(correction_translation[2])
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self._tf_broadcaster.sendTransform(transform)


def main() -> None:
    config = parse_args()
    ChannelFactoryInitialize, channel_subscriber_cls, dds_robot_posture_type = import_raw_dds_dependencies()
    ChannelFactoryInitialize(config.dds_domain_id, config.dds_interface)
    rclpy.init(args=None)

    node = FastLioFrameCorrectionNode(config, channel_subscriber_cls, dds_robot_posture_type)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()