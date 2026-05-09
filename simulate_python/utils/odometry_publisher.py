from __future__ import annotations

from time import time_ns

import torch

import utils.math_utils as math_utils
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.default import nav_msgs_msg_dds__Odometry_ as Odometry_default
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_


TOPIC_ESTIMATED_ODOMETRY = "rt/odom"


class EstimatedOdometryPublisher:
    """Publishes odometry using estimated body-frame linear velocity and IMU orientation."""

    def __init__(
        self,
        device: str | torch.device = "cpu",
        topic_name: str = TOPIC_ESTIMATED_ODOMETRY,
        odom_frame_id: str = "odom",
        child_frame_id: str = "base_link",
    ) -> None:
        self.device = torch.device(device)
        self.topic_name = topic_name
        self.odom_frame_id = odom_frame_id
        self.child_frame_id = child_frame_id

        self.publisher: ChannelPublisher = ChannelPublisher(self.topic_name, Odometry_)
        self.publisher.Init()

        self._integrated_position = torch.zeros(3, dtype=torch.float32, device=self.device)
        self._last_publish_time_ns: int | None = None

    def reset_time_reference(self) -> None:
        """Reset the integration clock without resetting the accumulated odom position."""
        self._last_publish_time_ns = None

    def publish(
        self,
        estimated_linear_velocity: torch.Tensor | None,
        base_quaternion: torch.Tensor,
        angular_velocity: torch.Tensor,
        publish_time_ns: int | None = None,
    ) -> bool:
        """Publish a DDS odometry message.

        The twist linear velocity is published in the base frame. Position is integrated in the odom frame by
        rotating the estimated body-frame linear velocity into the world frame using the IMU quaternion.
        """
        if estimated_linear_velocity is None:
            self.reset_time_reference()
            return False

        publish_time_ns = time_ns() if publish_time_ns is None else publish_time_ns
        dt = 0.0
        if self._last_publish_time_ns is not None and publish_time_ns > self._last_publish_time_ns:
            dt = (publish_time_ns - self._last_publish_time_ns) * 1e-9
        self._last_publish_time_ns = publish_time_ns

        estimated_linear_velocity = estimated_linear_velocity.detach().to(self.device)
        base_quaternion = base_quaternion.detach().to(self.device)
        angular_velocity = angular_velocity.detach().to(self.device)

        world_linear_velocity = math_utils.quat_rotate(
            base_quaternion.unsqueeze(0), estimated_linear_velocity.unsqueeze(0)
        ).squeeze(0)
        if dt > 0.0:
            self._integrated_position = self._integrated_position + world_linear_velocity * dt

        odometry_msg = Odometry_default()
        odometry_msg.header.stamp.sec = int(publish_time_ns // 1_000_000_000)
        odometry_msg.header.stamp.nanosec = int(publish_time_ns % 1_000_000_000)
        odometry_msg.header.frame_id = self.odom_frame_id
        odometry_msg.child_frame_id = self.child_frame_id

        odometry_msg.pose.pose.position.x = float(self._integrated_position[0].item())
        odometry_msg.pose.pose.position.y = float(self._integrated_position[1].item())
        odometry_msg.pose.pose.position.z = float(self._integrated_position[2].item())

        odometry_msg.pose.pose.orientation.w = float(base_quaternion[0].item())
        odometry_msg.pose.pose.orientation.x = float(base_quaternion[1].item())
        odometry_msg.pose.pose.orientation.y = float(base_quaternion[2].item())
        odometry_msg.pose.pose.orientation.z = float(base_quaternion[3].item())

        odometry_msg.twist.twist.linear.x = float(estimated_linear_velocity[0].item())
        odometry_msg.twist.twist.linear.y = float(estimated_linear_velocity[1].item())
        odometry_msg.twist.twist.linear.z = float(estimated_linear_velocity[2].item())

        odometry_msg.twist.twist.angular.x = float(angular_velocity[0].item())
        odometry_msg.twist.twist.angular.y = float(angular_velocity[1].item())
        odometry_msg.twist.twist.angular.z = float(angular_velocity[2].item())

        self.publisher.Write(odometry_msg)
        return True