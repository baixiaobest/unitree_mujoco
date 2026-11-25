from __future__ import annotations

import torch
from unitree_sdk2py.core.channel import ChannelPublisher

from utils.learned_motion_dds import LearnedMotion, LEARNED_ANGULAR_VELOCITY, RAW_GYROSCOPE


TOPIC_LEARNED_MOTION = "rt/learned_motion"
# ROS topics published by imu_bridge. Keep these names for DDS subscribers
# such as status_monitor; this controller does not publish on either topic.
TOPIC_ESTIMATED_ODOMETRY = "rt/odom"
TOPIC_ESTIMATED_BODY_VELOCITY = "rt/estimated_velocity"


class EstimatedOdometryPublisher:
    """Publish sensor-timed body-frame motion for the ROS IMU bridge to integrate."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.publisher = ChannelPublisher(TOPIC_LEARNED_MOTION, LearnedMotion)
        self.publisher.Init()
        self._sequence = 0

    def reset_time_reference(self) -> None:
        """The receiver owns integration time; retained for controller callers."""

    def publish(
        self,
        estimated_linear_velocity: torch.Tensor | None,
        angular_velocity: torch.Tensor,
        tick: int | None,
        angular_velocity_is_learned: bool,
    ) -> bool:
        if estimated_linear_velocity is None or tick is None:
            return False
        linear = estimated_linear_velocity.detach().to("cpu").tolist()
        angular = angular_velocity.detach().to("cpu").tolist()
        if len(linear) != 3 or len(angular) != 3:
            raise ValueError("Motion estimates must each have three components")
        msg = LearnedMotion(
            tick=int(tick) & 0xFFFFFFFF,
            sequence=self._sequence,
            linear_x=float(linear[0]), linear_y=float(linear[1]), linear_z=float(linear[2]),
            angular_x=float(angular[0]), angular_y=float(angular[1]), angular_z=float(angular[2]),
            angular_velocity_source=(LEARNED_ANGULAR_VELOCITY if angular_velocity_is_learned else RAW_GYROSCOPE),
        )
        self.publisher.Write(msg)
        self._sequence += 1
        return True
