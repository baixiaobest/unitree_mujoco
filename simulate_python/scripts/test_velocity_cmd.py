"""Manually drive rt/cmd_vel for testing, without a joystick.

``GameControllerPolicyHybridVelocityCommand`` (used by ``control_robot_hardware.py`` in
``--policy-mode velocity_control``) starts in CONTROLLER mode and only switches to POLICY mode via a
joystick button press, then reads its velocity command from the ``rt/cmd_vel`` DDS topic. This script
drives both DDS topics directly -- forcing POLICY mode on ``rt/locomotion_mode`` and publishing a
constant ``(vx, vy, wz)`` on ``rt/cmd_vel`` -- so the locomotion policy loop can be exercised
end-to-end with no joystick, no LiDAR, and no navigation/CBF stack involved.

Run alongside (in separate terminals):
    python3 unitree_mujoco.py
    python3 scripts/control_robot_hardware.py --run-mode simulation --policy-mode velocity_control \\
        --model-path <path to locomotion_policy_jit.pt>
    python3 scripts/test_velocity_cmd.py --vx 0.3
"""

import argparse
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.builtin_interfaces.msg.dds_ import Time_
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Twist_, TwistStamped_, Vector3_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import Header_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import UwbSwitch_

TOPIC_POLICY_VEL_CMD = "rt/cmd_vel"
TOPIC_LOCOMOTION_MODE = "rt/locomotion_mode"


def _twist_stamped(vx: float, vy: float, wz: float) -> TwistStamped_:
    now = time.time()
    return TwistStamped_(
        header=Header_(stamp=Time_(sec=int(now), nanosec=int((now % 1.0) * 1e9)), frame_id="base"),
        twist=Twist_(linear=Vector3_(x=vx, y=vy, z=0.0), angular=Vector3_(x=0.0, y=0.0, z=wz)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Manually drive rt/cmd_vel for testing.")
    parser.add_argument("--domain", type=int, default=1, help="DDS domain id (1=simulation, 0=hardware).")
    parser.add_argument("--interface", type=str, default="wlp128s20f3", help="Network interface.")
    parser.add_argument("--vx", type=float, default=0.3, help="Forward body-x velocity (m/s).")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral body-y velocity (m/s).")
    parser.add_argument("--wz", type=float, default=0.0, help="Yaw rate (rad/s).")
    parser.add_argument("--rate", type=float, default=20.0, help="Publish rate (Hz); must exceed the 0.5s timeout.")
    parser.add_argument("--duration", type=float, default=10.0, help="How long to publish, in seconds.")
    args = parser.parse_args()

    ChannelFactoryInitialize(args.domain, args.interface)

    mode_publisher = ChannelPublisher(TOPIC_LOCOMOTION_MODE, UwbSwitch_)
    mode_publisher.Init()
    mode_publisher.Write(UwbSwitch_(enabled=1))  # LocomotionMode.POLICY
    print("[INFO] Forced locomotion mode -> POLICY")

    vel_publisher = ChannelPublisher(TOPIC_POLICY_VEL_CMD, TwistStamped_)
    vel_publisher.Init()

    period = 1.0 / args.rate
    steps = int(args.duration / period)
    print(f"[INFO] Publishing (vx={args.vx}, vy={args.vy}, wz={args.wz}) at {args.rate} Hz for {args.duration}s")
    try:
        for _ in range(steps):
            vel_publisher.Write(_twist_stamped(args.vx, args.vy, args.wz))
            time.sleep(period)
    finally:
        print("[INFO] Stopping robot (publishing zero velocity)...")
        for _ in range(10):
            vel_publisher.Write(_twist_stamped(0.0, 0.0, 0.0))
            time.sleep(period)


if __name__ == "__main__":
    main()
