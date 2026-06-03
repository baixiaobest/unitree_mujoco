from __future__ import annotations

from enum import IntEnum


TOPIC_POLICY_VEL_CMD = "rt/cmd_vel"
TOPIC_LOCOMOTION_MODE = "rt/locomotion_mode"


class LocomotionMode(IntEnum):
    CONTROLLER = 0
    POLICY = 1


def format_locomotion_mode(state: int | None) -> str:
    if state is None:
        return "unknown"
    try:
        mode = LocomotionMode(state)
    except ValueError:
        return f"unknown ({state})"
    if mode == LocomotionMode.CONTROLLER:
        return "controller"
    if mode == LocomotionMode.POLICY:
        return "policy"
    return f"unknown ({state})"
