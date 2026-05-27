from __future__ import annotations

from enum import IntEnum


TOPIC_ROBOT_POSTURE = "rt/robot_posture"


class RobotPostureState(IntEnum):
    LAID_DOWN = 0
    STANDING = 1
    TRANSITIONING_TO_STAND = 2
    TRANSITIONING_TO_LAY = 3


def format_robot_posture_state(state: int | None) -> str:
    if state is None:
        return "unknown"
    try:
        posture_state = RobotPostureState(state)
    except ValueError:
        return f"unknown ({state})"

    if posture_state == RobotPostureState.LAID_DOWN:
        return "laid down"
    if posture_state == RobotPostureState.STANDING:
        return "standing"
    if posture_state == RobotPostureState.TRANSITIONING_TO_STAND:
        return "transitioning to stand"
    if posture_state == RobotPostureState.TRANSITIONING_TO_LAY:
        return "transitioning to lay"
    return f"unknown ({state})"