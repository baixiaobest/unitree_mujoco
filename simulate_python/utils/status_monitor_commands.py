from __future__ import annotations

from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_ as WirelessController_default
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_


TOPIC_STATUS_MONITOR_COMMAND = "rt/status_monitor_command"

# Reuse the Unitree wireless-controller bit layout for button-style commands.
_BUTTON_A_MASK = 1 << 8
_BUTTON_B_MASK = 1 << 9

COMMAND_STAND_UP = "stand_up"
COMMAND_LAY_DOWN = "lay_down"


def create_status_monitor_command(command_name: str):
    """Create a one-shot DDS command message for the status monitor control channel."""
    msg = WirelessController_default()
    msg.keys = 0
    msg.lx = 0.0
    msg.ly = 0.0
    msg.rx = 0.0
    msg.ry = 0.0

    if command_name == COMMAND_STAND_UP:
        msg.keys = _BUTTON_A_MASK
    elif command_name == COMMAND_LAY_DOWN:
        msg.keys = _BUTTON_B_MASK
    else:
        raise ValueError(f"Unsupported status monitor command: {command_name}")

    return msg


def decode_status_monitor_command(keys: int) -> str | None:
    """Decode a status monitor DDS command from the button-bit field."""
    has_stand_up = bool(keys & _BUTTON_A_MASK)
    has_lay_down = bool(keys & _BUTTON_B_MASK)

    if has_stand_up and not has_lay_down:
        return COMMAND_STAND_UP
    if has_lay_down and not has_stand_up:
        return COMMAND_LAY_DOWN
    return None
