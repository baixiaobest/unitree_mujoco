"""Wire definition for rt/learned_motion; keep identical to the bridge copy."""

from dataclasses import dataclass

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import uint8, uint32, uint64, float32


@dataclass
class LearnedMotion(IdlStruct, typename="go2_msgs.LearnedMotion"):
    tick: uint32
    sequence: uint64
    linear_x: float32
    linear_y: float32
    linear_z: float32
    angular_x: float32
    angular_y: float32
    angular_z: float32
    angular_velocity_source: uint8


LEARNED_ANGULAR_VELOCITY = 0
RAW_GYROSCOPE = 1
