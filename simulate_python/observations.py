import torch
from robot_communication import RobotCommunication
from environment import Environment
import math_utils

def base_lin_vel(env: Environment, robot_comm: RobotCommunication):
    """Extract base linear velocity from robot state"""
    return robot_comm.get_base_state()["velocity"]

def base_ang_vel(env: Environment, robot_comm: RobotCommunication):
    """Extract base angular velocity (gyroscope) from robot state"""
    return robot_comm.get_base_state()["gyroscope"]

def projected_gravity(env: Environment, robot_comm: RobotCommunication):
    """Calculate gravity vector in robot frame using quaternion"""
    quat = robot_comm.get_base_state()["quaternion"]  # (w, x, y, z)
    gravity = torch.tensor([0.0, 0.0, -1.0], device=quat.device, dtype=torch.float32)
    projected = math_utils.quat_rotate(quat.unsqueeze(0), gravity.unsqueeze(0))[0]
    return projected

def pose_2d_command(env: Environment, robot_comm: RobotCommunication, command_name: str):
    """Get 2D pose command (usually from external command source)"""
    return env.command_manager.get_command(command_name)

def pose_2d_zero_command(env: Environment, robot_comm: RobotCommunication):
    """Get a zero 2D pose command (for testing or default state)"""
    return torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=robot_comm.device)

def joint_positions(env: Environment, robot_comm: RobotCommunication, offset: torch.Tensor = None):
    """Extract joint positions from robot state"""
    if offset is not None:
        return robot_comm.get_joint_state()["positions"] - offset
    else:
        return robot_comm.get_joint_state()["positions"]

def joint_velocities(env: Environment, robot_comm: RobotCommunication, offset: torch.Tensor = None):
    """Extract joint velocities from robot state"""
    if offset is not None:
        return robot_comm.get_joint_state()["velocities"] - offset
    else:
        return robot_comm.get_joint_state()["velocities"]

def last_action(env: Environment, robot_comm: RobotCommunication, offset: torch.Tensor = None):
    """Get the last action sent to the robot"""
    # This assumes the last action was a position command
    previous_commands = robot_comm.get_previous_position_commands()
    if previous_commands is not None:
        if offset is not None:
            return previous_commands - offset
        else:
            return previous_commands
        return previous_commands
    else:
        return torch.zeros(12, device=robot_comm.device)