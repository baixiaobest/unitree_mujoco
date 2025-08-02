import torch
from robot_communication import RobotCommunication
from environment import Environment
import math_utils
from joint_mapping import JointMapping

def base_lin_vel(env: Environment, robot_comm: RobotCommunication):
    """Extract base linear velocity from robot state"""
    vel_w = robot_comm.get_base_state()["velocity"]
    quat = robot_comm.get_base_state()["quaternion"]
    vel_b = math_utils.quat_rotate_inverse(quat.unsqueeze(0), vel_w.unsqueeze(0))[0]
    return vel_b

def base_ang_vel(env: Environment, robot_comm: RobotCommunication):
    """Extract base angular velocity (gyroscope) from robot state"""
    return robot_comm.get_base_state()["gyroscope"]

def projected_gravity(env: Environment, robot_comm: RobotCommunication):
    """Calculate gravity vector in robot frame using quaternion"""
    quat = robot_comm.get_base_state()["quaternion"]  # (w, x, y, z)
    gravity = torch.tensor([0.0, 0.0, -1.0], device=quat.device, dtype=torch.float32)
    projected = math_utils.quat_rotate_inverse(quat.unsqueeze(0), gravity.unsqueeze(0))[0]
    return projected

def pose_2d_command(env: Environment, robot_comm: RobotCommunication, command_name: str):
    """Get 2D pose command (usually from external command source)"""
    return env.command_manager.get_command(command_name)

def pose_2d_zero_command(env: Environment, robot_comm: RobotCommunication):
    """Get a zero 2D pose command (for testing or default state)"""
    return torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=robot_comm.device)

def joint_positions(env: Environment, 
                    robot_comm: RobotCommunication, 
                    jointMap: JointMapping,
                    scale: float = 1.0,
                    offset: torch.Tensor = None):
    """Extract joint positions from robot state"""
    return jointMap.unitree_to_policy(robot_comm.get_joint_state()["positions"], scale=scale, offset=offset)
    

def joint_velocities(env: Environment, 
                    robot_comm: RobotCommunication, 
                    jointMap: JointMapping,
                    scale: float = 1.0,
                    offset: torch.Tensor = None):
    """Extract joint velocities from robot state"""
    return jointMap.unitree_to_policy(robot_comm.get_joint_state()["velocities"], scale=scale, offset=offset)

def last_policy_output(env: Environment, robot_comm: RobotCommunication, offset: torch.Tensor = None):
    """Get the last action sent to the robot"""
    return env.last_policy_output

def constant_observation(env: Environment, robot_comm: RobotCommunication, value: torch.Tensor):
    """Return a constant observation tensor"""
    return value