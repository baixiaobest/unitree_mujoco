import torch
from robot_communication import RobotCommunication
from scipy.spatial.transform import Rotation as R
from environment import Environment

def base_lin_vel(env: Environment, robot_comm: RobotCommunication):
    """Extract base linear velocity from robot state"""
    return robot_comm.get_base_state()["velocity"]

def base_ang_vel(env: Environment, robot_comm: RobotCommunication):
    """Extract base angular velocity (gyroscope) from robot state"""
    return robot_comm.get_base_state()["gyroscope"]

def projected_gravity(env: Environment, robot_comm: RobotCommunication):
    """Calculate gravity vector in robot frame using quaternion"""
    quat = robot_comm.get_base_state()["quaternion"]  # (w, x, y, z)
    gravity = torch.tensor([0.0, 0.0, -9.81], device=quat.device)
    # SciPy expects (x, y, z, w)
    quat_np = quat.cpu().numpy()
    r = R.from_quat([quat_np[1], quat_np[2], quat_np[3], quat_np[0]])
    projected = r.apply(gravity.cpu().numpy())
    return torch.tensor(projected, dtype=torch.float32, device=quat.device)

def pose_2d_command(env: Environment, robot_comm: RobotCommunication, command_name: str):
    """Get 2D pose command (usually from external command source)"""
    return env.command_manager.get_command(command_name)

def joint_positions(env: Environment, robot_comm):
    """Extract joint positions from robot state"""
    return robot_comm.get_joint_state()["positions"]

def joint_velocities(env: Environment, robot_comm):
    """Extract joint velocities from robot state"""
    return robot_comm.get_joint_state()["velocities"]

def last_action(env: Environment, robot_comm: RobotCommunication):
    """Get the last action sent to the robot"""
    # This assumes the last action was a position command
    previous_commands = robot_comm.get_previous_position_commands()
    if previous_commands is not None:
        return previous_commands
    else:
        return torch.zeros(12, device=robot_comm.device)