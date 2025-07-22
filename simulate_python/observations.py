import torch
from robot_communication import RobotCommunication
from scipy.spatial.transform import Rotation as R

def base_lin_vel(robot_comm: RobotCommunication):
    """Extract base linear velocity from robot state"""
    return robot_comm.get_base_state()["velocity"]

def base_ang_vel(robot_comm: RobotCommunication):
    """Extract base angular velocity (gyroscope) from robot state"""
    return robot_comm.get_base_state()["gyroscope"]

def projected_gravity(robot_comm: RobotCommunication):
    """Calculate gravity vector in robot frame using quaternion"""
    quat = robot_comm.get_base_state()["quaternion"]  # (w, x, y, z)
    gravity = torch.tensor([0.0, 0.0, -9.81], device=quat.device)
    # SciPy expects (x, y, z, w)
    quat_np = quat.cpu().numpy()
    r = R.from_quat([quat_np[1], quat_np[2], quat_np[3], quat_np[0]])
    projected = r.apply(gravity.cpu().numpy())
    return torch.tensor(projected, dtype=torch.float32, device=quat.device)

def pose_2d_command(robot_comm: RobotCommunication):
    """Get 2D pose command (usually from external command source)"""
    # This would typically come from a command input
    # For now returning zeros, you should replace this with actual command data
    return torch.zeros(4, device=robot_comm.device)

def pose_2d_command_random(robot_comm: RobotCommunication):
    """Generate random 2D pose command"""
    # Randomly generate a 2D pose command (x, y, theta)
    return (torch.rand(4, device=robot_comm.device) - 0.5) * torch.tensor([5, 5, 0, 2 * 3.14], device=robot_comm.device)

def joint_positions( robot_comm):
    """Extract joint positions from robot state"""
    return robot_comm.get_joint_state()["positions"]

def joint_velocities(robot_comm):
    """Extract joint velocities from robot state"""
    return robot_comm.get_joint_state()["velocities"]

def last_action(robot_comm: RobotCommunication):
    """Get the last action sent to the robot"""
    # This assumes the last action was a position command
    previous_commands = robot_comm.get_previous_position_commands()
    if previous_commands is not None:
        return previous_commands
    else:
        return torch.zeros(12, device=robot_comm.device)