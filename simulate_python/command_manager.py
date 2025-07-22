from environment import Environment
from robot_communication import RobotCommunication
from dataclasses import dataclass
import torch
from scipy.spatial.transform import Rotation as R
import math

@dataclass
class CommandConfig:
    resample_interval: float

class Command:
    def __init__(self, env: Environment, cfg: CommandConfig, device: str = "cpu"):
        """Initialize the command with environment and robot communication
        
        Args:
            env: Environment instance to interact with
            robot_comm: RobotCommunication instance to get data from
        """
        self.env = env
        self.cfg = cfg
        self.robot_comm = env.robot_comm
        self.device = device

    @property
    def config(self):
        """Get the command configuration"""
        return self.cfg

    @property
    def command(self):
        raise NotImplementedError("This method should be implemented by subclasses")

    def update(self):
        """Update the command based on the robot's state"""
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def resample(self):
        """Resample the command if needed"""
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def get_dimension(self):
        """Get the dimension of the command"""
        return self.command.size()

@dataclass
class Pose2dCommandConfig(CommandConfig):
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]
    angle_range: tuple[float, float]

class Pose2dCommand(Command):
    def __init__(self, env: Environment, cfg: Pose2dCommandConfig, device: str = "cpu"):
        super().__init__(env, cfg, device)
        self._command = torch.zeros(4, device=device)
        self.command_w = torch.zeros(4, device=device)  # Command in world frame
        self.cfg = cfg

    @property
    def command(self):
        """Get the command in robot base frame"""
        return self._command

    def resample(self):
        """Resample the 2D pose command within specified ranges"""
        x = torch.rand(1, device=self._command.device) * (self.cfg.x_range[1] - self.cfg.x_range[0]) + self.cfg.x_range[0]
        y = torch.rand(1, device=self._command.device) * (self.cfg.y_range[1] - self.cfg.y_range[0]) + self.cfg.y_range[0]
        z = torch.rand(1, device=self._command.device) * (self.cfg.z_range[1] - self.cfg.z_range[0]) + self.cfg.z_range[0]
        angle = torch.rand(1, device=self._command.device) * (self.cfg.angle_range[1] - self.cfg.angle_range[0]) + self.cfg.angle_range[0]
        self.command_w = torch.tensor([x.item(), y.item(), z.item(), angle.item(), 0.0], device=self._command.device)

    def update(self):
        """Transform world-frame command to robot base frame"""
        # Get robot position and orientation
        base_state = self.robot_comm.get_base_state()
        robot_pos = base_state["position"]  # [x, y, z]
        robot_quat = base_state["quaternion"]  # [w, x, y, z]

        # Step 1: Translate - Subtract robot position from world position
        local_pos = self.command_w[:3] - robot_pos

        # Step 2: Rotate - Apply inverse quaternion rotation to the position
        # We need to convert from world frame to robot frame
        quat_np = robot_quat.cpu().numpy()

        r = R.from_quat([quat_np[1], quat_np[2], quat_np[3], quat_np[0]])
        local_pos_rotated = torch.tensor(
            r.inv().apply(local_pos.cpu().numpy()),
            device=self.device
        )

        # Step 3: Get yaw directly from scipy
        # Extract yaw (using 'ZYX' convention where z is yaw)
        euler_angles = r.as_euler('ZYX')
        robot_yaw = euler_angles[0]
        
        # Transform angle to robot frame
        local_angle = self.command_w[3] - robot_yaw
        local_angle = ((local_angle + math.pi) % (2 * math.pi)) - math.pi
        
        # Combine into final command vector
        self._command = torch.cat([local_pos_rotated, torch.tensor([local_angle], device=self.device)])


@dataclass
class CommandManagerConfig:
    commands: list[tuple[str, type[Command], CommandConfig]]

class CommandManager:
    def __init__(self, 
                 env: Environment,
                 manager_cfg: CommandManagerConfig,
                 device: str = "cpu"):
        
        self.env = env
        self.robot_comm = env.robot_comm
        self.device = device
        self.manager_cfg = manager_cfg

        self.command_instances = [(name, cmd_class(env, cmd_cfg, device)) for name, cmd_class, cmd_cfg  in manager_cfg.commands]
        self.prev_resample_time = torch.zeros(len(manager_cfg.commands), device=device)

        for _, cmd in self.command_instances:
            cmd.resample()  # Initialize each command
    
    def update(self):
        for i, (_, cmd) in enumerate(self.command_instances):
            if self.env.time_elapsed - self.prev_resample_time[i] > cmd.config.resample_interval:
                cmd.resample()
                self.prev_resample_time[i] = self.env.time_elapsed

            cmd.update()

    def get_command(self, name: str) -> torch.Tensor:
        """Get the command tensor by name"""
        for cmd_name, cmd_instance in self.command_instances:
            if cmd_name == name:
                return cmd_instance.command
        raise ValueError(f"Command '{name}' not found in CommandManager")