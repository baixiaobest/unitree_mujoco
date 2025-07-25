from environment import Environment
from dataclasses import dataclass
import torch
import math
import math_utils
from mujoco_visualizer import MujocoVisualizer


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
    
    def visualize(self, visualizer: MujocoVisualizer):
        """Visualize the command in the environment"""
        raise NotImplementedError("This method should be implemented by subclasses")

@dataclass
class Pose2dCommandConfig(CommandConfig):
    x_range: tuple[float, float]  = (-5.0, 5.0)
    y_range: tuple[float, float] = (-5.0, 5.0)
    z_range: tuple[float, float] = (0.4, 0.4)
    angle_range: tuple[float, float] = (-math.pi, math.pi)
    visualize: bool = False

class Pose2dCommand(Command):
    def __init__(self, env: Environment, cfg: Pose2dCommandConfig, device: str = "cpu"):
        super().__init__(env, cfg, device)
        self._command = torch.zeros(4, device=device, dtype=torch.float32)  # Command in robot base frame
        self.command_w = torch.zeros(4, device=device, dtype=torch.float32)  # Command in world frame
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
        self.command_w = torch.tensor([x.item(), y.item(), z.item(), angle.item(), 0.0], device=self._command.device, dtype=torch.float32)

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
        local_pos_rotated = math_utils.quat_rotate_inverse(robot_quat.unsqueeze(0), local_pos.unsqueeze(0))[0]

        # Step 3: Get yaw 
        _, _, robot_yaw = math_utils.euler_xyz_from_quat(robot_quat.unsqueeze(0))
        robot_yaw = robot_yaw[0]
        
        # Transform angle to robot frame
        local_angle = self.command_w[3] - robot_yaw
        local_angle = ((local_angle + math.pi) % (2 * math.pi)) - math.pi
        
        # Combine into final command vector
        self._command = torch.cat([local_pos_rotated, torch.tensor([local_angle], device=self.device, dtype=torch.float32)])

    def visualize(self, visualizer: MujocoVisualizer):
        """Visualize the 2D pose command in the environment"""
        if not self.cfg.visualize:
            return
        
        # Get command position and yaw
        command_pos = self.command_w[:3]
        command_yaw = self.command_w[3]
        arrow_dir = torch.tensor([1.0, 0.0, 0.0]).to(device=self.device, dtype=torch.float32)

        quat = math_utils.quat_from_euler_xyz(
            torch.zeros(1, device=self.device), 
            torch.zeros(1, device=self.device), 
            command_yaw.unsqueeze(0))
        
        arrow_dir = math_utils.quat_rotate(quat, arrow_dir.unsqueeze(0))[0]

        visualizer.add_arrow(command_pos.cpu().numpy(), 
                             (command_pos + arrow_dir).cpu().numpy(), 
                             size=MujocoVisualizer.DEFAULT_ARROW_SIZE, 
                             color=MujocoVisualizer.GREEN)

@dataclass
class GameControllerPose2dCommandConfig(Pose2dCommandConfig):
    """Configuration for GameControllerPose2dCommand"""
    max_distance: float = 5.0  # Maximum distance of command from robot position
    controller_index: int = 0  # Index of the controller to use
    joystick_deadzone: float = 0.1  # Deadzone for joystick input
    x_axis: int = 0  # Controller axis index for X movement (typically left stick X)
    y_axis: int = 1  # Controller axis index for Y movement (typically left stick Y)

class GameControllerPose2dCommand(Pose2dCommand):
    """Pose2d command controlled by an Xbox controller"""
    def __init__(self, env: Environment, cfg: GameControllerPose2dCommandConfig, device: str = "cpu"):
        super().__init__(env, cfg, device)
        self.cfg = cfg
        self.has_controller = False
        self.controller = None
        
        # Initialize controller
        self._init_controller()
    
    def _init_controller(self):
        """Initialize the game controller"""
        try:
            import pygame
            if not pygame.get_init():
                pygame.init()
            if not pygame.joystick.get_init():
                pygame.joystick.init()
            
            if pygame.joystick.get_count() > self.cfg.controller_index:
                self.controller = pygame.joystick.Joystick(self.cfg.controller_index)
                self.controller.init()
                self.has_controller = True
                print(f"Controller initialized: {self.controller.get_name()}")
            else:
                print(f"No controller found at index {self.cfg.controller_index}")
                self.has_controller = False
        except ImportError:
            print("pygame not available. Install with 'pip install pygame'")
            self.has_controller = False
        except Exception as e:
            print(f"Error initializing controller: {e}")
            self.has_controller = False
    
    def read_controller_input(self):
        """Read input from the game controller"""
        if not self.has_controller:
            # Try to initialize the controller if it's not available
            self._init_controller()
            if not self.has_controller:
                return 0.0, 0.0  # Default to no movement
        
        try:
            import pygame
            pygame.event.pump()  # Process event queue
            
            # Read joystick axes
            x = self.controller.get_axis(self.cfg.x_axis)
            y = self.controller.get_axis(self.cfg.y_axis)
            
            # Invert X since positive X is pointing backward
            x = -x
            y = -y
            
            # Apply deadzone
            if abs(x) < self.cfg.joystick_deadzone:
                x = 0.0
            if abs(y) < self.cfg.joystick_deadzone:
                y = 0.0
                
            return x, y
        except Exception as e:
            print(f"Error reading controller: {e}")
            self.has_controller = False
            return 0.0, 0.0
    
    def resample(self):
        """Override resample to read from controller instead of random sampling"""
        # Get robot position and orientation
        base_state = self.robot_comm.get_base_state()
        robot_pos = base_state["position"]  # [x, y, z]
        robot_quat = base_state["quaternion"]  # [w, x, y, z]
        
        # Get robot's current yaw
        _, _, robot_yaw = math_utils.euler_xyz_from_quat(robot_quat.unsqueeze(0))
        robot_yaw = robot_yaw[0]
        
        # Read controller input
        x_input, y_input = self.read_controller_input()
        print(f"Controller input: x={x_input}, y={y_input}")
        
        # Calculate distance based on stick position (magnitude)
        magnitude = min(1.0, math.sqrt(x_input**2 + y_input**2))
        distance = magnitude * self.cfg.max_distance
        
        # Calculate joystick angle in robot's local frame
        if magnitude > 0:
            local_angle = math.atan2(y_input, x_input)
            
            # Convert to world frame by adding robot's yaw
            world_angle = local_angle + robot_yaw
        else:
            # Default to robot's current orientation if stick is centered
            world_angle = robot_yaw
        
        # Calculate command position relative to robot position
        x = robot_pos[0] + distance * math.cos(world_angle)
        y = robot_pos[1] + distance * math.sin(world_angle)
        z = robot_pos[2]  # Keep the same height as the robot
        
        # Calculate the heading to point from robot to commanded position
        if magnitude > 0:
            heading = world_angle  # Use same angle for heading
        else:
            heading = robot_yaw  # Keep current heading when no input
        
        # Set the command in world frame
        self.command_w = torch.tensor([x, y, z, heading, 0.0], device=self._command.device, dtype=torch.float32)