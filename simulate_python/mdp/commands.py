from env.environment import Environment
from dataclasses import dataclass
import torch
import math
import utils.math_utils as math_utils
from utils.mujoco_visualizer import MujocoVisualizer
import pygame

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
    
    def setup(self):
        """Setup the command if needed"""
        pass

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
    
    def setup(self):
        base_state = self.robot_comm.get_base_state()
        robot_pos = base_state["position"]  # [x, y, z]
        robot_quat = base_state["quaternion"]  # [w, x, y, z]
        _, _, robot_yaw = math_utils.euler_xyz_from_quat(robot_quat.unsqueeze(0))
        robot_yaw = robot_yaw[0]
        self.command_w = torch.tensor([robot_pos[0], robot_pos[1], robot_pos[2], robot_yaw, 0.0], device=self.device, dtype=torch.float32)

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
    standing_height: float | None = None  # Optional standing height for command z
    mode: str = "local"  # Mode of operation: "local" or "global"
    a_button_index: int = 0  # Button index for A button (to set global position)

class GameControllerPose2dCommand(Pose2dCommand):
    """Pose2d command controlled by an Xbox controller"""
    def __init__(self, env: Environment, cfg: GameControllerPose2dCommandConfig, device: str = "cpu"):
        super().__init__(env, cfg, device)
        self.cfg = cfg
        self.has_controller = False
        self.controller = None
        
        # Global mode tracking
        self.global_position = None  # Fixed global position when set
        self.global_heading = None   # Fixed global heading when set
        self.a_button_pressed_prev = False  # Track button state to detect transitions
        
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
    
    def is_a_button_pressed(self):
        """Check if the A button is pressed"""
        if not self.has_controller:
            return False
        
        try:
            return self.controller.get_button(self.cfg.a_button_index)
        except Exception as e:
            print(f"Error reading A button: {e}")
            return False
    
    def calculate_command_position(self, robot_pos, robot_yaw, x_input, y_input):
        """Calculate command position based on stick input relative to robot position"""
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
        
        # Use standing_height if set, otherwise robot's z
        if self.cfg.standing_height is not None:
            z = self.cfg.standing_height
        else:
            z = robot_pos[2]
            
        # Calculate the heading
        if magnitude > 0:
            heading = world_angle  # Use same angle for heading
        else:
            heading = robot_yaw  # Keep current heading when no input
            
        return x, y, z, heading, magnitude > 0
    
    def handle_global_mode(self, robot_pos, robot_yaw, x_input, y_input):
        """Handle global mode command generation"""
        # Check if A button was just pressed (transition from not pressed to pressed)
        a_button_pressed = self.is_a_button_pressed()
        a_button_just_pressed = a_button_pressed and not self.a_button_pressed_prev
        self.a_button_pressed_prev = a_button_pressed
        
        # If A button was just pressed, set new global position based on stick input
        if a_button_just_pressed:
            x, y, z, heading, has_input = self.calculate_command_position(
                robot_pos, robot_yaw, x_input, y_input)
            
            if has_input:  # Only update if there's actual stick input
                self.global_position = torch.tensor([x, y, z], device=self.device)
                self.global_heading = heading
            else:
                # If no input, set to current robot position and heading
                self.global_position = torch.tensor(robot_pos, device=self.device)
                self.global_heading = robot_yaw
        
        # If global position is set, use it; otherwise use current robot position
        if self.global_position is not None:
            return (
                self.global_position[0],
                self.global_position[1],
                self.global_position[2],
                self.global_heading
            )
        else:
            # Default to current position if no global position is set
            if self.cfg.standing_height is not None:
                z = self.cfg.standing_height
            else:
                z = robot_pos[2]
            return robot_pos[0], robot_pos[1], z, robot_yaw
    
    def handle_local_mode(self, robot_pos, robot_yaw, x_input, y_input):
        """Handle local mode command generation"""
        x, y, z, heading, _ = self.calculate_command_position(
            robot_pos, robot_yaw, x_input, y_input)
        return x, y, z, heading
    
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
        
        # Generate command based on mode
        if self.cfg.mode == "global":
            x, y, z, heading = self.handle_global_mode(robot_pos, robot_yaw, x_input, y_input)
        else:  # local mode
            x, y, z, heading = self.handle_local_mode(robot_pos, robot_yaw, x_input, y_input)
        
        self.command_w = torch.tensor([x, y, z, heading, 0.0], device=self._command.device, dtype=torch.float32)


@dataclass
class WasdKeyboardCommandConfig(Pose2dCommandConfig):
    """Configuration for WasdKeyboardCommand"""
    command_distance: float = 2.0  # Distance of the command point from robot
    command_turn_distance: float = 1.0  # Distance for turn commands
    standing_height: float = 0.3  # Height of the command point when standing
    input_hold_time: float = 0.5  # Time to hold the command before allowing changes
    rotate_angle: float = 30
    visualize: bool = True
    mode: str = "local"  # Mode of operation: "local" or "global"

class WasdKeyboardCommand(Pose2dCommand):
    """Pose2d command controlled by WASD keyboard keys"""
    def __init__(self, env: Environment, cfg: WasdKeyboardCommandConfig, device: str = "cpu"):
        super().__init__(env, cfg, device)
        self.cfg = cfg
        self.initialized = False
        self.last_key = None
        self.last_key_time = 0
        self.keys_pressed = {
            pygame.K_w: False,
            pygame.K_a: False,
            pygame.K_s: False,
            pygame.K_d: False
        }
        
        # Global mode tracking
        self.active_global_command = False
        self.global_command_key = None
        self.global_command_position = None
        self.global_command_heading = None
        
        # Track key release to update command only on release
        self.key_just_released = False
        self.command_set_after_release = False
        
        # Initialize pygame for key handling
        self._init_pygame()
    
    def _init_pygame(self):
        """Initialize pygame for keyboard input"""
        try:
            if not pygame.get_init():
                pygame.init()
            pygame.display.set_mode((100, 100))
            pygame.display.set_caption("WASD Control")
            self.initialized = True
            print("Keyboard control initialized")
        except Exception as e:
            print(f"Error initializing pygame: {e}")
            self.initialized = False
    
    def read_keyboard_input(self):
        """Read input from keyboard"""
        if not self.initialized:
            self._init_pygame()
            if not self.initialized:
                return None
        
        try:
            # Reset key_just_released flag
            self.key_just_released = False
            
            # Process events
            any_key_pressed = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
                elif event.type == pygame.KEYDOWN:
                    if event.key in self.keys_pressed:
                        self.keys_pressed[event.key] = True
                        self.last_key = event.key
                        self.last_key_time = pygame.time.get_ticks() / 1000.0
                        self.command_set_after_release = False
                        if self.cfg.mode == "global":
                            self.global_command_key = event.key
                elif event.type == pygame.KEYUP:
                    if event.key in self.keys_pressed:
                        self.keys_pressed[event.key] = False
                        self.key_just_released = True
                        self.command_set_after_release = False
                        # If in global mode and we released the active key, reset global command
                        if self.cfg.mode == "global" and event.key == self.global_command_key:
                            self.active_global_command = False
                            self.global_command_key = None
                            self.global_command_position = None
                            self.global_command_heading = None
        
            # Check if any key is pressed
            any_key_pressed = any(self.keys_pressed.values())
            if not any_key_pressed and self.cfg.mode == "global":
                # If no keys are pressed, reset global command
                self.active_global_command = False
                self.global_command_key = None
                self.global_command_position = None
                self.global_command_heading = None
            
            # Return the current active key
            current_time = pygame.time.get_ticks() / 1000.0
            if current_time - self.last_key_time < self.cfg.input_hold_time:
                return self.last_key
            
            # If we're in global mode with an active command, return that key
            if self.cfg.mode == "global" and self.active_global_command:
                return self.global_command_key
            
            # Check for currently pressed keys (priority: W, A, S, D)
            if self.keys_pressed[pygame.K_w]:
                self.last_key = pygame.K_w
                self.last_key_time = current_time
                return pygame.K_w
            elif self.keys_pressed[pygame.K_a]:
                self.last_key = pygame.K_a
                self.last_key_time = current_time
                return pygame.K_a
            elif self.keys_pressed[pygame.K_s]:
                self.last_key = pygame.K_s
                self.last_key_time = current_time
                return pygame.K_s
            elif self.keys_pressed[pygame.K_d]:
                self.last_key = pygame.K_d
                self.last_key_time = current_time
                return pygame.K_d
            
            # Return None if no keys are currently pressed and hold time has expired
            return None
        except Exception as e:
            print(f"Error reading keyboard: {e}")
            return None
    
    def resample(self):
        """Override resample to read from keyboard instead of random sampling"""
        # Get robot position and orientation
        base_state = self.robot_comm.get_base_state()
        robot_pos = base_state["position"]  # [x, y, z]
        robot_quat = base_state["quaternion"]  # [w, x, y, z]
        
        # Get robot's current yaw
        _, _, robot_yaw = math_utils.euler_xyz_from_quat(robot_quat.unsqueeze(0))
        robot_yaw = robot_yaw[0]
        
        # Read keyboard input
        key = self.read_keyboard_input()
        
        # Default to current position and orientation
        x, y, z = robot_pos[0], robot_pos[1], self.cfg.standing_height
        heading = robot_yaw
        
        # Handle key presses for both modes
        if key is not None:
            # Check if we need to store a new global command
            is_new_global_command = (self.cfg.mode == "global" and 
                                    (not self.active_global_command or 
                                     key != self.global_command_key))
            
            # If in global mode with an active global command, use the stored command
            if self.cfg.mode == "global" and self.active_global_command and self.global_command_position is not None and not is_new_global_command:
                x, y, z = self.global_command_position
                heading = self.global_command_heading
            else:
                # Calculate new command positions based on current key
                if key == pygame.K_w:  # Forward
                    distance = self.cfg.command_distance
                    x = robot_pos[0] + distance * math.cos(robot_yaw)
                    y = robot_pos[1] + distance * math.sin(robot_yaw)
                    heading = robot_yaw  # Keep current heading
                    
                elif key == pygame.K_a:  # Left rotation (30 degrees)
                    angle = math.radians(self.cfg.rotate_angle) + robot_yaw
                    x = robot_pos[0] + self.cfg.command_turn_distance * math.cos(angle)
                    y = robot_pos[1] + self.cfg.command_turn_distance * math.sin(angle)
                    heading = angle  # 30 degrees left rotation
                    
                elif key == pygame.K_s:  # Backward
                    distance = self.cfg.command_distance
                    back_angle = robot_yaw + math.pi
                    x = robot_pos[0] + distance * math.cos(back_angle)
                    y = robot_pos[1] + distance * math.sin(back_angle)
                    heading = back_angle  # Turn around
                    
                elif key == pygame.K_d:  # Right rotation (30 degrees)
                    angle = -math.radians(self.cfg.rotate_angle) + robot_yaw
                    x = robot_pos[0] + self.cfg.command_turn_distance * math.cos(angle)
                    y = robot_pos[1] + self.cfg.command_turn_distance * math.sin(angle)
                    heading = angle  # 30 degrees right rotation
            
            # If in global mode, store the new command
            if is_new_global_command:
                self.active_global_command = True
                self.global_command_key = key
                self.global_command_position = torch.tensor([x, y, z], device=self.device)
                self.global_command_heading = heading
            
            # Mark that we've handled a key press
            self.command_set_after_release = False
        elif key is None:
            # Only set command to current position when a key is first released
            # if self.cfg.mode == "global":
            x, y, z = robot_pos[0], robot_pos[1], self.cfg.standing_height
            heading = robot_yaw
            self.command_set_after_release = True
        # elif key is None and not self.key_just_released:
        #     # If no key is pressed and it's not a new release, keep the previous command
        #     if self.cfg.mode == "global":
        #         return  # Don't update command
    
        # Set the command in world frame
        self.command_w = torch.tensor([x, y, z, heading, 0.0], device=self._command.device, dtype=torch.float32)