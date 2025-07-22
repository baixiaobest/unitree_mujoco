from typing import Dict, List, Union, Optional
import torch
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher

# Import the message types based on robot type
import config
if config.ROBOT=="g1":
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmd_default
else:
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_ as LowCmd_default

from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

class RobotCommunication:
    """Handles all communication with the robot (state subscription and command publishing)"""
    
    def __init__(self, device: str = "cpu") -> None:
        """Initialize communication channels for robot control
        
        Args:
            device: PyTorch device to store tensors on (default: 'cpu')
        """
        self.device = torch.device(device)
        
        # Global state storage as torch tensors
        self.robot_joint_state: Dict[str, torch.Tensor] = {
            "positions": torch.zeros(0, dtype=torch.float32, device=self.device),
            "velocities": torch.zeros(0, dtype=torch.float32, device=self.device),
            "torques": torch.zeros(0, dtype=torch.float32, device=self.device)
        }
        
        self.robot_base_state: Dict[str, torch.Tensor] = {
            "position": torch.zeros(3, dtype=torch.float32, device=self.device),
            "velocity": torch.zeros(3, dtype=torch.float32, device=self.device),
            "quaternion": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
            "gyroscope": torch.zeros(3, dtype=torch.float32, device=self.device),
            "accelerometer": torch.zeros(3, dtype=torch.float32, device=self.device)
        }
        
        # Initialize channel factory
        ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
        
        # Initialize subscribers for robot state
        self.low_state_subscriber: ChannelSubscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_subscriber.Init(self._low_state_handler, 10)
        
        self.high_state_subscriber: ChannelSubscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.high_state_subscriber.Init(self._high_state_handler, 10)
        
        # Initialize command publisher
        self.low_cmd_publisher: ChannelPublisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.low_cmd_publisher.Init()
    
    def _low_state_handler(self, msg: LowState_) -> None:
        """Handler for low state messages from the robot
        
        Args:
            msg: Low-level state message from the robot
        """
        # Get joint positions, velocities, and torques
        positions: List[float] = []
        velocities: List[float] = []
        torques: List[float] = []
        
        for motor_state in msg.motor_state:
            positions.append(motor_state.q)
            velocities.append(motor_state.dq)
            torques.append(motor_state.tau_est)
        
        # Update global state with torch tensors
        self.robot_joint_state["positions"] = torch.tensor(
            positions, dtype=torch.float32, device=self.device
        )
        self.robot_joint_state["velocities"] = torch.tensor(
            velocities, dtype=torch.float32, device=self.device
        )
        self.robot_joint_state["torques"] = torch.tensor(
            torques, dtype=torch.float32, device=self.device
        )
        
        # Update IMU data with torch tensors
        self.robot_base_state["quaternion"] = torch.tensor(
            list(msg.imu_state.quaternion), dtype=torch.float32, device=self.device
        )
        self.robot_base_state["gyroscope"] = torch.tensor(
            list(msg.imu_state.gyroscope), dtype=torch.float32, device=self.device
        )
        self.robot_base_state["accelerometer"] = torch.tensor(
            list(msg.imu_state.accelerometer), dtype=torch.float32, device=self.device
        )
    
    def _high_state_handler(self, msg: SportModeState_) -> None:
        """Handler for high state messages from the robot
        
        Args:
            msg: High-level state message from the robot
        """
        # Update global base state with torch tensors
        self.robot_base_state["position"] = torch.tensor(
            list(msg.position), dtype=torch.float32, device=self.device
        )
        self.robot_base_state["velocity"] = torch.tensor(
            list(msg.velocity), dtype=torch.float32, device=self.device
        )
    
    def get_joint_state(self) -> Dict[str, torch.Tensor]:
        """Get the current robot joint state from subscribers
        
        Returns:
            Dictionary containing joint positions, velocities, and torques as tensors
        """
        return self.robot_joint_state
    
    def get_base_state(self) -> Dict[str, torch.Tensor]:
        """Get the robot base state from subscribers
        
        Returns:
            Dictionary containing base position, velocity, orientation, and IMU data as tensors
        """
        return self.robot_base_state
    
    def send_position_commands(
        self, 
        desired_positions: Union[List[float], torch.Tensor], 
        num_joints: int, 
        kp: float = 30.0, 
        kd: float = 2.0
    ) -> bool:
        """Send joint position commands to the robot
        
        Args:
            desired_positions: PyTorch tensor or list of desired joint positions
            num_joints: Number of joints to control
            kp: Position gain (default: 30.0)
            kd: Velocity gain (default: 2.0)
            
        Returns:
            True if command was sent successfully, False otherwise
        """
        try:
            # Create a LowCmd message
            cmd = LowCmd_default()
            
            # Set header and mode
            cmd.head[0] = 0xFE
            cmd.head[1] = 0xEF
            cmd.level_flag = 0xFF
            
            # Convert tensor to list if needed
            if isinstance(desired_positions, torch.Tensor):
                desired_positions = desired_positions.cpu().tolist()
            
            # Set joint commands
            for i in range(min(num_joints, len(cmd.motor_cmd))):
                cmd.motor_cmd[i].mode = 0x0A  # Position control mode
                cmd.motor_cmd[i].q = desired_positions[i]
                cmd.motor_cmd[i].kp = kp
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].kd = kd
                cmd.motor_cmd[i].tau = 0.0
            
            # Publish the command - this will be received by the bridge's LowCmdHandler
            self.low_cmd_publisher.Write(cmd)
            return True
            
        except Exception as e:
            print(f"Failed to send commands: {e}")
            return False