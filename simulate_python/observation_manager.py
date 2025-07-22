import torch
from robot_communication import RobotCommunication
from observations import *

class ObservationConfig:
    def __init__(self):
        self.obs = None

    @property 
    def obs_functions(self):
        """Return the list of observation functions"""
        if self.obs is None:
            raise ValueError("Observation list 'self.obs' is not set in the config.")

        return [obs_tuple[1] for obs_tuple in self.obs]

    def get_obs_dim(self):
        """Calculate total observation dimension from the obs tuples"""
        if self.obs is None:
            raise ValueError("Observation list 'self.obs' is not set in the config.")
        
        return sum(obs_tuple[2] for obs_tuple in self.obs)

class E2EObservationConfig(ObservationConfig):
    def __init__(self):
        super().__init__()
        self.obs = [
            ("base_linear_velocity", base_lin_vel, 3),
            ("base_angular_velocity", base_ang_vel, 3),
            ("projected_gravity", projected_gravity, 3),
            ("pose_2d_command_random", pose_2d_command_random, 4),
            ("joint_positions", joint_positions, 12),
            ("joint_velocities", joint_velocities, 12),
            ("last_action", last_action, 12),
        ]
    
class ObservationManager:
    def __init__(self, observation_cfg: ObservationConfig, robot_comm: RobotCommunication, device: str = "cpu"):
        """Initialize the observation constructor
        
        Args:
            observation_cfg: Configuration specifying what observations to include
            device: PyTorch device to store tensors on
        """
        self.observation_cfg = observation_cfg
        self.device = device
        self.robot_comm = robot_comm
    
    def set_robot_comm(self, robot_comm: RobotCommunication):
        """Set the robot communication instance
        
        Args:
            robot_comm: RobotCommunication instance to get data from
        """
        self.robot_comm = robot_comm
    
    def get_observation(self):
        """Construct observation by calling functions from observation config
        
        Returns:
            Tensor containing concatenated observation vectors
        """
        if self.robot_comm is None:
            raise ValueError("Robot communication instance not set. Call set_robot_comm first.")
        
        # Call each observation function in order and collect results
        obs_parts = []
        for obs_func in self.observation_cfg.obs_functions:
            obs_part = obs_func(self.robot_comm)
            obs_parts.append(obs_part.flatten())  # Ensure each part is flattened
        
        # Concatenate all observation parts into a single tensor
        return torch.cat(obs_parts)
    
    def get_observation_dim(self):
        """Get the total dimension of the observation vector"""
        return self.observation_cfg.get_obs_dim()