import torch
from robot_communication import RobotCommunication
from observations import *
from environment import Environment
from dataclasses import dataclass, field

@dataclass
class ObsItem:
    name: str
    function: callable
    dimension: int
    params: dict = field(default_factory=dict)

@dataclass
class ObservationConfig:
    observations: list[ObsItem]

    @property 
    def obs_functions(self):
        """Return the list of observation functions"""
        if self.observations is None:
            raise ValueError("Observation list 'self.obs' is not set in the config.")

        return [obs.function for obs in self.observations]

    def get_obs_dim(self):
        """Calculate total observation dimension from the obs tuples"""
        if self.observations is None:
            raise ValueError("Observation list 'self.obs' is not set in the config.")
        
        return sum(obs.dimension for obs in self.observations)
    
class ObservationManager:
    def __init__(self, env: Environment, observation_cfg: ObservationConfig, device: str = "cpu"):
        """Initialize the observation constructor
        
        Args:
            observation_cfg: Configuration specifying what observations to include
            device: PyTorch device to store tensors on
        """
        self.env = env
        self.observation_cfg = observation_cfg
        self.device = device
        self.robot_comm = env.robot_comm
        self.obs_map = {}
    
    def get_observation(self):
        """Construct observation by calling functions from observation config
        
        Returns:
            Tensor containing concatenated observation vectors
        """
        if self.robot_comm is None:
            raise ValueError("Robot communication instance not set. Call set_robot_comm first.")
        
        # Call each observation function in order and collect results
        obs_parts = []
        for obs_item in self.observation_cfg.observations:
            obs_part = obs_item.function(self.env, self.robot_comm, **obs_item.params)
            obs_parts.append(obs_part.flatten())  # Ensure each part is flattened
            self.obs_map[obs_item.name] = obs_part
        
        # Concatenate all observation parts into a single tensor
        return torch.cat(obs_parts)
    
    def get_observation_dim(self):
        """Get the total dimension of the observation vector"""
        return self.observation_cfg.get_obs_dim()
    
    def get_obs_map(self):
        """Get the mapping of observation names to their tensors"""
        return self.obs_map