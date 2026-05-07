from collections import deque
from dataclasses import dataclass, field

import torch

from env.environment import Environment
from mdp.observations import *

@dataclass
class ObsItem:
    name: str
    function: callable
    dimension: int
    params: dict = field(default_factory=dict)
    use_history: bool = False

@dataclass
class ObservationConfig:
    observations: list[ObsItem]
    layouts: dict[str, list[str]] = field(default_factory=dict)
    history_length: int = 1
    default_layout: str = "default"

    def __post_init__(self):
        if self.observations is None:
            raise ValueError("Observation list 'self.observations' is not set in the config.")

        observation_names = [obs.name for obs in self.observations]
        duplicate_names = {name for name in observation_names if observation_names.count(name) > 1}
        if duplicate_names:
            duplicate_names_str = ", ".join(sorted(duplicate_names))
            raise ValueError(f"Duplicate observation names are not allowed: {duplicate_names_str}")

        if not self.layouts:
            self.layouts = {self.default_layout: observation_names}

        if self.default_layout not in self.layouts:
            raise ValueError(f"Default layout '{self.default_layout}' is not defined in the observation config.")

        unknown_names = set()
        known_names = set(observation_names)
        for layout_name, layout_terms in self.layouts.items():
            if not layout_terms:
                raise ValueError(f"Observation layout '{layout_name}' must contain at least one observation term.")
            unknown_names.update(term_name for term_name in layout_terms if term_name not in known_names)

        if unknown_names:
            unknown_names_str = ", ".join(sorted(unknown_names))
            raise ValueError(f"Observation layouts reference unknown observation terms: {unknown_names_str}")

    @property 
    def obs_functions(self):
        """Return the list of observation functions"""
        if self.observations is None:
            raise ValueError("Observation list 'self.obs' is not set in the config.")

        return [obs.function for obs in self.observations]

    @property
    def observations_by_name(self):
        return {obs.name: obs for obs in self.observations}

    def get_layout_terms(self, layout_name: str | None = None):
        layout_key = self.default_layout if layout_name is None else layout_name
        if layout_key not in self.layouts:
            raise ValueError(f"Observation layout '{layout_key}' is not defined.")
        return self.layouts[layout_key]

    def get_obs_dim(self, layout_name: str | None = None):
        """Calculate total observation dimension from the obs tuples"""
        if self.observations is None:
            raise ValueError("Observation list 'self.obs' is not set in the config.")

        observations_by_name = self.observations_by_name
        return sum(observations_by_name[obs_name].dimension for obs_name in self.get_layout_terms(layout_name))

    def get_layout_slices(self, layout_name: str | None = None):
        observations_by_name = self.observations_by_name
        slices = {}
        offset = 0
        for obs_name in self.get_layout_terms(layout_name):
            obs_item = observations_by_name[obs_name]
            slices[obs_name] = slice(offset, offset + obs_item.dimension)
            offset += obs_item.dimension
        return slices
    
class ObservationManager:
    def __init__(self, env: Environment, observation_cfg: ObservationConfig, device: str = "cpu", debug=False):
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
        self.debug = debug
        self.observations_by_name = self.observation_cfg.observations_by_name
        self.layout_slices = {
            layout_name: self.observation_cfg.get_layout_slices(layout_name)
            for layout_name in self.observation_cfg.layouts
        }
        self.history_buffers = {
            obs_item.name: deque(maxlen=self.observation_cfg.history_length)
            for obs_item in self.observation_cfg.observations
            if obs_item.use_history
        }

    def update_observations(self):
        """Refresh the current observation map and any configured history buffers."""
        if self.robot_comm is None:
            raise ValueError("Robot communication instance not set. Call set_robot_comm first.")

        current_obs_map = {}
        for obs_item in self.observation_cfg.observations:
            obs_part = obs_item.function(self.env, self.robot_comm, **obs_item.params).flatten()
            current_obs_map[obs_item.name] = obs_part

            if obs_item.use_history:
                self.history_buffers[obs_item.name].append(obs_part.detach().clone())

            if self.debug:
                print(f"Observation '{obs_item.name}': {obs_part.cpu().numpy()}")

        self.obs_map = current_obs_map
        return current_obs_map

    def _get_history_tensor(self, obs_name: str, history_length: int):
        history_buffer = self.history_buffers.get(obs_name)
        if history_buffer is None or len(history_buffer) == 0:
            if obs_name not in self.obs_map:
                raise RuntimeError(f"Observation '{obs_name}' is not available in the current observation map.")
            current_value = self.obs_map[obs_name]
            return current_value.unsqueeze(0).repeat(history_length, 1)

        history_values = list(history_buffer)
        while len(history_values) < history_length:
            history_values.insert(0, history_values[0].clone())
        history_values = history_values[-history_length:]
        return torch.stack(history_values, dim=0)

    def _build_layout_observation(self, layout_name: str | None = None):
        layout_terms = self.observation_cfg.get_layout_terms(layout_name)
        obs_parts = [self.obs_map[obs_name] for obs_name in layout_terms]
        return torch.cat(obs_parts)

    def _build_history_layout_observation(self, layout_name: str | None = None, history_length: int | None = None):
        effective_history_length = self.observation_cfg.history_length if history_length is None else history_length
        if effective_history_length <= 0:
            raise ValueError("History length must be greater than zero.")

        layout_terms = self.observation_cfg.get_layout_terms(layout_name)
        history_parts = []
        for obs_name in layout_terms:
            obs_item = self.observations_by_name[obs_name]
            current_value = self.obs_map[obs_name]
            if obs_item.use_history:
                history_tensor = self._get_history_tensor(obs_name, effective_history_length)
            else:
                history_tensor = current_value.unsqueeze(0).repeat(effective_history_length, 1)
            history_parts.append(history_tensor)
        return torch.cat(history_parts, dim=-1)
    
    def get_observation(self, layout_name: str | None = None, use_history: bool = False, history_length: int | None = None):
        """Construct observation by calling functions from observation config
        
        Returns:
            Tensor containing concatenated observation vectors
        """
        self.update_observations()

        if use_history:
            return self._build_history_layout_observation(layout_name, history_length)
        return self._build_layout_observation(layout_name)
    
    def get_observation_dim(self, layout_name: str | None = None):
        """Get the total dimension of the observation vector"""
        return self.observation_cfg.get_obs_dim(layout_name)

    def get_observation_slices(self, layout_name: str | None = None):
        layout_key = self.observation_cfg.default_layout if layout_name is None else layout_name
        if layout_key not in self.layout_slices:
            raise ValueError(f"Observation layout '{layout_key}' is not defined.")
        return self.layout_slices[layout_key]
    
    def get_obs_map(self):
        """Get the mapping of observation names to their tensors"""
        return self.obs_map