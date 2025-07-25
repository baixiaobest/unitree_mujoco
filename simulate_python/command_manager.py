from environment import Environment
from dataclasses import dataclass
import torch
from mujoco_visualizer import MujocoVisualizer
from commands import Command, CommandConfig


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
    
    def visulize_commands(self, visualizer: MujocoVisualizer):
        """Visualize commands in the environment"""
        for cmd_name, cmd_instance in self.command_instances:
            if hasattr(cmd_instance, 'visualize'):
                cmd_instance.visualize(visualizer)