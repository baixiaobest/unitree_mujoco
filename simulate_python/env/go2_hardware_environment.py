from env.environment import Go2Environment
from mdp.observation_manager import ObservationManager, ObservationConfig, ObsItem
from mdp.observations import *
from mdp.command_manager import CommandManager, CommandManagerConfig
from mdp.commands import GameControllerPose2dCommandConfig, GameControllerPose2dCommand, \
    WasdKeyboardCommand, WasdKeyboardCommandConfig
from time import sleep

class GO2HardwareEnvironment(Go2Environment):
    @property
    def last_policy_output(self):
        return self._last_policy_output
    
    def __init__(self, robot_comm, device="cpu", disable_send=False, rate=200):
        super().__init__(robot_comm, device)
        
        self.disable_send = disable_send
        self.rate = rate

        # Observation manager
        E2EObservationConfig = ObservationConfig(
            observations=[
                ObsItem("base_linear_velocity", base_lin_vel, 3),
                ObsItem("base_angular_velocity", base_ang_vel, 3),
                ObsItem("projected_gravity", projected_gravity, 3),
                ObsItem("wasd_controller_pose_2d_command_obs", pose_2d_command, 4, params={"command_name": "wasd_controller_pose_2d_command"}),
                ObsItem("joint_positions", joint_positions, 12, 
                    params={
                        "jointMap": self.joint_map,
                        "scale": 1.0, 
                        "offset": self.joints_offset}),
                ObsItem("joint_velocities", joint_velocities, 12,
                    params={
                        "jointMap": self.joint_map
                    }),
                ObsItem("last_policy_output", last_policy_output, 12),
                ObsItem("count_down", constant_observation, 1, 
                        params={"value": 5 * torch.ones(1, dtype=torch.float32, device=self.device)}),
                ObsItem("constant_observation", constant_observation, 32, 
                        params={"value": 10 * torch.ones(32, dtype=torch.float32, device=self.device)})
                ])
        self._observation_manager = ObservationManager(self, E2EObservationConfig, device=self.device, debug=True)

        # Command manager
        command_cfg = CommandManagerConfig(
            commands=[
                ("wasd_controller_pose_2d_command",
                WasdKeyboardCommand,
                WasdKeyboardCommandConfig(
                    resample_interval=0.05,
                    command_distance=2.0,  # Distance of the command point from robot
                    input_hold_time=0.5,  # Time to hold the command before allowing changes
                    visualize=True
                ))
            ]
        )
        self._command_manager = CommandManager(self, command_cfg, device=self.device)

        self.num_joints = 12

        self.desired_positions = [0.0] * self.num_joints

        self.policy = torch.jit.load("../../../logs/rsl_rl/EncoderActorCriticGO2/E2ENavigation/MujocoModel/model_backward_jit.pt")

        self._last_policy_output = torch.zeros(self.num_joints, dtype=torch.float32, device=self.device)

    def step(self):
        obs = self._observation_manager.get_observation().unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            policy_action = self.policy(obs).squeeze(0)
        
        self.desired_positions = self.joint_map.policy_to_unitree(policy_action, self.joint_scale, self.joints_offset)
        
        if not self.disable_send:
            self._robot_comm.send_position_commands(self.desired_positions, self.num_joints, kp=25.0, kd=1.0)
        
        self._last_policy_output = policy_action.detach()

        self._command_manager.update()

    def run(self):
        while True:
            self.step()
            sleep(1.0 / self.rate)

