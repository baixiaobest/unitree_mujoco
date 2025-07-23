import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading
import numpy as np

from environment import Environment
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand
from observation_manager import ObservationManager, ObservationConfig, ObsItem
from command_manager import CommandManager, CommandManagerConfig, Pose2dCommandConfig, Pose2dCommand
from observations import *
import math

import config

class MujocoEnvironment(Environment):
    def __init__(self, device="cpu"):
        super().__init__(device)

        self.locker = threading.Lock()

        # Initialize simulation
        self.mj_model, self.mj_data = self.initialize_simulation()
        # Setup viewer and elastic band
        self.viewer, self.elastic_band, self.band_attached_link = self.setup_viewer()
        # Initialize robot bridge
        self.unitree_bridge = self.initialize_robot_bridge()

        self.joints_offset = torch.tensor(
            [ 0.1000, -0.1000,  0.1000, -0.1000,  0.8000,  0.8000,  1.0000,  1.0000, -1.5000, -1.5000, -1.5000, -1.5000], 
            device=self.device, dtype=torch.float32)
        self.joint_scale = 0.5

        # Observation manager
        E2EObservationConfig = ObservationConfig(
            observations=[
                ObsItem("base_linear_velocity", base_lin_vel, 3),
                ObsItem("base_angular_velocity", base_ang_vel, 3),
                ObsItem("projected_gravity", projected_gravity, 3),
                ObsItem("pose_2d_command_obs", pose_2d_command, 4, params={"command_name": "pose_2d_command"}),
                # ObsItem("pose_2d_command_obs", pose_2d_zero_command, 4),
                ObsItem("joint_positions", joint_positions, 12, params={"offset": self.joints_offset}),
                ObsItem("joint_velocities", joint_velocities, 12),
                ObsItem("last_action", last_action, 12, params={"offset": self.joints_offset})
                ])
        self._observation_manager = ObservationManager(self, E2EObservationConfig, device=self.device)
        # Command manager
        command_cfg = CommandManagerConfig(
            commands=[
                ("pose_2d_command",
                Pose2dCommand,
                Pose2dCommandConfig(
                    resample_interval=10.0, x_range=(-5, 5), y_range=(-5, 5), z_range=(0.4, 0.4), 
                    angle_range=(-math.pi, math.pi)))]
        )
        self._command_manager = CommandManager(self, command_cfg, device=self.device)

        # Example desired positions for a standing pose
        self.num_joints = self.mj_model.nu
        self.desired_positions = [0.0] * self.num_joints

        self.policy = torch.jit.load("../../../logs/rsl_rl/EncoderActorCriticGO2/E2ENavigation/NoCNN/model_jit.pt")

        self.policy_to_unitree_joint_order_map = self.construct_policy_to_unitree_joint_order_map()

    def initialize_simulation(self):
        mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
        mj_data = mujoco.MjData(mj_model)
        mj_model.opt.timestep = config.SIMULATE_DT
        initial_joint_angles = np.array([
        -0.1, 0.8, -1.5,  # FR_hip, FR_thigh, FR_calf
        0.1, 0.8, -1.5,  # FL_hip, FL_thigh, FL_calf
        -0.1, 0.8, -1.5,  # RR_hip, RR_thigh, RR_calf
        0.1, 0.8, -1.5   # RL_hip, RL_thigh, RL_calf
        ], dtype=np.float32)

        # Make sure the length matches mj_model.nu (number of actuated joints)
        mj_data.qpos[7:7+len(initial_joint_angles)] = initial_joint_angles

        return mj_model, mj_data

    def simulation_step(self):
        if config.ENABLE_ELASTIC_BAND and self.elastic_band and self.elastic_band.enable:
            self.mj_data.xfrc_applied[self.band_attached_link, :3] = self.elastic_band.Advance(
                self.mj_data.qpos[:3], self.mj_data.qvel[:3]
            )
        mujoco.mj_step(self.mj_model, self.mj_data)

    def initialize_robot_bridge(self):
        unitree = UnitreeSdk2Bridge(self.mj_model, self.mj_data)
        if config.USE_JOYSTICK:
            unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
        if config.PRINT_SCENE_INFORMATION:
            unitree.PrintSceneInformation()
        return unitree

    def setup_viewer(self):
        elastic_band = None
        band_attached_link = None
        if config.ENABLE_ELASTIC_BAND:
            elastic_band = ElasticBand()
            if config.ROBOT == "h1" or config.ROBOT == "g1":
                band_attached_link = self.mj_model.body("torso_link").id
            else:
                band_attached_link = self.mj_model.body("base_link").id
            viewer = mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data, key_callback=elastic_band.MujuocoKeyCallback
            )
        else:
            viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        return viewer, elastic_band, band_attached_link
    
    def construct_policy_to_unitree_joint_order_map(self):
        """
        Constructs a mapping from policy joint order to Unitree joint order.
        
        Returns:
            A dictionary mapping policy joint indices to Unitree joint indices.
        """
        policy_joint_order = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 
                            'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 
                            'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']
        
        unitree_joint_order = ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 
                            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 
                            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', 
                            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint']
        
        # Create mapping from policy index to unitree index
        mapping = {}
        for i, joint in enumerate(policy_joint_order):
            unitree_idx = unitree_joint_order.index(joint)
            mapping[i] = unitree_idx
        
        return mapping
    
    def reorder_policy_to_unitree_joint_order(self, policy_action: torch.Tensor, device="cpu"):
        """
        Reorders joint values from policy joint order to Unitree joint order.
        
        Args:
            policy_action: Tensor of shape (12,) in policy joint order
            device: Device to place the resulting tensor on
            
        Returns:
            Tensor of shape (12,) in Unitree joint order
        """
        
        # Create new tensor in unitree order
        unitree_action = torch.zeros_like(policy_action)
        for policy_idx, unitree_idx in self.policy_to_unitree_joint_order_map.items():
            unitree_action[unitree_idx] = policy_action[policy_idx]
        
        return unitree_action

    def simulation_thread(self):
        # Wait for the viewer to initialize
        time.sleep(0.2)
        while self.viewer.is_running():
            step_start = time.perf_counter()
            self.locker.acquire()

            # Get current robot state
            joint_state = self._robot_comm.get_joint_state()
            base_state = self._robot_comm.get_base_state()

            # Print robot state periodically
            if int(self.mj_data.time * 100) % 100 == 0:
                print("\nCurrent time:", self.mj_data.time)
                if len(joint_state["positions"]) > 0:
                    print("Current joint positions:", joint_state["positions"].cpu().numpy())
                    print("Current joint velocities:", joint_state["velocities"].cpu().numpy())
                print("Base position:", base_state["position"].cpu().numpy())
                obs_map = self._observation_manager.get_obs_map()
                for obs in obs_map:
                    print(f"{obs}: {obs_map[obs].cpu().numpy()}")
                print("Joint output: ", self.desired_positions)
                print(f"Elapsed time: {self.elapsed_time:.3f}s, Steps: {self.steps}")

            # Send commands to robot
            obs = self._observation_manager.get_observation()
            obs = obs.unsqueeze(0) # Add batch dimension
            policy_action = self.policy(obs).squeeze(0)  # Remove batch dimension

            policy_action = self.joint_scale * policy_action + self.joints_offset
            self.desired_positions = self.reorder_policy_to_unitree_joint_order(policy_action, device=self.device)

            self._robot_comm.send_position_commands(self.desired_positions, self.num_joints, kp=20.0, kd=1.0)

            # Execute simulation step
            self.simulation_step()
            self.locker.release()

            self._command_manager.update()

            # Track elapsed time and steps
            self.elapsed_time += self.mj_model.opt.timestep
            self.steps += 1

            # Maintain simulation timing
            time_until_next_step = self.mj_model.opt.timestep - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    def viewer_thread(self):
        while self.viewer.is_running():
            self.locker.acquire()
            self.viewer.sync()
            self.locker.release()
            time.sleep(config.VIEWER_DT)

    def run(self):
        viewer_thread = Thread(target=self.viewer_thread)
        sim_thread = Thread(target=self.simulation_thread)
        viewer_thread.start()
        sim_thread.start()
        viewer_thread.join()
        sim_thread.join()