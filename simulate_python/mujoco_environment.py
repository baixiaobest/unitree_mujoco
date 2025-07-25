import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading
import numpy as np
import math

from environment import Environment, Go2Environment
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand
from observation_manager import ObservationManager, ObservationConfig, ObsItem
from command_manager import CommandManager, CommandManagerConfig
from commands import Pose2dCommand, Pose2dCommandConfig, GameControllerPose2dCommandConfig, GameControllerPose2dCommand
from observations import *
from mujoco_visualizer import MujocoVisualizer

import config

class MujocoEnvironment(Go2Environment):
    """Mujoco simulation specific environment"""
    @property
    def last_policy_output(self):
        return self._last_policy_output

    def __init__(self, device="cpu"):
        super().__init__(device)

        self.locker = threading.Lock()

        # Initialize simulation
        self.mj_model, self.mj_data = self.initialize_simulation()
        # Setup viewer and elastic band
        self.viewer, self.elastic_band, self.band_attached_link = self.setup_viewer()
        # Initialize robot bridge
        self.unitree_bridge = self.initialize_robot_bridge()
        # Visualization
        self.visualizer = MujocoVisualizer(self.viewer._user_scn)

        self.print_debug = False

        # Observation manager
        E2EObservationConfig = ObservationConfig(
            observations=[
                ObsItem("base_linear_velocity", base_lin_vel, 3),
                ObsItem("base_angular_velocity", base_ang_vel, 3),
                ObsItem("projected_gravity", projected_gravity, 3),
                # ObsItem("pose_2d_command_obs", pose_2d_command, 4, params={"command_name": "pose_2d_command"}),
                # ObsItem("pose_2d_command_obs", pose_2d_zero_command, 4),
                ObsItem("game_controller_pose_2d_command", pose_2d_command, 4, params={"command_name": "game_controller_pose_2d_command"}),
                ObsItem("joint_positions", joint_positions, 12, 
                    params={
                        "jointMap": self.joint_map,
                        "scale": 1.0, 
                        "offset": self.joints_offset}),
                ObsItem("joint_velocities", joint_velocities, 12,
                    params={
                        "jointMap": self.joint_map
                    }),
                ObsItem("last_policy_output", last_policy_output, 12)
                ])
        self._observation_manager = ObservationManager(self, E2EObservationConfig, device=self.device)

        # Command manager
        command_cfg = CommandManagerConfig(
            commands=[
                # ("pose_2d_command",
                # Pose2dCommand,
                # Pose2dCommandConfig(
                #     resample_interval=10.0, x_range=(-5, 5), y_range=(-5, 5), z_range=(0.4, 0.4), 
                #     angle_range=(-math.pi, math.pi), visualize=True
                #     ))
                ("game_controller_pose_2d_command",
                GameControllerPose2dCommand,
                GameControllerPose2dCommandConfig(
                    resample_interval=0.05,
                    max_distance=3.0,  # Maximum distance from robot position
                    controller_index=0,  # Use the first controller
                    joystick_deadzone=0.1,  # Deadzone for joystick input
                    x_axis=1,  # Left stick X axis
                    y_axis=0,  # Left stick Y axis
                    visualize=True
                ))
            ]
        )
        self._command_manager = CommandManager(self, command_cfg, device=self.device)

        # Example desired positions for a standing pose
        self.num_joints = self.mj_model.nu
        self.desired_positions = [0.0] * self.num_joints

        self.policy = torch.jit.load("../../../logs/rsl_rl/EncoderActorCriticGO2/E2ENavigation/TorqueOffset/model_jit.pt")

        self._last_policy_output = torch.zeros(self.num_joints, dtype=torch.float32, device=self.device)

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

        # Set height
        mj_data.qpos[2] = 0.34  # Set base height

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
            # if int(self.mj_data.time * 100) % 100 == 0:
            if self.print_debug:
                if len(joint_state["positions"]) > 0:
                    print("Current joint positions:", joint_state["positions"].cpu().numpy())
                    print("Current joint velocities:", joint_state["velocities"].cpu().numpy())
                print("Base position:", base_state["position"].cpu().numpy())
                print("Joint output: ", self.desired_positions)
                print("--------OBS-------")
                obs_map = self._observation_manager.get_obs_map()
                for obs in obs_map:
                    print(f"{obs}: {obs_map[obs].detach().cpu().numpy()}")
                print("----------------")
                print(f"euler xyz: {self.robot_comm.get_euler_angles().cpu().numpy()}")
                print(f"Elapsed time: {self.elapsed_time:.3f}s, Steps: {self.steps}")

            # Send commands to robot
            obs = self._observation_manager.get_observation()
            obs = obs.unsqueeze(0) # Add batch dimension
            policy_action = self.policy(obs).squeeze(0)  # Remove batch dimension

            self.desired_positions = self.joint_map.policy_to_unitree(policy_action, self.joint_scale, self.joints_offset)

            self._robot_comm.send_position_commands(self.desired_positions, self.num_joints, kp=25.0, kd=1.0)

            self._last_policy_output = policy_action.detach()

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

    def debug_visualization(self):
        """Render debug arrows in the viewer"""
        self.command_manager.visulize_commands(self.visualizer)

    def viewer_thread(self):
        while self.viewer.is_running():
            self.locker.acquire()
            self.visualizer.clear_buffer()
            self.debug_visualization()
            self.visualizer.render()
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