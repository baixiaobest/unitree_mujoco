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
        # Observation manager
        E2EObservationConfig = ObservationConfig(
            observations=[
                ObsItem("base_linear_velocity", base_lin_vel, 3),
                ObsItem("base_angular_velocity", base_ang_vel, 3),
                ObsItem("projected_gravity", projected_gravity, 3),
                ObsItem("pose_2d_command", pose_2d_command, 4, params={"command_name": "pose_2d_command"}),
                ObsItem("joint_positions", joint_positions, 12),
                ObsItem("joint_velocities", joint_velocities, 12),
                ObsItem("last_action", last_action, 12)
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
        if self.num_joints >= 12:
            self.desired_positions = [
                0.0, 0.8, -1.6,  # Front Right leg
                0.0, 0.8, -1.6,  # Front Left leg
                0.0, 0.8, -1.6,  # Rear Right leg
                0.0, 0.8, -1.6   # Rear Left leg
            ]

    def initialize_simulation(self):
        mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
        mj_data = mujoco.MjData(mj_model)
        mj_model.opt.timestep = config.SIMULATE_DT
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
            if int(self.mj_data.time * 100) % 100 == 0:
                print("\nCurrent time:", self.mj_data.time)
                if len(joint_state["positions"]) > 0:
                    print("Current joint positions:", joint_state["positions"].cpu().numpy())
                    print("Current joint velocities:", joint_state["velocities"].cpu().numpy())
                print("Base position:", base_state["position"].cpu().numpy())
                print("Base linear velocity:", base_state["velocity"].cpu().numpy())
                print("Base angular velocity (gyroscope):", base_state["gyroscope"].cpu().numpy())
                print("Base orientation (quaternion):", base_state["quaternion"].cpu().numpy())
                obs_tensor = self._observation_manager.get_observation()
                print("Observation tensor:", obs_tensor.cpu().numpy())
                print(f"Elapsed time: {self.elapsed_time:.3f}s, Steps: {self.steps}")

            # Send commands to robot
            self._robot_comm.send_position_commands(self.desired_positions, self.num_joints)

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