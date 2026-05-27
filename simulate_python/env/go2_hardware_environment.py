from env.environment import Go2Environment
from mdp.observation_manager import ObservationManager, ObservationConfig, ObsItem
from mdp.observations import *
from mdp.command_manager import CommandManager, CommandManagerConfig
from mdp.commands import (
    GameControllerPose2dCommand,
    GameControllerPose2dCommandConfig,
    GameControllerVelocityCommand,
    GameControllerVelocityCommandConfig,
    WasdKeyboardCommand,
    WasdKeyboardCommandConfig,
)
from time import sleep, time
import torch
from utils.odometry_publisher import EstimatedOdometryPublisher
from utils.robot_posture import RobotPostureState, TOPIC_ROBOT_POSTURE
from utils.robot_logger import RobotLogger  # Import the logger
from utils.status_monitor_commands import TOPIC_STATUS_MONITOR_COMMAND, decode_status_monitor_command
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__UwbSwitch_ as UwbSwitch_default
from unitree_sdk2py.idl.unitree_go.msg.dds_ import UwbSwitch_, WirelessController_

class GO2HardwareEnvironment(Go2Environment):
    POSITION_CONTROL_POLICY_LAYOUT = "position_control_policy"
    VELOCITY_CONTROL_POLICY_LAYOUT = "velocity_control_policy"
    REMOTE_COMMAND_MAX_LINEAR_SPEED = 0.15
    REMOTE_COMMAND_MAX_ANGULAR_SPEED = 0.35
    ROBOT_POSTURE_PUBLISH_PERIOD_SEC = 0.1

    @property
    def last_policy_output(self):
        return self._last_policy_output
    
    def __init__(self, robot_comm, model_path, device="cpu", up_down_test=False, rate=200, kp=25.0, kd=0.5,
                 log_dir="logs", log_frequency=10, enable_logging=True, policy_mode="position_control",
                 runtime_mode="hardware",
                 jit_history_length=10, debug_print=False):
        super().__init__(robot_comm, device, kp=kp, kd=kd)

        self.model_path = model_path
        self.up_down_test = up_down_test
        self.rate = rate
        self.policy_mode = policy_mode
        self.runtime_mode = runtime_mode
        self.policy_history_length = jit_history_length
        self.debug_print = debug_print
        self.robot_initialized = False
        self._rate_window_start_time = time()
        self._rate_window_step_count = 0
        self._validate_policy_mode()
        self.policy_observation_layout = (
            self.VELOCITY_CONTROL_POLICY_LAYOUT
            if self.policy_mode == "velocity_control"
            else self.POSITION_CONTROL_POLICY_LAYOUT
        )
        self.command_observation_name = (
            "velocity_commands" if self.policy_mode == "velocity_control" else "pose_commands"
        )
        
        # Initialize high-level services only
        self._init_unitree_services()
        
        # Observation manager and other components
        self._init_observation_manager()
        self._init_command_manager()
        
        self.num_joints = 12
        self.desired_positions = [0.0] * self.num_joints
        
        self.policy = torch.jit.load(self.model_path, map_location=self.device)
        self._last_policy_output = torch.zeros(self.num_joints, dtype=torch.float32, device=self.device)
        self._last_estimated_base_lin_vel = None
        self._pending_remote_command = None
        self._posture_motion_step_callback = None
        self._control_session_initialized = False
        self._robot_posture_state = RobotPostureState.LAID_DOWN
        self._last_robot_posture_publish_time = 0.0
        self._estimated_odometry_publisher = (
            EstimatedOdometryPublisher(device=self.device) if self.policy_mode == "velocity_control" else None
        )

        self.robot_posture_publisher: ChannelPublisher = ChannelPublisher(TOPIC_ROBOT_POSTURE, UwbSwitch_)
        self.robot_posture_publisher.Init()
        self._set_robot_posture_state(
            RobotPostureState.STANDING if self.is_standing else RobotPostureState.LAID_DOWN,
            force_publish=True,
        )

        self.status_monitor_command_subscriber: ChannelSubscriber = ChannelSubscriber(
            TOPIC_STATUS_MONITOR_COMMAND, WirelessController_
        )
        self.status_monitor_command_subscriber.Init(self._status_monitor_command_handler, 10)

        # Initialize logger if enabled
        self.enable_logging = enable_logging
        if enable_logging:
            self.logger = RobotLogger(
                log_dir=log_dir,
                log_frequency=log_frequency,
                metadata={
                    "policy_mode": self.policy_mode,
                    "runtime_mode": self.runtime_mode,
                    "policy_observation_layout": self.policy_observation_layout,
                    "command_observation_name": self.command_observation_name,
                },
            )
        else:
            self.logger = None

        self.init_time = time()

    def _status_monitor_command_handler(self, msg: WirelessController_) -> None:
        """Handle stand-up / lay-down commands coming from the status monitor over DDS."""
        command_name = decode_status_monitor_command(msg.keys)
        if command_name is not None:
            self._pending_remote_command = command_name

    def _write_robot_posture_state(self) -> None:
        posture_msg = UwbSwitch_default()
        posture_msg.enabled = int(self._robot_posture_state)
        self.robot_posture_publisher.Write(posture_msg)

    def _set_robot_posture_state(self, state: RobotPostureState, force_publish: bool = False) -> None:
        if state != self._robot_posture_state:
            self._robot_posture_state = state
            force_publish = True
        if force_publish:
            self._write_robot_posture_state()
            self._last_robot_posture_publish_time = time()

    def _maybe_publish_robot_posture_state(self) -> None:
        now = time()
        if now - self._last_robot_posture_publish_time >= self.ROBOT_POSTURE_PUBLISH_PERIOD_SEC:
            self._write_robot_posture_state()
            self._last_robot_posture_publish_time = now

    def set_posture_motion_step_callback(self, callback) -> None:
        """Set an optional simulation-step callback used during stand-up and lay-down sequences."""
        self._posture_motion_step_callback = callback

    def _wait_for_robot_state(self):
        """Block until the joint-state subscriber has received the first valid state."""
        print("Waiting for robot state...")
        joint_state = self._robot_comm.get_joint_state()
        while len(joint_state["positions"]) == 0:
            sleep(0.1)
            joint_state = self._robot_comm.get_joint_state()

    def execute_posture_command(
        self,
        command_name: str,
        require_stationary: bool = False,
        stand_up_hold_time: float | None = None,
    ):
        """Execute a posture command through the shared hardware-environment path."""
        if require_stationary and not self._is_stationary_for_remote_command():
            return False

        if command_name == "stand_up":
            print("Received remote stand-up command from status monitor" if require_stationary else "Starting stand-up command")
            if not self.is_standing:
                if stand_up_hold_time is None:
                    self.hardware_stand_up(sim_step_callback=self._posture_motion_step_callback)
                else:
                    self.hardware_stand_up(
                        hold_time=stand_up_hold_time,
                        sim_step_callback=self._posture_motion_step_callback,
                    )
            self.robot_initialized = True
            if self._estimated_odometry_publisher is not None:
                self._estimated_odometry_publisher.reset_time_reference()
            return True

        if command_name == "lay_down":
            print("Received remote lay-down command from status monitor" if require_stationary else "Starting lay-down command")
            if not self.is_laid_down:
                self.hardware_lay_down(sim_step_callback=self._posture_motion_step_callback)
            self.robot_initialized = False
            if self._estimated_odometry_publisher is not None:
                self._estimated_odometry_publisher.reset_time_reference()
            return True

        raise ValueError(f"Unsupported posture command: {command_name}")

    def initialize_control_session(self):
        """Prepare the environment for policy control using the shared hardware path."""
        if self._control_session_initialized:
            return

        self._wait_for_robot_state()

        if self.command_manager is None:
            raise RuntimeError("Command manager is not initialized.")
        self._command_manager.setup()
        self._control_session_initialized = True
        print("Control session initialized. Use the status monitor to stand up the robot.")

    def _process_pending_remote_command(self):
        """Execute the latest status-monitor command, if any."""
        command_name = self._pending_remote_command
        if command_name is None:
            return

        self._pending_remote_command = None
        self.execute_posture_command(command_name, require_stationary=True)

    def _is_stationary_for_remote_command(self) -> bool:
        """Return whether the robot is stationary enough to safely accept remote posture commands."""
        base_state = self._robot_comm.get_base_state()
        linear_speed = float(torch.linalg.vector_norm(base_state["velocity"]).item())
        angular_speed = float(torch.linalg.vector_norm(base_state["gyroscope"]).item())

        is_stationary = (
            linear_speed <= self.REMOTE_COMMAND_MAX_LINEAR_SPEED
            and angular_speed <= self.REMOTE_COMMAND_MAX_ANGULAR_SPEED
        )
        if not is_stationary:
            print(
                "Ignoring remote posture command because robot is not stationary "
                f"(linear_speed={linear_speed:.3f} m/s, angular_speed={angular_speed:.3f} rad/s)"
            )
        return is_stationary

    def _maybe_print_update_rate(self):
        if not self.debug_print:
            return

        self._rate_window_step_count += 1
        now = time()
        elapsed = now - self._rate_window_start_time
        if elapsed < 1.0:
            return

        actual_rate = self._rate_window_step_count / max(elapsed, 1e-6)
        print(
            f"Actual update rate: {actual_rate:.1f} Hz "
            f"(target: {self.rate:.1f} Hz, samples: {self._rate_window_step_count}, window: {elapsed:.2f} s)"
        )
        self._rate_window_start_time = now
        self._rate_window_step_count = 0

    def _validate_policy_mode(self):
        if self.policy_mode not in {"position_control", "velocity_control"}:
            raise ValueError(
                f"Unsupported policy mode '{self.policy_mode}'. "
                "Expected 'position_control' or 'velocity_control'."
            )
    
    def _init_unitree_services(self):
        """Initialize high-level Unitree SDK services"""
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        
        # Initialize sport client
        self.sc = SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()
        
        # Initialize motion switcher client
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()
        
        # Check and release any active modes
        status, result = self.msc.CheckMode()

        while result['name']:
            print(f"Releasing active mode: {result['name']}")
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            sleep(1)
        
        print("Robot services initialized and ready for low-level control")
    
    def _init_observation_manager(self):
        """Initialize the observation manager"""
        E2EObservationConfig = ObservationConfig(
            observations=[
                ObsItem("base_lin_vel", base_lin_vel, 3),
                ObsItem("base_ang_vel", base_ang_vel, 3),
                ObsItem("imu_ang_vel", imu_ang_vel, 3, use_history=True),
                ObsItem("imu_lin_acc", imu_lin_acc, 3, use_history=True),
                ObsItem("projected_gravity", projected_gravity, 3, use_history=True),
                ObsItem(
                    "pose_commands",
                    pose_2d_command,
                    4,
                    params={"command_name": "game_controller_pose_2d_command"},
                ),
                ObsItem(
                    "velocity_commands",
                    velocity_command,
                    3,
                    params={"command_name": "game_controller_velocity_command"},
                    use_history=True,
                ),
                ObsItem("joint_pos", joint_positions, 12,
                    params={
                        "jointMap": self.joint_map,
                        "scale": 1.0,
                        "offset": self.joints_offset,
                    },
                    use_history=True,
                ),
                ObsItem("joint_vel", joint_velocities, 12,
                    params={
                        "jointMap": self.joint_map
                    },
                    use_history=True,
                ),
                ObsItem("actions", last_policy_output, 12, use_history=True),
                ObsItem(
                    "count_down",
                    constant_observation,
                    1,
                    params={"value": 2 * torch.ones(1, dtype=torch.float32, device=self.device)},
                ),
                ObsItem(
                    "obstacle_lidar",
                    constant_observation,
                    32,
                    params={"value": 10 * torch.ones(32, dtype=torch.float32, device=self.device)},
                ),
            ],
            layouts={
                self.POSITION_CONTROL_POLICY_LAYOUT: [
                    "base_lin_vel",
                    "base_ang_vel",
                    "projected_gravity",
                    "pose_commands",
                    "joint_pos",
                    "joint_vel",
                    "actions",
                    "count_down",
                    "obstacle_lidar",
                ],
                self.VELOCITY_CONTROL_POLICY_LAYOUT: [
                    "actions",
                    "imu_ang_vel",
                    "imu_lin_acc",
                    "joint_pos",
                    "joint_vel",
                    "projected_gravity",
                    "velocity_commands",
                ],
            },
            history_length=self.policy_history_length,
            default_layout=self.POSITION_CONTROL_POLICY_LAYOUT,
        )
        self._observation_manager = ObservationManager(self, E2EObservationConfig, device=self.device, debug=False)
    
    def _init_command_manager(self):
        """Initialize the command manager"""
        command_cfg = CommandManagerConfig(
            commands=[
                ("game_controller_pose_2d_command",
                GameControllerPose2dCommand,
                GameControllerPose2dCommandConfig(
                    resample_interval=0.05,
                    max_distance=1.0,  # Maximum distance from robot position
                    standing_height=0.3,
                    controller_index=0,  # Use the first controller
                    joystick_deadzone=0.1,  # Deadzone for joystick input
                    x_axis=1,  # Left stick X axis
                    y_axis=0,  # Left stick Y axis
                    mode="global",  # "global" or "local"
                    a_button_index=0,  # Button index for 'A' button
                    visualize=self.policy_mode == "position_control"
                )),
                ("game_controller_velocity_command",
                GameControllerVelocityCommand,
                GameControllerVelocityCommandConfig(
                    resample_interval=0.05,
                    max_linear_velocity=1.0,
                    max_angular_velocity=1.0,
                    controller_index=0,
                    joystick_deadzone=0.1,
                    left_x_axis=0,
                    left_y_axis=1,
                    right_x_axis=3,
                    right_y_axis=4,
                    visualize=self.policy_mode == "velocity_control",
                    visualize_height=0.35,
                    visualize_scale=0.5,
                )),

                # ("wasd_controller_pose_2d_command",
                # WasdKeyboardCommand,
                # WasdKeyboardCommandConfig(
                #     resample_interval=0.05,
                #     command_distance=1.0,
                #     command_turn_distance=0.5,
                #     input_hold_time=0.1,
                #     rotate_angle=90,
                #     mode="global",
                #     visualize=True
                # ))
            ]
        )
        self._command_manager = CommandManager(self, command_cfg, device=self.device)

    def _get_policy_observation(self):
        if self.policy_mode == "velocity_control":
            return self._observation_manager.get_observation(
                layout_name=self.policy_observation_layout,
                use_history=True,
                history_length=self.policy_history_length,
            )
        return self._observation_manager.get_observation(layout_name=self.policy_observation_layout)

    def _estimate_base_lin_vel(self, obs_batched: torch.Tensor):
        if self.policy_mode != "velocity_control":
            return None
        if not hasattr(self.policy, "estimate_velocity"):
            return None

        estimated_velocity = self.policy.estimate_velocity(obs_batched).squeeze(0).detach()
        self._last_estimated_base_lin_vel = estimated_velocity
        return estimated_velocity

    def hardware_stand_up(self, hold_time=2.0, sim_step_callback=None):
        """Execute stand-up sequence using RobotCommunication
        
        Args:
            hold_time: Time to hold the final standing position
            sim_step_callback: Optional callback function to update simulation after each step
        """
        if self.is_standing:
            print("Robot is already standing")
            self._set_robot_posture_state(RobotPostureState.STANDING, force_publish=True)
            return True
        
        print("Starting hardware stand-up sequence...")
        self._set_robot_posture_state(RobotPostureState.TRANSITIONING_TO_STAND, force_publish=True)
        
        # Wait for robot communication to be ready
        wait_count = 0
        joint_state = self._robot_comm.get_joint_state()
        while len(joint_state["positions"]) == 0:
            sleep(0.1)
            joint_state = self._robot_comm.get_joint_state()
            wait_count += 1
            if wait_count > 50:  # 5 seconds timeout
                print("Timeout waiting for robot state")
                return False
        
        # Store start position for the sequence
        self.standup_start_pos = joint_state["positions"]
        
        # Total number of steps in the sequence
        total_steps = self.standup_duration_1 + self.standup_duration_2 + self.standup_duration_3

        # Execute each step using the inherited compute_standup_position method
        for i in range(total_steps):
            progress = i / (total_steps - 1)
            target_pos = self.compute_standup_position(progress)
            
            self._robot_comm.send_position_commands(target_pos, self.num_joints, kp=40.0, kd=1.0)
            self._maybe_publish_robot_posture_state()
            
            # Call simulation step callback if provided
            if sim_step_callback:
                sim_step_callback()
            
            sleep(0.002)  # ~500Hz control rate

        # Get the final standing position
        final_stand_position = self.compute_standup_position(1.0)

        self.is_standing = True
        self.is_laid_down = False
        
        # Hold the final position for the specified time
        print(f"Holding final stand position for {hold_time} seconds...")
        hold_start_time = time()
        while time() - hold_start_time < hold_time:
            self._robot_comm.send_position_commands(final_stand_position, self.num_joints, kp=40.0, kd=1.0)
            self._maybe_publish_robot_posture_state()
            
            # Call simulation step callback if provided
            if sim_step_callback:
                sim_step_callback()
                
            sleep(0.002)  # Continue at the same control rate

        self._set_robot_posture_state(RobotPostureState.STANDING, force_publish=True)
        print("Stand-up sequence complete")
        return True

    def hardware_lay_down(self, sim_step_callback=None):
        """Execute lay-down sequence using RobotCommunication
        
        Args:
            sim_step_callback: Optional callback function to update simulation after each step
        """
        if self.is_laid_down:
            print("Robot is already laid down")
            self._set_robot_posture_state(RobotPostureState.LAID_DOWN, force_publish=True)
            return True
        
        print("Starting hardware lay-down sequence...")
        self._set_robot_posture_state(RobotPostureState.TRANSITIONING_TO_LAY, force_publish=True)
        
        # Get current joint positions
        joint_state = self._robot_comm.get_joint_state()
        self.laydown_start_pos = joint_state["positions"]
        
        # Total number of steps in the sequence
        total_steps = self.laydown_duration_1 + self.laydown_duration_2 + self.laydown_duration_3

        # Execute each step using the inherited compute_laydown_position method
        for i in range(total_steps):
            progress = i / (total_steps - 1)
            target_pos = self.compute_laydown_position(progress)
            
            self._robot_comm.send_position_commands(target_pos, self.num_joints, kp=40.0, kd=1.0)
            self._maybe_publish_robot_posture_state()
            
            # Call simulation step callback if provided
            if sim_step_callback:
                sim_step_callback()
                
            sleep(0.002)  # ~500Hz control rate

        self.is_standing = False
        self.is_laid_down = True
        self._set_robot_posture_state(RobotPostureState.LAID_DOWN, force_publish=True)
        print("Lay-down sequence complete")
        return True

    def step(self):
        self._maybe_publish_robot_posture_state()
        self._process_pending_remote_command()
        if not self.robot_initialized:
            return False

        self.elapsed_time = time() - self.init_time
        self.steps += 1

        # Update commands
        self._command_manager.update()
        
        # Get observation and run policy
        policy_obs = self._get_policy_observation()
        obs_batched = policy_obs.unsqueeze(0)
        estimated_base_lin_vel = None
        
        with torch.no_grad():
            estimated_base_lin_vel = self._estimate_base_lin_vel(obs_batched)
            policy_action = self.policy(obs_batched).squeeze(0)
        
        # Convert policy output to robot commands
        self.desired_positions = self.joint_map.policy_to_unitree(policy_action, self.joint_scale, self.joints_offset)
        
        # Send commands to the robot
        self._robot_comm.send_position_commands(self.desired_positions, self.num_joints, kp=self.Kp, kd=self.Kd)
        
        # Store policy output for observation
        self._last_policy_output = policy_action.detach()

        base_state = self.robot_comm.get_base_state()
        if self._estimated_odometry_publisher is not None:
            self._estimated_odometry_publisher.publish(
                estimated_linear_velocity=estimated_base_lin_vel,
                base_quaternion=base_state["quaternion"],
                angular_velocity=base_state["gyroscope"],
            )
        
        # Log data if enabled
        if self.enable_logging and self.logger:
            current_obs = self._observation_manager.get_obs_map()
            
            # Log all relevant data
            self.logger.log(
                # Robot state from direct sensors
                base_position=base_state["position"],
                base_quaternion=base_state["quaternion"],
                
                # Command and policy data
                policy_output=policy_action,
                desired_positions=self.desired_positions,
                
                # Actual policy inputs (observation tensor values)
                obs_base_lin_vel=current_obs["base_lin_vel"],
                estimated_base_lin_vel=estimated_base_lin_vel,
                obs_base_ang_vel=current_obs["base_ang_vel"],
                obs_imu_ang_vel=current_obs["imu_ang_vel"],
                obs_imu_lin_acc=current_obs["imu_lin_acc"],
                obs_projected_gravity=current_obs["projected_gravity"],
                obs_command=current_obs[self.command_observation_name],
                obs_joint_positions=current_obs["joint_pos"],
                obs_joint_velocities=current_obs["joint_vel"],
                obs_last_policy_output=current_obs["actions"],
                obs_count_down=current_obs["count_down"],
            )

            self._maybe_print_update_rate()
        return True
    
    def run(self):
        print("WARNING: Please ensure there are no obstacles around the robot while running this example.")

        try:
            if self.up_down_test:
                print("Starting up-down test...")
                self._wait_for_robot_state()
                self.execute_posture_command("stand_up", stand_up_hold_time=5.0)
                self.execute_posture_command("lay_down")
                print("Up-down test complete. Exiting.")
                return

            self.initialize_control_session()
            
            # Calculate the desired period in seconds
            period = 1.0 / self.rate
            
            # Then run the main control loop
            print(
                f"Running {self.policy_mode} control loop at {self.rate} Hz "
                f"(period: {period*1000:.2f} ms)"
            )
            while True:
                # Start timing this iteration
                iteration_start = time()
                
                # Execute control step
                self.step()
                
                # Calculate how much time has elapsed
                elapsed = time() - iteration_start
                
                # Sleep only for the remaining time to maintain desired rate
                sleep_time = period - elapsed
                if sleep_time > 0:
                    sleep(sleep_time)

        except KeyboardInterrupt:
            print("Stopping and laying down the robot...")
            self.execute_posture_command("lay_down")
            print("Robot laid down safely")
            
        finally:
            # Always clean up resources
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.enable_logging and self.logger:
            self.logger.close()
        print("Resources cleaned up")

