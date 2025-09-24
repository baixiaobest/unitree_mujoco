from env.environment import Go2Environment
from mdp.observation_manager import ObservationManager, ObservationConfig, ObsItem
from mdp.observations import *
from mdp.command_manager import CommandManager, CommandManagerConfig
from mdp.commands import WasdKeyboardCommand, WasdKeyboardCommandConfig, GameControllerPose2dCommand, GameControllerPose2dCommandConfig
from time import sleep, time
import torch
from utils.robot_logger import RobotLogger  # Import the logger

class GO2HardwareEnvironment(Go2Environment):
    @property
    def last_policy_output(self):
        return self._last_policy_output
    
    def __init__(self, robot_comm, model_path, device="cpu", up_down_test=False, rate=200, kp=25.0, kd=0.5, 
                 log_dir="logs", log_frequency=10, enable_logging=True):
        super().__init__(robot_comm, device, kp=kp, kd=kd)

        self.model_path = model_path
        self.up_down_test = up_down_test
        self.rate = rate
        self.robot_initialized = False
        
        # Initialize high-level services only
        self._init_unitree_services()
        
        # Observation manager and other components
        self._init_observation_manager()
        self._init_command_manager()
        
        self.num_joints = 12
        self.desired_positions = [0.0] * self.num_joints
        
        self.policy = torch.jit.load(self.model_path)
        self._last_policy_output = torch.zeros(self.num_joints, dtype=torch.float32, device=self.device)

        # Initialize logger if enabled
        self.enable_logging = enable_logging
        if enable_logging:
            self.logger = RobotLogger(log_dir=log_dir, log_frequency=log_frequency)
        else:
            self.logger = None

        self.init_time = time()
    
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
                ObsItem("base_linear_velocity", base_lin_vel, 3),
                ObsItem("base_angular_velocity", base_ang_vel, 3),
                ObsItem("projected_gravity", projected_gravity, 3),
                ObsItem("game_controller_pose_2d_command_obs", pose_2d_command, 4, params={"command_name": "game_controller_pose_2d_command"}),
                # ObsItem("wasd_controller_pose_2d_command_obs", pose_2d_command, 4, params={"command_name": "wasd_controller_pose_2d_command"}),
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
                        params={"value": 2 * torch.ones(1, dtype=torch.float32, device=self.device)}),
                ObsItem("obstacle_lidar", constant_observation, 32, 
                        params={"value": 10 * torch.ones(32, dtype=torch.float32, device=self.device)})
            ])
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
                    visualize=True
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

    def hardware_stand_up(self, hold_time=2.0, sim_step_callback=None):
        """Execute stand-up sequence using RobotCommunication
        
        Args:
            hold_time: Time to hold the final standing position
            sim_step_callback: Optional callback function to update simulation after each step
        """
        if self.is_standing:
            print("Robot is already standing")
            return True
        
        print("Starting hardware stand-up sequence...")
        
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
            
            # Call simulation step callback if provided
            if sim_step_callback:
                sim_step_callback()
                
            sleep(0.002)  # Continue at the same control rate

        print("Stand-up sequence complete")
        return True

    def hardware_lay_down(self, sim_step_callback=None):
        """Execute lay-down sequence using RobotCommunication
        
        Args:
            sim_step_callback: Optional callback function to update simulation after each step
        """
        if self.is_laid_down:
            print("Robot is already laid down")
            return True
        
        print("Starting hardware lay-down sequence...")
        
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
            
            # Call simulation step callback if provided
            if sim_step_callback:
                sim_step_callback()
                
            sleep(0.002)  # ~500Hz control rate

        self.is_standing = False
        self.is_laid_down = True
        print("Lay-down sequence complete")
        return True

    def step(self):
        self.elapsed_time = time() - self.init_time

        # Update commands
        self._command_manager.update()
        
        # Get observation and run policy
        obs = self._observation_manager.get_observation()  # Without batch dimension for logging
        obs_batched = obs.unsqueeze(0)  # Add batch dimension for policy
        
        with torch.no_grad():
            policy_action = self.policy(obs_batched).squeeze(0)
        
        # Convert policy output to robot commands
        self.desired_positions = self.joint_map.policy_to_unitree(policy_action, self.joint_scale, self.joints_offset)
        
        # Send commands to the robot
        self._robot_comm.send_position_commands(self.desired_positions, self.num_joints, kp=self.Kp, kd=self.Kd)
        
        # Store policy output for observation
        self._last_policy_output = policy_action.detach()
        
        # Log data if enabled
        if self.enable_logging and self.logger:
            # Get current state
            base_state = self.robot_comm.get_base_state()
            joint_state = self.robot_comm.get_joint_state()
            command = self.command_manager.get_command("game_controller_pose_2d_command")
            
            # Index into observation tensor based on the observation config structure
            # These indices match the order in _init_observation_manager
            base_lin_vel_obs = obs[0:3]                # base_linear_velocity (3)
            base_ang_vel_obs = obs[3:6]                # base_angular_velocity (3)
            projected_gravity_obs = obs[6:9]           # projected_gravity (3)
            command_obs = obs[9:13]                    # game_controller_pose_2d_command_obs (4)
            joint_positions_obs = obs[13:25]           # joint_positions (12)
            joint_velocities_obs = obs[25:37]          # joint_velocities (12)
            last_policy_output_obs = obs[37:49]        # last_policy_output (12)
            count_down_obs = obs[49:50]                # count_down (1)
            obstacle_lidar = obs[50:82]                  # constant_observation (32)
            
            # Log all relevant data
            self.logger.log(
                # Robot state from direct sensors
                base_position=base_state["position"],
                base_quaternion=base_state["quaternion"],
                
                # Command and policy data
                policy_output=policy_action,
                desired_positions=self.desired_positions,
                
                # Actual policy inputs (observation tensor values)
                obs_base_lin_vel=base_lin_vel_obs,
                obs_base_ang_vel=base_ang_vel_obs,
                obs_projected_gravity=projected_gravity_obs,
                obs_command=command_obs,
                obs_joint_positions=joint_positions_obs,
                obs_joint_velocities=joint_velocities_obs,
                obs_last_policy_output=last_policy_output_obs,
                obs_count_down=count_down_obs,
            )
    
    def run(self):
        print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
        input("Press Enter to continue...")
        
        # Wait for robot communication to be ready
        print("Waiting for robot state...")
        joint_state = self._robot_comm.get_joint_state()
        while len(joint_state["positions"]) == 0:
            sleep(0.1)
            joint_state = self._robot_comm.get_joint_state()

        try:
            if self.up_down_test:
                print("Starting up-down test...")
                self.hardware_stand_up(hold_time=5.0)
                self.hardware_lay_down()
                print("Up-down test complete. Exiting.")
                return

            # First, stand up if not already standing
            if not self.robot_initialized:
                self.hardware_stand_up()
                self.robot_initialized = True
                print("Robot is ready for policy control")
            
            # Calculate the desired period in seconds
            period = 1.0 / self.rate
            
            # Then run the main control loop
            print(f"Running control loop at {self.rate} Hz (period: {period*1000:.2f} ms)")
            self.command_manager.setup()
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
            self.hardware_lay_down()
            print("Robot laid down safely")
            
        finally:
            # Always clean up resources
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.enable_logging and self.logger:
            self.logger.close()
        print("Resources cleaned up")

