from env.environment import Go2Environment
from mdp.observation_manager import ObservationManager, ObservationConfig, ObsItem
from mdp.observations import *
from mdp.command_manager import CommandManager, CommandManagerConfig
from mdp.commands import WasdKeyboardCommand, WasdKeyboardCommandConfig
from time import sleep, time
import torch

class GO2HardwareEnvironment(Go2Environment):
    @property
    def last_policy_output(self):
        return self._last_policy_output
    
    def __init__(self, robot_comm, model_path, device="cpu", rate=200):
        super().__init__(robot_comm, device)

        self.model_path = model_path
        
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
        
        self.Kp = 60.0
        self.Kd = 2.0

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
            self.sc.StandDown()
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
        self._observation_manager = ObservationManager(self, E2EObservationConfig, device=self.device, debug=False)
    
    def _init_command_manager(self):
        """Initialize the command manager"""
        command_cfg = CommandManagerConfig(
            commands=[
                ("wasd_controller_pose_2d_command",
                WasdKeyboardCommand,
                WasdKeyboardCommandConfig(
                    resample_interval=0.05,
                    command_distance=2.0,
                    input_hold_time=0.5,
                    visualize=True
                ))
            ]
        )
        self._command_manager = CommandManager(self, command_cfg, device=self.device)

    def hardware_stand_up(self):
        """Execute stand-up sequence using RobotCommunication"""
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
            
            self._robot_comm.send_position_commands(target_pos, self.num_joints, kp=self.Kp, kd=self.Kd)
            
            sleep(0.002)  # ~500Hz control rate
    
        self.is_standing = True
        self.is_laid_down = False
        print("Stand-up sequence complete")
        return True

    def hardware_lay_down(self):
        """Execute lay-down sequence using RobotCommunication"""
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
            
            self._robot_comm.send_position_commands(target_pos, self.num_joints, kp=self.Kp, kd=self.Kd)
            
            sleep(0.002)  # ~500Hz control rate
    
        self.is_standing = False
        self.is_laid_down = True
        print("Lay-down sequence complete")
        return True

    def step(self):
        self.elapsed_time = time() - self.init_time

        obs = self._observation_manager.get_observation().unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            policy_action = self.policy(obs).squeeze(0)
        
        self.desired_positions = self.joint_map.policy_to_unitree(policy_action, self.joint_scale, self.joints_offset)
        
        self._robot_comm.send_position_commands(self.desired_positions, self.num_joints, kp=self.Kp, kd=self.Kd)
        
        self._last_policy_output = policy_action.detach()
        self._command_manager.update()

    def run(self):
        print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
        input("Press Enter to continue...")
        
        # Wait for robot communication to be ready
        print("Waiting for robot state...")
        joint_state = self._robot_comm.get_joint_state()
        while len(joint_state["positions"]) == 0:
            sleep(0.1)
            joint_state = self._robot_comm.get_joint_state()
        
        # First, stand up if not already standing
        if not self.robot_initialized:
            self.hardware_stand_up()
            self.robot_initialized = True
            print("Robot is ready for policy control")
        
        # Then run the main control loop
        try:
            while True:
                self.step()
                sleep(1.0 / self.rate)
        except KeyboardInterrupt:
            print("Stopping and laying down the robot...")
            self.hardware_lay_down()
            print("Robot laid down safely")
            
    def cleanup(self):
        """Clean up resources"""
        print("Resources cleaned up")

