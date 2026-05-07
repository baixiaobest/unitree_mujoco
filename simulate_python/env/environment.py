from utils.joint_mapping import JointMapping
import torch
import time


class Environment:
    def __init__(self, robot_comm, device="cpu"):
        self.device = device
        # Initialize robot communication
        self._robot_comm = robot_comm
        self.elapsed_time = 0.0
        self.steps = 0
        self._command_manager = None
        self._observation_manager = None

    @property
    def robot_comm(self):
        """Get the robot communication instance"""
        return self._robot_comm
    
    @property
    def command_manager(self):
        """Get the command manager instance"""
        return self._command_manager
    
    @property
    def observation_manager(self):
        """Get the observation manager instance"""
        return self._observation_manager
    
    @property
    def last_policy_output(self):
        raise NotImplementedError("This method should be implemented by subclasses")
    
    @property
    def time_elapsed(self):
        return self.elapsed_time
    
    @property
    def steps_elapsed(self):
        return self.steps

    def run(self):
        raise NotImplementedError("This method should be implemented by subclasses")


class Go2Environment(Environment):
    def __init__(self, robot_comm, device="cpu", kp=60.0, kd=5.0):
        super().__init__(robot_comm, device)
        self.Kp = kp
        self.Kd = kd
        # Initialize specific configurations for Go2 robot
        self.joint_map = self.construct_policy_to_unitree_joint_order_map()
        
        # Initialize stand-up and lay-down sequence parameters
        self.init_robot_motion_sequences()
        
        # Status tracking
        self.is_standing = False
        self.is_laid_down = True

    def init_robot_motion_sequences(self):
        """Initialize parameters for stand-up and lay-down sequences"""
        # Stand-up sequence positions (in unitree joint order)
        self.standup_pos_1 = torch.tensor([
            0.0, 1.36, -2.65,  # FR
            0.0, 1.36, -2.65,  # FL
            -0.2, 1.36, -2.65,  # RR
            0.2, 1.36, -2.65   # RL
        ], device=self.device, dtype=torch.float32)
        
        self.standup_pos_2 = torch.tensor([
            0.0, 0.5, -1.3,  # FR
            0.0, 0.5, -1.3,  # FL
            0.0, 1.0, -1.2,  # RR
            0.0, 1.0, -1.2   # RL
        ], device=self.device, dtype=torch.float32)
        
        self.standup_pos_3 = torch.tensor([
            -0.1, 0.5, -1.3,  # FR
            0.1, 0.5, -1.3,   # FL
            -0.1, 1.0, -1.2,  # RR
            0.1, 1.0, -1.2    # RL
        ], device=self.device, dtype=torch.float32)
        
        # Lay-down sequence positions (in unitree joint order)
        self.laydown_pos_1 = torch.tensor([
            0.0, 0.9, -1.8,  # FR
            0.0, 0.9, -1.8,  # FL
            0.0, 0.9, -1.8,  # RR
            0.0, 0.9, -1.8   # RL
        ], device=self.device, dtype=torch.float32)
        
        self.laydown_pos_2 = torch.tensor([
            0.0, 1.2, -2.0,  # FR
            0.0, 1.2, -2.0,  # FL
            0.0, 1.2, -2.4,  # RR
            0.0, 1.2, -2.4   # RL
        ], device=self.device, dtype=torch.float32)
        
        self.laydown_pos_3 = torch.tensor([
            0.0, 1.6, -2.8,  # FR
            0.0, 1.6, -2.8,  # FL
            0.0, 1.6, -2.8,  # RR
            0.0, 1.6, -2.8   # RL
        ], device=self.device, dtype=torch.float32)
        
        # Durations (in steps)
        self.standup_duration_1 = 500
        self.standup_duration_2 = 500
        self.standup_duration_3 = 500
        
        self.laydown_duration_1 = 500
        self.laydown_duration_2 = 500
        self.laydown_duration_3 = 500

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
        
        self.joints_offset = torch.tensor(
            [ 0.1000, -0.1000, 0.1000, -0.1000, 0.8000, 0.8000, 1.0000, 1.0000, -1.5000, -1.5000, -1.5000, -1.5000], 
            device=self.device, dtype=torch.float32)
        
        self.joint_scale = 0.25
        
        return JointMapping(
            policy_joint_order=policy_joint_order,
            unitree_joint_order=unitree_joint_order,
            device=self.device
        )
        
    def compute_standup_position(self, progress):
        """
        Compute joint position target for a point in the stand-up sequence
        
        Args:
            progress: Float between 0.0 and 1.0 representing progress through the sequence
            
        Returns:
            torch.Tensor: Target joint positions for the given progress point
        """
        # Ensure progress is between 0 and 1
        progress = max(0.0, min(1.0, progress))
        
        # Get current joint positions (only needed at start)
        if progress == 0.0:
            joint_state = self._robot_comm.get_joint_state()
            self.standup_start_pos = joint_state["positions"]
        
        # Determine which phase of the sequence we're in
        total_phases = 3
        phase_duration = 1.0 / total_phases
        
        if progress < phase_duration:  # Phase 1: initial position to first stance
            phase_progress = progress / phase_duration
            return (1 - phase_progress) * self.standup_start_pos + phase_progress * self.standup_pos_1
            
        elif progress < 2 * phase_duration:  # Phase 2: first stance to standing position
            phase_progress = (progress - phase_duration) / phase_duration
            return (1 - phase_progress) * self.standup_pos_1 + phase_progress * self.standup_pos_2
            
        else:  # Phase 3: standing position to ready pose
            phase_progress = (progress - 2 * phase_duration) / phase_duration
            return (1 - phase_progress) * self.standup_pos_2 + phase_progress * self.standup_pos_3

    def compute_laydown_position(self, progress):
        """
        Compute joint position target for a point in the lay-down sequence
        
        Args:
            progress: Float between 0.0 and 1.0 representing progress through the sequence
            
        Returns:
            torch.Tensor: Target joint positions for the given progress point
        """
        # Ensure progress is between 0 and 1
        progress = max(0.0, min(1.0, progress))
        
        # Get current joint positions (only needed at start)
        if progress == 0.0:
            joint_state = self._robot_comm.get_joint_state()
            self.laydown_start_pos = joint_state["positions"]
        
        # Determine which phase of the sequence we're in
        total_phases = 3
        phase_duration = 1.0 / total_phases
        
        if progress < phase_duration:  # Phase 1: initial position to first lay-down stance
            phase_progress = progress / phase_duration
            return (1 - phase_progress) * self.laydown_start_pos + phase_progress * self.laydown_pos_1
            
        elif progress < 2 * phase_duration:  # Phase 2: lower body further
            phase_progress = (progress - phase_duration) / phase_duration
            return (1 - phase_progress) * self.laydown_pos_1 + phase_progress * self.laydown_pos_2
            
        else:  # Phase 3: final lay-down position
            phase_progress = (progress - 2 * phase_duration) / phase_duration
            return (1 - phase_progress) * self.laydown_pos_2 + phase_progress * self.laydown_pos_3

    # Add convenience methods to execute the full sequence
    def stand_up(self, step_fn, kp, kd):
        """
        Execute full stand-up sequence using the provided step function
        
        Args:
            step_fn: Function to execute after sending position commands
        """
        if self.is_standing:
            print("Robot is already standing")
            return True
            
        print("Starting stand-up sequence...")
        
        # Store start position for the sequence
        joint_state = self._robot_comm.get_joint_state()
        self.standup_start_pos = joint_state["positions"]
        
        # Total number of steps in the sequence
        total_steps = self.standup_duration_1 + self.standup_duration_2 + self.standup_duration_3
        
        # Execute each step
        for i in range(total_steps):
            progress = i / (total_steps - 1)
            target_pos = self.compute_standup_position(progress)
            self._robot_comm.send_position_commands(target_pos, len(target_pos), kp=kp, kd=kd)
            step_fn()
        
        self.is_standing = True
        self.is_laid_down = False
        print("Stand-up sequence complete")
        return True

    def lay_down(self, step_fn, kp, kd):
        """
        Execute full lay-down sequence using the provided step function
        
        Args:
            step_fn: Function to execute after sending position commands
        """
        if self.is_laid_down:
            print("Robot is already laid down")
            return True
            
        print("Starting lay-down sequence...")
        
        # Store start position for the sequence
        joint_state = self._robot_comm.get_joint_state()
        self.laydown_start_pos = joint_state["positions"]
        
        # Total number of steps in the sequence
        total_steps = self.laydown_duration_1 + self.laydown_duration_2 + self.laydown_duration_3
        
        # Execute each step
        for i in range(total_steps):
            progress = i / (total_steps - 1)
            target_pos = self.compute_laydown_position(progress)
            self._robot_comm.send_position_commands(target_pos, len(target_pos), kp=kp, kd=kd)
            step_fn()
        
        self.is_standing = False
        self.is_laid_down = True
        print("Lay-down sequence complete")
        return True