from robot_communication import RobotCommunication
from joint_mapping import JointMapping
import torch


class Environment:
    def __init__(self, device="cpu"):
        self.device = device
        # Initialize robot communication
        self._robot_comm = RobotCommunication(device)
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
    def __init__(self, device="cpu"):
        super().__init__(device)
        # Initialize specific configurations for Go2 robot
        self.joint_map = self.construct_policy_to_unitree_joint_order_map()


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
            [ 0.1000, -0.1000,  0.1000, -0.1000,  0.8000,  0.8000,  1.0000,  1.0000, -1.5000, -1.5000, -1.5000, -1.5000], 
            device=self.device, dtype=torch.float32)
        
        self.joint_scale = 0.5
        
        return JointMapping(
            policy_joint_order=policy_joint_order,
            unitree_joint_order=unitree_joint_order,
            device=self.device
        )