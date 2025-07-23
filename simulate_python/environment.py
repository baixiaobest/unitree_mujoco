from robot_communication import RobotCommunication


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
    def time_elapsed(self):
        return self.elapsed_time
    
    @property
    def steps_elapsed(self):
        return self.steps

    def run(self):
        raise NotImplementedError("This method should be implemented by subclasses")

