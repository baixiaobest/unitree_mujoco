from env.go2_hardware_environment import GO2HardwareEnvironment
from comm.robot_communication import RobotCommunication
from env.hardware_simulation_environment import HardwareSimulationEnvironment
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
import time

# Configuration
USE_SIMULATION = True  # Set to False to use real hardware
DEVICE = "cuda"
MODEL_PATH = "../../../logs/rsl_rl/EncoderActorCriticGO2/E2ENavigation/MujocoModel/model_2498_jit.ptrom"

# Create a subclass that overrides the _init_unitree_services method for simulation
class SimulationGO2HardwareEnvironment(GO2HardwareEnvironment):
    def _init_unitree_services(self):
        """Skip service initialization in simulation mode"""
        print("Skipping Unitree service initialization (simulation mode)")
        # No need to initialize services that don't exist in simulation
    
    def run(self):
        """Override run method to allow lock-step simulation control"""
        print("Hardware environment ready - control handed to simulation environment")
        # In lock-step mode, we don't run the main loop here
        # The simulation environment will call step() as needed

if __name__ == "__main__":
    if USE_SIMULATION:
        print("Using simulated hardware environment")
        # Initialize DDS communication with domain ID for simulation
        ChannelFactoryInitialize(1, "wlo1")
        
        # Create robot communication that will connect to the simulation
        robot_comm = RobotCommunication(device=DEVICE)
    else:
        print("Connecting to real hardware")
        # Initialize DDS communication with real hardware interface
        ChannelFactoryInitialize(0, "enp108s0")
        
        # Create robot communication that will connect to real hardware
        robot_comm = RobotCommunication(device=DEVICE)
        
    # Create hardware environment
    hw_env = SimulationGO2HardwareEnvironment(
        robot_comm=robot_comm, 
        device=DEVICE,
        model_path=MODEL_PATH,
        kp=25.0, 
        kd=0.5, 
        up_down_test=False)
        
    if USE_SIMULATION:
        # Create simulation environment and set the hardware environment
        sim_env = HardwareSimulationEnvironment()
        sim_env.hardware_env = hw_env  # Use the setter here
        
        # Start lock-step simulation
        try:
            sim_env.start()
            
            # Wait for simulation to finish (Ctrl+C to stop)
            while sim_env.running:
                time.sleep(0.1)
        finally:
            sim_env.stop()
    else:
        hw_env.run()