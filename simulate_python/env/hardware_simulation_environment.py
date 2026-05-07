import time
import mujoco
import mujoco.viewer
import threading
import numpy as np
import torch
from threading import Thread

from robot_comm.unitree_sdk2py_bridge import UnitreeSdk2Bridge
from utils.mujoco_visualizer import MujocoVisualizer
from config import SIMULATION_CONFIG as sim_config

class HardwareSimulationEnvironment:
    """Mujoco simulation environment that controls the hardware environment in lock-step"""
    
    def __init__(self, simulator_update_time=None):
        """
        simulator_update_time: real time between two simulator step function call.
            this is different from step dt used in the simulation, which is simulated time step.
        """

        self.locker = threading.Lock()
        
        # Initialize simulation
        self.mj_model, self.mj_data = self.initialize_simulation()
        
        # Setup viewer
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        
        # Initialize robot bridge
        self.unitree_bridge = self.initialize_robot_bridge()
        
        # Visualization
        self.visualizer = MujocoVisualizer(self.viewer._user_scn)
        
        # Track simulation state
        self.running = False
        self.elapsed_time = 0
        self.steps = 0

        if simulator_update_time is None:
            self.simulator_update_time = self.mj_model.opt.timestep
        else:
            self.simulator_update_time = simulator_update_time
        
        # The hardware environment will be set from the outside
        self._hardware_env = None  # Use a private attribute for encapsulation

    @property
    def hardware_env(self):
        """Getter for the hardware environment"""
        return self._hardware_env

    @hardware_env.setter
    def hardware_env(self, hardware_env):
        """Setter for the hardware environment"""
        print("Setting hardware environment for lock-step simulation")
        self._hardware_env = hardware_env

    def initialize_simulation(self):
        mj_model = mujoco.MjModel.from_xml_path(sim_config["ROBOT_SCENE"])
        mj_data = mujoco.MjData(mj_model)
        mj_model.opt.timestep = sim_config["SIMULATE_DT"]

        # Initialize robot in a laid-down position
        initial_joint_angles = np.array([
            0.0, 1.6, -2.8,  # FR_hip, FR_thigh, FR_calf
            0.0, 1.6, -2.8,  # FL_hip, FL_thigh, FL_calf
            0.0, 1.6, -2.8,  # RR_hip, RR_thigh, RR_calf
            0.0, 1.6, -2.8   # RL_hip, RL_thigh, RL_calf
        ], dtype=np.float32)
        
        height = 0.2
        
        # Set joint angles
        mj_data.qpos[7:7+len(initial_joint_angles)] = initial_joint_angles
        
        # Set base height
        mj_data.qpos[2] = height
        
        return mj_model, mj_data

    def initialize_robot_bridge(self):
        unitree = UnitreeSdk2Bridge(self.mj_model, self.mj_data)
        if sim_config["PRINT_SCENE_INFORMATION"]:
            unitree.PrintSceneInformation()
        return unitree
        
    def simulation_step(self):
        """Execute one lock-step simulation with hardware environment"""
        with self.locker:
            # First, step MuJoCo simulation
            mujoco.mj_step(self.mj_model, self.mj_data)
            
            # If hardware environment is available, step it too
            if self.hardware_env:
                self.hardware_env.step()  # This will read state from MuJoCo via RobotCommunication
    
    def create_step_callback(self):
        """Create a callback function that steps the simulation safely"""
        def callback():
            with self.locker:
                mujoco.mj_step(self.mj_model, self.mj_data)
                # Update the simulation state counter
                self.steps += 1
                self.elapsed_time += self.simulator_update_time
                # Small sleep to maintain real-time factor
                time.sleep(self.mj_model.opt.timestep)
        return callback
        
    def simulation_thread(self):
        """Main simulation thread"""
        # Wait for the viewer to initialize
        time.sleep(0.2)
        
        self.elapsed_time = 0.0
        self.steps = 0
        
        # If hardware environment exists and isn't initialized, do initial setup
        if self.hardware_env and not self.hardware_env.robot_initialized:
            print("Initializing hardware environment...")
            self.hardware_env.robot_initialized = True
            
            # Wait a moment to ensure communication is established
            time.sleep(1.0)
            
            # Create a callback for the stand-up sequence
            step_callback = self.create_step_callback()
            
            # Run the stand-up sequence if needed
            if hasattr(self.hardware_env, 'hardware_stand_up') and not getattr(self.hardware_env, 'is_standing', False):
                print("Executing stand-up sequence with simulation updates...")
                self.hardware_env.hardware_stand_up(hold_time=2.0, sim_step_callback=step_callback)

        self.hardware_env.command_manager.setup()
        print("Starting lock-step simulation...")
        while self.running and self.viewer.is_running():
            step_start = time.perf_counter()
            
            # Execute one simulation step (which also steps the hardware environment)
            self.simulation_step()
                
            # Track elapsed time and steps
            self.elapsed_time += self.mj_model.opt.timestep
            self.steps += 1
            
            # Maintain simulation timing
            time_until_next_step = self.simulator_update_time - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    def debug_visualization(self):
        """Render debug visualizations in the viewer"""
        # If the hardware environment has a command manager, visualize commands
        if self.hardware_env and hasattr(self.hardware_env, '_command_manager'):
            self.hardware_env._command_manager.visualize_commands(self.visualizer)
    
    def viewer_thread(self):
        """Thread for updating the viewer"""
        while self.running and self.viewer.is_running():
            with self.locker:
                self.visualizer.clear_buffer()
                # Add visualization of commands
                self.debug_visualization()
                self.visualizer.render()
                self.viewer.sync()
            time.sleep(sim_config["VIEWER_DT"])
    
    def lay_down_robot(self):
        """Execute lay-down sequence with simulation updates"""
        if self.hardware_env and hasattr(self.hardware_env, 'hardware_lay_down'):
            print("Executing lay-down sequence with simulation updates...")
            step_callback = self.create_step_callback()
            self.hardware_env.hardware_lay_down(sim_step_callback=step_callback)
            return True
        return False
    
    def stand_up_robot(self):
        """Execute stand-up sequence with simulation updates"""
        if self.hardware_env and hasattr(self.hardware_env, 'hardware_stand_up'):
            print("Executing stand-up sequence with simulation updates...")
            step_callback = self.create_step_callback()
            self.hardware_env.hardware_stand_up(hold_time=2.0, sim_step_callback=step_callback)
            return True
        return False
    
    def start(self):
        """Start the simulation threads"""
        if not self.hardware_env:
            print("Warning: No hardware environment set, simulation will run independently")
            
        self.running = True
        self.sim_thread = Thread(target=self.simulation_thread)
        self.view_thread = Thread(target=self.viewer_thread)
        self.sim_thread.start()
        self.view_thread.start()
        print("Lock-step simulation started with MuJoCo viewer")
        
    def stop(self):
        """Stop the simulation threads"""
        self.running = False
        if hasattr(self, 'sim_thread') and self.sim_thread.is_alive():
            self.sim_thread.join()
        if hasattr(self, 'view_thread') and self.view_thread.is_alive():
            self.view_thread.join()
        print("Simulation stopped")