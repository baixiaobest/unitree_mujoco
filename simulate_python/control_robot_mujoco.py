import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading
import numpy as np

from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand
from robot_communication import RobotCommunication

# Import the message types based on robot type
import config


def initialize_simulation():
    """Initialize MuJoCo model and data objects."""
    mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
    mj_data = mujoco.MjData(mj_model)
    
    # Set simulation parameters
    mj_model.opt.timestep = config.SIMULATE_DT
    
    return mj_model, mj_data

def simulation_step(mj_model, mj_data, elastic_band, band_attached_link):
    """Execute one simulation step with elastic band if enabled."""
    if config.ENABLE_ELASTIC_BAND and elastic_band and elastic_band.enable:
        mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
            mj_data.qpos[:3], mj_data.qvel[:3]
        )
    
    mujoco.mj_step(mj_model, mj_data)

def initialize_robot_bridge(mj_model, mj_data):
    """Initialize the robot bridge for simulation control."""
    # Initialize bridge for simulation control
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)
    
    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()
    
    return unitree

def setup_viewer(mj_model, mj_data):
    """Setup MuJoCo viewer and elastic band if enabled."""
    elastic_band = None
    band_attached_link = None
    
    if config.ENABLE_ELASTIC_BAND:
        elastic_band = ElasticBand()
        if config.ROBOT == "h1" or config.ROBOT == "g1":
            band_attached_link = mj_model.body("torso_link").id
        else:
            band_attached_link = mj_model.body("base_link").id
        viewer = mujoco.viewer.launch_passive(
            mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
        )
    else:
        viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
    
    return viewer, elastic_band, band_attached_link

def SimulationThread():
    """Main simulation thread that handles physics and robot control."""
    global mj_data, mj_model, elastic_band, band_attached_link, viewer, robot_comm
    
    # Initialize the robot bridge to publish simulation data to DDS
    unitree = initialize_robot_bridge(mj_model, mj_data)
    num_joints = mj_model.nu
    
    # Example desired positions for a standing pose
    desired_positions = [0.0] * num_joints
    if num_joints >= 12:  # Assuming it's a quadruped with 12 joints
        desired_positions = [
            0.0, 0.8, -1.6,  # Front Right leg
            0.0, 0.8, -1.6,  # Front Left leg
            0.0, 0.8, -1.6,  # Rear Right leg
            0.0, 0.8, -1.6   # Rear Left leg
        ]

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()
        
        # Get current robot state from the communication class
        joint_state = robot_comm.get_joint_state()
        base_state = robot_comm.get_base_state()
        
        # Periodically print robot state
        if int(mj_data.time * 100) % 100 == 0:
            print("\nCurrent time:", mj_data.time)
            if len(joint_state["positions"]) > 0:
                print("Current joint positions:", joint_state["positions"].cpu().numpy())
                print("Current joint velocities:", joint_state["velocities"].cpu().numpy())
            print("Base position:", base_state["position"].cpu().numpy())
            print("Base orientation (quaternion):", base_state["quaternion"].cpu().numpy())
        
        # Send commands to robot through the communication class
        robot_comm.send_position_commands(desired_positions, num_joints)
        
        # Execute simulation step
        simulation_step(mj_model, mj_data, elastic_band, band_attached_link)

        locker.release()

        # Maintain simulation timing
        time_until_next_step = mj_model.opt.timestep - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

def PhysicsViewerThread():
    """Thread that keeps the viewer synchronized with simulation."""
    global viewer
    
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)

if __name__ == "__main__":
    locker = threading.Lock()
    
    # Initialize simulation
    mj_model, mj_data = initialize_simulation()
    
    # Setup viewer and elastic band
    viewer, elastic_band, band_attached_link = setup_viewer(mj_model, mj_data)
    
    # Initialize robot communication
    robot_comm = RobotCommunication()
    
    # Wait for the viewer to initialize
    time.sleep(0.2)
    
    # Start simulation threads
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()