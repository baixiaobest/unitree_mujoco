import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

# Import the message types based on robot type
import config
if config.ROBOT=="g1":
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmd_default
else:
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_ as LowCmd_default

from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_


class RobotCommunication:
    """Handles all communication with the robot (state subscription and command publishing)"""
    
    def __init__(self):
        """Initialize communication channels for robot control"""
        # Global state storage
        self.robot_joint_state = {
            "positions": [],
            "velocities": [],
            "torques": []
        }
        
        self.robot_base_state = {
            "position": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "gyroscope": [0.0, 0.0, 0.0],
            "accelerometer": [0.0, 0.0, 0.0]
        }
        
        # Initialize channel factory
        ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
        
        # Initialize subscribers for robot state
        self.low_state_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_subscriber.Init(self._low_state_handler, 10)
        
        self.high_state_subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.high_state_subscriber.Init(self._high_state_handler, 10)
        
        # Initialize command publisher
        self.low_cmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.low_cmd_publisher.Init()
    
    def _low_state_handler(self, msg: LowState_):
        """Handler for low state messages from the robot"""
        # Get joint positions, velocities, and torques
        positions = []
        velocities = []
        torques = []
        
        for motor_state in msg.motor_state:
            positions.append(motor_state.q)
            velocities.append(motor_state.dq)
            torques.append(motor_state.tau_est)
        
        # Update global state
        self.robot_joint_state["positions"] = positions
        self.robot_joint_state["velocities"] = velocities
        self.robot_joint_state["torques"] = torques
        
        # Update IMU data
        self.robot_base_state["quaternion"] = list(msg.imu_state.quaternion)
        self.robot_base_state["gyroscope"] = list(msg.imu_state.gyroscope)
        self.robot_base_state["accelerometer"] = list(msg.imu_state.accelerometer)
    
    def _high_state_handler(self, msg: SportModeState_):
        """Handler for high state messages from the robot"""
        # Update global base state
        self.robot_base_state["position"] = list(msg.position)
        self.robot_base_state["velocity"] = list(msg.velocity)
    
    def get_joint_state(self):
        """Get the current robot joint state from subscribers"""
        return self.robot_joint_state
    
    def get_base_state(self):
        """Get the robot base state from subscribers"""
        return self.robot_base_state
    
    def send_position_commands(self, desired_positions, num_joints, kp=30.0, kd=2.0):
        """Send joint position commands to the robot"""
        try:
            # Create a LowCmd message
            cmd = LowCmd_default()
            
            # Set header and mode
            cmd.head[0] = 0xFE
            cmd.head[1] = 0xEF
            cmd.level_flag = 0xFF
            
            # Set joint commands
            for i in range(min(num_joints, len(cmd.motor_cmd))):
                cmd.motor_cmd[i].mode = 0x0A  # Position control mode
                cmd.motor_cmd[i].q = desired_positions[i]
                cmd.motor_cmd[i].kp = kp
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].kd = kd
                cmd.motor_cmd[i].tau = 0.0
            
            # Publish the command - this will be received by the bridge's LowCmdHandler
            self.low_cmd_publisher.Write(cmd)
            return True
            
        except Exception as e:
            print(f"Failed to send commands: {e}")
            return False

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
            if joint_state["positions"]:
                print("Current joint positions:", joint_state["positions"])
                print("Current joint velocities:", joint_state["velocities"])
            print("Base position:", base_state["position"])
            print("Base orientation (quaternion):", base_state["quaternion"])
        
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