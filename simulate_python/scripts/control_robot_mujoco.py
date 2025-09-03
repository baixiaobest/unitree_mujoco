from env.mujoco_environment import MujocoEnvironment
from comm.robot_communication import RobotCommunication
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

if __name__ == "__main__":
    ChannelFactoryInitialize(1, "wlo1")
    robot_comm = RobotCommunication(device="cuda")
    env = MujocoEnvironment(robot_comm=robot_comm, 
                            model_path="../../../logs/rsl_rl/EncoderActorCriticGO2/E2ENavigation/MujocoModel/model_2498_jit.ptrom", 
                            device="cuda",
                            kp=25.0, kd=0.5)
    env.run()