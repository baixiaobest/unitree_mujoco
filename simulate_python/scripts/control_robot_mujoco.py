from env.mujoco_environment import MujocoEnvironment
from comm.robot_communication import RobotCommunication

if __name__ == "__main__":
    robot_comm = RobotCommunication(domain_id=1, interface="wlo1", device="cuda")
    env = MujocoEnvironment(robot_comm=robot_comm, device="cuda")
    env.run()