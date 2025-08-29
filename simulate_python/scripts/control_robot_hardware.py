from env.go2_hardware_environment import GO2HardwareEnvironment
from comm.robot_communication import RobotCommunication

if __name__ == "__main__":
    robot_comm = RobotCommunication(domain_id=0, interface="enp108s0", device="cuda")
    env = GO2HardwareEnvironment(robot_comm=robot_comm, device="cuda", disable_send=True)
    env.run()