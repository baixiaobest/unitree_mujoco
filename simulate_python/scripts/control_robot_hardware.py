from env.go2_hardware_environment import GO2HardwareEnvironment
from comm.robot_communication import RobotCommunication

if __name__ == "__main__":
    robot_comm = RobotCommunication(domain_id=0, interface="enp108s0", device="cuda")

    env = GO2HardwareEnvironment(
        robot_comm=robot_comm, device="cuda",
        model_path="../../../logs/rsl_rl/EncoderActorCriticGO2/E2ENavigation/MujocoModel/model_1799_jit.ptrom",
        kp=25.0, kd=1.0, up_down_test=False)
    env.run()