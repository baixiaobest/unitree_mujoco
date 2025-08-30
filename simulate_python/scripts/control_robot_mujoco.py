from env.mujoco_environment import MujocoEnvironment
from comm.robot_communication import RobotCommunication

if __name__ == "__main__":
    robot_comm = RobotCommunication(domain_id=1, interface="wlo1", device="cuda")
    env = MujocoEnvironment(robot_comm=robot_comm, 
                            model_path="../../../logs/rsl_rl/EncoderActorCriticGO2/E2ENavigation/MujocoModel/model_no_goal_penalty_jit.pt", 
                            device="cuda")
    env.run()