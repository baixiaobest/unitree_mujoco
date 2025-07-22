from mujoco_environment import MujocoEnvironment

if __name__ == "__main__":
    env = MujocoEnvironment(device="cpu")
    env.run()