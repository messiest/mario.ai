import torch
import torchvision
import gym
import gym_super_mario_bros


def packages():  # TODO: Move this to utils
    print(f"pytorch {torch.__version__}")
    print(f"torchvision {torchvision.__version__}")
    print(f"gym {gym.__version__}")
    # print(f"gym-super-mario-bros {gym_super_mario_bros.__version__}")  # should be 5.0.0
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Cores: {torch.cuda.device_count()}")
