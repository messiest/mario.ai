import time

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


if __name__ == "__main__":
    for ver in range(1, 4):
        for world in range(1, 9):
            for stage in range(1, 5):
                env = gym_super_mario_bros.make(f'SuperMarioBros-{world}-{stage}-v{ver}')
                env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
                done = True
                for step in range(5):
                    if done:
                        state = env.reset()
                    state, reward, done, info = env.step(env.action_space.sample())
                    env.render()
                time.sleep(1.)
                env.close()
