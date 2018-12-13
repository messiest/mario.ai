import os
from collections import deque

import numpy as np
import gym
from gym.spaces.box import Box
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import torch
from torchvision import transforms

from utils import setup_logger


def _process_frame(frame, shape=(84, 84)):
    if frame is not None:
        tsfms = [
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize(shape),
            transforms.ToTensor(),
        ]
        process = transforms.Compose(tsfms)
        frame_t = process(frame)
    else:
        frame_t = torch.zeros((1, 84, 84))

    return frame_t


class ProcessMarioFrame(gym.Wrapper):
    def __init__(self, env=None):
        super(ProcessMarioFrame, self).__init__(env)
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(1, 84, 84),
            dtype=np.uint8,
        )
        self.prev_time = 400
        self.prev_stat = 0
        self.prev_score = 0
        self.prev_dist = 40  # starting position

    def step(self, action):
        obs, _, is_done, info = self.env.step(action)  # custom reward calculated below

        # custom reward
        dist = min(max((info['x_pos'] - self.prev_dist), 0), 2)
        self.prev_dist = info['x_pos'] # + 1

        time = (self.prev_time - info['time']) * -0.1
        self.prev_time = info['time']

        statuses = {
            'small': 1,
            'tall': 2,
            'fireball': 3,
        }
        status = info['status']
        stat = (statuses[status] - self.prev_stat) * 5
        self.prev_stat = statuses[status]

        score = (info['score'] - self.prev_score) * 0.025
        self.prev_score = info['score']

        flag = 0
        if is_done:
            if info['flag_get']:
                flag = 20
            else:
                flag = -20

        reward = dist + time + stat + score + flag

        return _process_frame(obs), reward, is_done, info

    def reset(self):
        self.prev_time = 400
        self.prev_stat = 0
        self.prev_score = 0
        self.prev_dist = 40

        return _process_frame(self.env.reset())


class FrameBuffer(gym.Wrapper):
    def __init__(self, env=None, skip=4, shape=(84, 84)):
        super(FrameBuffer, self).__init__(env)
        self.counter = 0
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
        self.skip = skip
        self.buffer = deque(maxlen=self.skip)

    def step(self, action):
        obs, reward, is_done, info = self.env.step(action)
        counter = 1
        total_reward = reward
        self.buffer.append(obs)

        for i in range(self.skip - 1):
            if not is_done:
                obs, reward, is_done, info = self.env.step(action)
                total_reward += reward
                counter += 1
                self.buffer.append(obs)
            else:
                self.buffer.append(obs)

        frame = np.stack(self.buffer, axis=0)
        frame = np.reshape(frame, (4, 84, 84))

        return frame, total_reward, is_done, info

    def reset(self):
        self.buffer.clear()
        obs = self.env.reset()
        for i in range(self.skip):
            self.buffer.append(obs)

        frame = np.stack(self.buffer, axis=0)
        frame = np.reshape(frame, (4, 84, 84))

        return frame


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, obs):
        if obs is not None:
            self.num_steps += 1
            self.state_mean = self.state_mean * self.alpha + obs.mean() * (1 - self.alpha)
            self.state_std = self.state_std * self.alpha + obs.std() * (1 - self.alpha)

            unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
            unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

            return (obs - unbiased_mean) / (unbiased_std + 1e-8)

        else:
            return obs


def wrap_mario(env):
    env = ProcessMarioFrame(env)
    env = NormalizedEnv(env)
    env = FrameBuffer(env)
    return env


def create_mario_env(env_id, move_set=COMPLEX_MOVEMENT):
    env = gym_super_mario_bros.make(env_id)
    env = BinarySpaceToDiscreteSpaceEnv(env, move_set)
    env = wrap_mario(env)
    return env
