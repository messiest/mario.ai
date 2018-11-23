import os
import gc
import csv
import time
import random
from fnmatch import filter
from collections import deque
from itertools import count

import numpy as np
import cv2
import gym
from gym import wrappers
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import ActorCritic
from mario_actions import ACTIONS
from mario_wrapper import create_mario_env
from shared_adam import SharedAdam
from utils import fetch_name, get_epsilon, FontColor, save_checkpoint


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return

        shared_param._grad = param.grad

def choose_action(model, state, hx, cx):
    model.eval()  # set to eval mode
    _, logits, _ = model.forward((state.unsqueeze(0), (hx, cx)))
    prob = F.softmax(logits, dim=-1).detach()
    action = prob.max(-1, keepdim=True)[1]
    model.train()

    return action


def test(rank, args, shared_model, counter):

    torch.manual_seed(args.seed + rank)

    env = create_mario_env(args.env_name)
    if args.record:
        env = gym.wrappers.Monitor(env, 'playback/', force=True)

    # env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], len(ACTIONS))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True
    episode_length = 0
    actions = deque(maxlen=4000)
    start_time = time.time()
    while True:
        episode_length += 1
        # shared model sync
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 512)
            hx = torch.zeros(1, 512)

        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))

        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()

        action_out = ACTIONS[action[0, 0]]

        print(f"{args.env_name} || {' + '.join(action_out):^13s} || ", end='\r')

        state, reward, done, info = env.step(action[0, 0])  # action.item()

        save_file = os.getcwd() + f'/save/{args.env_name}_performance.csv'

        if not os.path.exists(save_file):
            headers = [
                        'id',
                        'time',
                        'steps',
                        'reward',
                        'episode_length',
                        'coins',
                        'flag_get',
                        'life',
                        'score',
                        'stage',
                        'status',
                        'time',
                        'world',
                        'x_pos',
                ]

            with open(save_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)

        env.render()
        done = done or episode_length >= args.max_episode_length # or info['flag_get']

        reward_sum += reward

        actions.append(action[0, 0])

        if actions.count(actions[0]) >= actions.maxlen:
            done = True

        if done:
            t = time.time() - start_time

            print(
                f"{args.env_name} || " + \
                f"{' + '.join(action_out):^13s} || " + \
                f"ID: {args.model_id}, " + \
                f"Time: {time.strftime('%H:%M:%S', time.gmtime(t)):^10s}, " + \
                f"FPS: {counter.value/t: 6.2f}, " + \
                f"Reward: {reward_sum: 10.2f}, " + \
                f"Episode Length: {episode_length: 7d}, " + \
                f"Progress: {(info['x_pos'] / 3225) * 100: 3.2f}%",
                end='\r',
                flush=True,
            )

            data = [
                args.model_id,  # ID
                t,  # Time
                counter.value,  # Total Steps
                reward_sum,  # Cummulative Reward
                episode_length,  # Episode Step Length
                info['coins'],
                info['flag_get'],
                info['life'],
                info['score'],
                info['stage'],
                info['status'],
                info['time'],
                info['world'],
                info['x_pos'],
            ]

            with open(save_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows([data])

            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            time.sleep(60.)

        state = torch.from_numpy(state)


if __name__ == "__main__":
    pass
