import os
import csv
import time
import uuid
import random
import logging
import argparse
from collections import deque
from itertools import count

import gym
import torch
import torch.nn.functional as F
from emoji import emojize

from models import ActorCritic
from mario_actions import ACTIONS
from mario_wrapper import create_mario_env
from optimizers import SharedAdam
from utils import FontColor, decode_info, setup_logger, get_args, restore_checkpoint

from a3c.utils import ensure_shared_grads, choose_action


args = get_args()

# hack for demo purposes
args.model_id = 'game_n_watch'
args.env_name = 'SuperMarioBros-1-1-v0'
args.reset_delay = 3.


def play(args):
    env = create_mario_env(args.env_name, ACTIONS[args.move_set])

    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    model = ActorCritic(observation_space, action_space)

    checkpoint_file = \
        f"{args.env_name}/{args.model_id}_{args.algorithm}_params.tar"
    checkpoint = restore_checkpoint(checkpoint_file)
    assert args.env_name == checkpoint['env'], \
        "This checkpoint is for different environment: {checkpoint['env']}"
    args.model_id = checkpoint['id']

    print(f"Environment: {args.env_name}")
    print(f"      Agent: {args.model_id}")
    model.load_state_dict(checkpoint['model_state_dict'])

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True
    episode_length = 0
    start_time = time.time()
    for step in count():
        episode_length += 1

        # shared model sync
        if done:
            cx = torch.zeros(1, 512)
            hx = torch.zeros(1, 512)

        else:
            cx = cx.data
            hx = hx.data

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))

        prob = F.softmax(logit, dim=-1)
        action = prob.max(-1, keepdim=True)[1]

        action_idx = action.item()
        action_out = ACTIONS[args.move_set][action_idx]
        state, reward, done, info = env.step(action_idx)
        reward_sum += reward

        print(
            f"{emojize(':mushroom:')} World {info['world']}-{info['stage']} | {emojize(':video_game:')}: [ {' + '.join(action_out):^13s} ] | ",
            end='\r',
        )

        env.render()

        if done:
            t = time.time() - start_time

            print(
                f"{emojize(':mushroom:')} World {info['world']}-{info['stage']} |" + \
                f" {emojize(':video_game:')}: [ {' + '.join(action_out):^13s} ] | " + \
                f"ID: {args.model_id}, " + \
                f"Time: {time.strftime('%H:%M:%S', time.gmtime(t)):^9s}, " + \
                f"Reward: {reward_sum: 10.2f}, " + \
                f"Progress: {(info['x_pos'] / 3225) * 100: 3.2f}%",
                end='\r',
                flush=True,
            )

            reward_sum = 0
            episode_length = 0
            time.sleep(args.reset_delay)
            state = env.reset()

        state = torch.from_numpy(state)


if __name__ == "__main__":
    try:
        _ = play(args)
    except KeyboardInterrupt:
        print()
        pass
