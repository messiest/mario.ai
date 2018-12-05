import os
import csv
import time
import random
import logging
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
from utils import FontColor, decode_info

from a3c.utils import ensure_shared_grads, choose_action


def test(rank, args, shared_model, counter, device):
    if not os.path.exists(f'logs/{args.env_name}/'):
        os.mkdir(f'logs/{args.env_name}/')

    logging.basicConfig(
        filename=f'logs/{args.env_name}/{args.model_id}.info.log',
        format='%(asctime)s, %(message)s',
        level=logging.INFO,
    )

    torch.manual_seed(args.seed + rank)

    env = create_mario_env(args.env_name, ACTIONS[args.move_set])
    if args.record:
        env = gym.wrappers.Monitor(env, 'playback/', force=True)

    # env.seed(args.seed + rank)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    model = ActorCritic(observation_space, action_space)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    save_file = os.path.join(args.save_dir, args.env_name, args.save_file)

    if not os.path.exists(save_file):
        os.mkdir(os.path.join(args.save_dir, args.env_name))
        headers = [
            'environment',
            'algorithm',
            'id',
            'time',
            'steps',
            'reward',
            'episode_length',
        ]

        with open(save_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

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
            cx = cx.data
            hx = hx.data

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))

        prob = F.softmax(logit, dim=-1)
        action = prob.max(-1, keepdim=True)[1].cpu().numpy()  #  .data.numpy()

        action_out = ACTIONS[args.move_set][action.item()]

        state, reward, done, info = env.step(action[0, 0])  # action.item()

        reward_sum += reward

        print(
            f"{emojize(':mushroom:')} World {info['world']}-{info['stage']} | {emojize(':video_game:')}: [ {' + '.join(action_out):^13s} ] | ",
            end='\r',
        )

        info_log = {
            'id': args.model_id,
            'action': action_out,
            'reward': reward_sum,
        }
        info_log.update(decode_info(env))  #
        logging.info(info_log)

        env.render()

        actions.append(action[0, 0])

        if done:
            t = time.time() - start_time

            print(
                f"{emojize(':mushroom:')} World {info['world']}-{info['stage']} |" + \
                f" {emojize(':video_game:')}: [ {' + '.join(action_out):^13s} ] | " + \
                f"ID: {args.model_id}, " + \
                f"Time: {time.strftime('%H:%M:%S', time.gmtime(t)):^9s}, " + \
                f"FPS: {counter.value/t: 6.2f}, " + \
                f"Reward: {reward_sum: 10.2f}, " + \
                f"Progress: {(info['x_pos'] / 3225) * 100: 3.2f}%",
                end='\r',
                flush=True,
            )

            data = [
                args.env_name,  # Environment
                args.algorithm,  # Algorithm
                args.model_id,  # ID
                t,  # Time
                counter.value,  # Total Steps
                reward_sum,  # Cummulative Reward
                episode_length,  # Episode Step Length
            ]

            with open(save_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows([data])

            reward_sum = 0
            episode_length = 0
            actions.clear()
            time.sleep(args.reset_delay)
            state = env.reset()

        state = torch.from_numpy(state)


if __name__ == "__main__":
    pass
