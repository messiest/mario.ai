import os
import gc
import csv
import time
import random
from collections import deque
from itertools import count

import gym
import torch
import torch.nn.functional as F
from xvfbwrapper import Xvfb
from emoji import emojize

from models import ActorCritic
from mario_actions import ACTIONS
from mario_wrapper import create_mario_env
from optimizers import SharedAdam
from utils import get_epsilon, FontColor, save_checkpoint

from a3c.utils import ensure_shared_grads, choose_action


def test(rank, args, shared_model, counter, device):
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

    save_file = os.getcwd() + f'/save/{args.env_name}_performance.csv'
    if not os.path.exists(save_file):
        headers = [
            'environment',
            'algorithm',
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
            'time_remaining',
            'world',
            'x_pos',
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
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))

        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].cpu().numpy()

        action_out = ACTIONS[args.move_set][action[0, 0]]

        state, reward, done, info = env.step(action[0, 0])  # action.item()

        print(
            f"{emojize(':mushroom:')} World {info['world']}-{info['stage']} | {emojize(':video_game:')}: [ {' + '.join(action_out):^13s} ]| ",
            end='\r',
        )



        env.render()
        # done = done or episode_length >= args.max_episode_length

        reward_sum += reward

        actions.append(action[0, 0])

        # if actions.count(actions[0]) >= actions.maxlen:
        #     done = True

        if done:
            t = time.time() - start_time

            print(
                f"{emojize(':mushroom:')} World {info['world']}-{info['stage']} |" + \
                f" {emojize(':video_game:')}: [ {' + '.join(action_out):^13s} ]| " + \
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
                info['coins'],  # Coins Collected
                info['flag_get'],  # Got Flag
                info['life'],  # Remaining Lives
                info['score'],  # Score
                info['stage'],  # Stage Number
                info['status'],  # small, super
                info['time'],  # Remaining Time
                info['world'],  # World Number
                info['x_pos'],  # X Distance
            ]

            with open(save_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows([data])

            reward_sum = 0
            episode_length = 0
            actions.clear()
            time.sleep(60.)
            state = env.reset()


        state = torch.from_numpy(state)


if __name__ == "__main__":
    pass
