import os
import csv
import time
import uuid
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
from utils import FontColor, decode_info, setup_logger

from a3c.utils import ensure_shared_grads, choose_action


def test(rank, args, shared_model, counter, device):
    # time.sleep(10.)

    # logging
    log_dir = f'logs/{args.env_name}/{args.model_id}/{args.uuid}/'
    info_logger = setup_logger('info', log_dir, f'info.log')
    result_logger = setup_logger('results', log_dir, f'results.log')

    # torch.manual_seed(args.seed + rank)

    env = create_mario_env(args.env_name, ACTIONS[args.move_set])
    if args.record:
        if not os.path.exists(f'playback/{args.env_name}/'):
            os.makedirs(f'playback/{args.env_name}/{args.model_id}', exist_ok=True)
        env = gym.wrappers.Monitor(env, f'playback/{args.env_name}/{args.model_id}/', force=True)

    # env.seed(args.seed + rank)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    model = ActorCritic(observation_space, action_space)
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
            cx = cx.data
            hx = hx.data

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))

        prob = F.softmax(logit, dim=-1)
        action = prob.max(-1, keepdim=True)[1]
        # action = prob.max(-1, keepdim=True)[1].cpu().numpy()  #  .data.numpy()

        action_out = ACTIONS[args.move_set][action.item()]

        state, reward, done, info = env.step(action.item())

        reward_sum += reward

        info_log = {
            'id': args.model_id,
            'algorithm': args.algorithm,
            'greedy-eps': args.greedy_eps,
            'episode': counter.value,
            'episode_length': episode_length,
            'reward': reward_sum,
            'done': done,
        }
        info_log.update(decode_info(env))
        info_logger.info(info_log)

        print(
            f"{emojize(':mushroom:')} World {info['world']}-{info['stage']} | {emojize(':video_game:')}: [ {' + '.join(action_out):^13s} ] | ",
            end='\r',
        )

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

            result_logger.info(info_log)

            reward_sum = 0
            episode_length = 0
            actions.clear()
            time.sleep(args.reset_delay)
            state = env.reset()

        state = torch.from_numpy(state)


if __name__ == "__main__":
    pass
