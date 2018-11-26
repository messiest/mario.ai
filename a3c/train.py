import os
import gc
# import csv
# import time
import random
# from fnmatch import filter
from collections import deque
from itertools import count

import numpy as np
# import cv2
import gym
# from gym import wrappers
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import ActorCritic
from mario_actions import ACTIONS
from mario_wrapper import create_mario_env
from optimizers import SharedAdam
from utils import get_epsilon, FontColor, save_checkpoint

from a3c.utils import ensure_shared_grads, choose_action


def train(rank, args, shared_model, counter, lock, optimizer=None, device='cpu', select_sample=True):
    torch.manual_seed(args.seed + rank)

    text_color = FontColor.RED if select_sample else FontColor.GREEN
    print(text_color + f"Process: {rank: 3d} | Sampling: {str(select_sample):5s} | DEVICE: {device}", FontColor.END)

    env = create_mario_env(args.env_name, ACTIONS[args.move_set])
    # env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    if torch.cuda.is_available():
        model.cuda()

    if torch.cuda.is_available():
        model.cuda()

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    episode_length = 0
    for t in count(start=counter.value):
        if t % args.save_interval == 0 and t > 0:  # and rank == 1:
            save_checkpoint(shared_model, optimizer, args, counter.value)
        # env.render()  # don't render training environments
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1, 512)
            hx = torch.zeros(1, 512)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            episode_length += 1

            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))

            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            reason = ''
            epsilon = get_epsilon(step)
            if select_sample:  # and random.random() < epsilon:
                action = torch.randint(0, env.action_space.n, (1,1)).detach()
                reason = 'random'
            else:
                action = choose_action(model, state, hx, cx)
                model.train()  # may be redundant
                reason = 'choice'

            if torch.cuda.is_available():
                action = action.to(f'{device}')

            log_prob = log_prob.gather(1, action)

            action_out = ACTIONS[args.move_set][action.item()]

            state, reward, done, info = env.step(action.item())
            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 15), -15)  # as per gym-super-mario-bros

            with lock:
                counter.value += 1  # episodes?

            if done:
                episode_length = 0
                state = env.reset()

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))
            R = value.detach()

        if torch.cuda.is_available():
            R.cuda()

        values.append(R)
        policy_loss = 0
        value_loss = 0

        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            # if torch.cuda.is_available():
            #     gae.cuda()

            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * values[i + 1] - values[i]
            # if torch.cuda.is_available():
            #     delta_t.cuda()

            print("gae", type(gae), gae.is_cuda)
            print("args.gamma", type(args.gamma))
            print("args.tau", type(args.tau))
            print("delta_t", type(delta_t), delta_t.is_cuda)
            assert gae.is_cuda == delta_t.is_cuda, "CUDA mismatch!"

            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                          log_probs[i] * gae.detach() - \
                          args.entropy_coef * entropies[i]

        optimizer.zero_grad()
        (policy_loss + args.value_loss_coef * value_loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)

        optimizer.step()

        gc.collect()


if __name__ == "__main__":
    pass
