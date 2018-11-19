import os
import time
import csv
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
from torch.autograd import Variable

from actor_critic import ActorCritic
from mario_actions import ACTIONS
from mario_wrapper import create_mario_env
from shared_adam import SharedAdam
from utils import fetch_name, FontColor


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

    return action

def _print():
    print("{} | ID: {}, Time: {}, Num Steps: {}, FPS: {:.2f}, Reward: {:.2f}, Episode Length: {}, Progress: {: 3.2f}%".format(
            args.env_name,
            args.model_id,
            time.strftime("%Hh %Mm %Ss", time.gmtime(stop_time - start_time)),
            counter.value,
            counter.value / (stop_time - start_time),
            reward_sum,
            episode_length,
            (info['x_pos'] / 3225) * 100,
        ),
        end='\r',
    )

def train(rank, args, shared_model, counter, lock, optimizer=None, select_sample=True):
    torch.manual_seed(args.seed + rank)

    text_color = FontColor.RED if select_sample else FontColor.GREEN
    print(text_color + f"Process No: {rank: 3d} | Sampling: {select_sample}", FontColor.END)

    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    DoubleTensor = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor
    ByteTensor = torch.cuda.ByteTensor if torch.cuda.is_available() else torch.ByteTensor

    env = create_mario_env(args.env_name)

    # env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], len(ACTIONS))

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
        if rank == 0:
            if t % args.save_interval == 0 and t > 0:
                for file in filter(os.listdir('checkpoints/'), f"{args.env_name}_{args.model_id}_a3c_params.tar"):
                    os.remove(os.path.join('checkpoints', file))
                torch.save(
                    dict(
                        env=args.env_name,
                        id=args.model_id,
                        episode=t,
                        step=episode_length,
                        model_state_dict=shared_model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                    ),
                    os.path.join("checkpoints", f"{args.env_name}_{args.model_id}_a3c_params.tar")
                )

        if t % args.save_interval == 0 and t > 0:  # and rank == 1:
            for file in filter(os.listdir('checkpoints/'), f"{args.env_name}_{args.model_id}_a3c_params.tar"):
                os.remove(os.path.join('checkpoints', file))
            torch.save(
                dict(
                    env=args.env_name,
                    id=args.model_id,
                    step=counter.value,
                    model_state_dict=shared_model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                ),
                os.path.join("checkpoints", f"{args.env_name}_{args.model_id}_a3c_params.tar")
            )
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

            # state_inp = Variable(state.unsqueeze(0)).type(FloatTensor)
            # value, logit, (hx, cx) = model((state_inp, (hx, cx)))
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))

            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            if select_sample:
                # action = prob.multinomial(num_samples=1).detach()
                action = torch.randint(0, len(ACTIONS), (1,1))
            else:
                # action = prob.max(-1, keepdim=True)[1].detach()
                # action = prob.multinomial(num_samples=1).detach()
                action = choose_action(model, state, hx, cx)

            log_prob = log_prob.gather(1, action)

            action_out = ACTIONS[action]

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
            state_inp = Variable(state.unsqueeze(0)).type(FloatTensor)
            value, _, _ = model((state_inp, (hx, cx)))
            R = value.detach()

        values.append(Variable(R).type(FloatTensor))
        policy_loss = 0
        value_loss = 0
        R = Variable(R).type(FloatTensor)

        gae = torch.zeros(1, 1).type(FloatTensor)

        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae).type(FloatTensor) - \
                args.entropy_coef * entropies[i]

        total_loss = policy_loss + args.value_loss_coef * value_loss

        optimizer.zero_grad()
        (total_loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)

        optimizer.step()


def test(rank, args, shared_model, counter):

    torch.manual_seed(args.seed + rank)

    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    DoubleTensor = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor
    ByteTensor = torch.cuda.ByteTensor if torch.cuda.is_available() else torch.ByteTensor

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

    start_time = time.time()
    reward_sum = 0
    done = True
    episode_length = 0
    actions = deque(maxlen=4000)
    while True:
        episode_length += 1
        # shared model sync
        if done:
            model.load_state_dict(shared_model.state_dict())
            with torch.no_grad():
                cx = Variable(torch.zeros(1, 512)).type(FloatTensor)
                hx = Variable(torch.zeros(1, 512)).type(FloatTensor)

        else:
            with torch.no_grad():
                cx = Variable(cx.data).type(FloatTensor)
                hx = Variable(hx.data).type(FloatTensor)

        with torch.no_grad():
            state_inp = Variable(state.unsqueeze(0)).type(FloatTensor)

        value, logit, (hx, cx) = model((state_inp, (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(-1, keepdim=True)[1]

        action_out = ACTIONS[action]

        state, reward, done, info = env.step(action.item())

        save_file = os.getcwd() + f'/save/{args.model_id}_performance.csv'

        if not os.path.exists(save_file):
            headers = ['ID', 'Time', 'Steps', 'Total Reward', 'Episode Length', 'coins', 'flag_get', 'life', 'score', 'stage', 'status', 'time', 'world', 'x_pos']
            with open(save_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)

        env.render()
        done = done or episode_length >= args.max_episode_length

        reward_sum += reward

        actions.append(action[0, 0])

        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            stop_time = time.time()
            print("{} | ID: {}, Time: {}, FPS: {:4.2f}, Reward: {:6.2f}, Episode Length: {:4d}, Progress: {:3.2f}%".format(
                    args.env_name,
                    args.model_id,
                    time.strftime("%H:%M:%Ss", time.gmtime(stop_time - start_time)),
                    counter.value / (stop_time - start_time),
                    reward_sum,
                    episode_length,
                    (info['x_pos'] / 3225) * 100,
                ),
                end='\r',
            )

            data = [
                args.model_id,  # ID
                stop_time - start_time,  # Time
                counter.value,  # Episodes?
                reward_sum,
                episode_length,
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

            time.sleep(3.)
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()

        state = torch.from_numpy(state)


if __name__ == "__main__":
    pass
