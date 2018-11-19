import os
import csv
import time
from collections import deque

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from fuzzywuzzy import process

from mario_wrapper import create_mario_env
from actor_critic import ActorCritic
from shared_adam import SharedAdam
from mario_actions import ACTIONS
from utils import fetch_name


def get_checkpoint(env_id, dir='checkpoints/'):
    file, p = process.extractOne(
        env_id + '_a3c_params_*.pkl',
        os.listdir(dir)
    )
    print(file)
    checkpoint = torch.load(os.path.join(dir, file))
    return checkpoint


def test(env_id):
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    DoubleTensor = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor
    ByteTensor = torch.cuda.ByteTensor if torch.cuda.is_available() else torch.ByteTensor

    env = create_mario_env(env_id)
    # env = gym.wrappers.Monitor(env, 'playback/', force=True)

    model = ActorCritic(env.observation_space.shape[0], len(ACTIONS))
    checkpoint = get_checkpoint(env_id)

    print(checkpoint.keys())

    run_id = checkpoint['id']

    model.load_state_dict(checkpoint['model_state_dict'])

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)

    save_file = os.getcwd() + '/save/mario_play_performance.csv'

    if not os.path.exists(save_file):
        print("Generating new record file.")
        title = ['ID', 'Time', 'Steps', 'Total Reward', 'Episode Length']
        with open(save_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(title)

    start_time = time.time()
    reward_sum = 0
    done = True
    actions = deque(maxlen=4000)
    episode_length = 0
    while True:
        episode_length += 1
        ep_start_time = time.time()

        # shared model sync
        if done:
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

        action_out = ACTIONS[action]  #[0, 0]

        state, reward, done, _ = env.step(action.item())
        env.render()

        reward_sum += reward

        actions.append(action[0, 0])

        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            stop_time = time.time()
            print("ID: {}, Time: {}, Num Steps: {}, FPS: {:.2f}, Episode Reward: {}, Episode Length: {}".format(
                run_id,
                time.strftime("%Hh %Mm %Ss", time.gmtime(stop_time - start_time)),
                counter.value,
                counter.value / (stop_time - start_time),
                reward_sum,
                episode_length,
            ))

            data = [
                run_id,
                stop_time - ep_start_time,
                counter.value,
                counter.value / (stop_time - start_time),
                reward_sum,
                episode_length,
            ]

            with open(save_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows([data])

            # for video_path, meta_path in env.videos:
            #     print("VIDEOS:", video_path, meta_path)

            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()

        state = torch.from_numpy(state)



if __name__ == "__main__":
    ENV_ID = 'SuperMarioBros-1-1-v0'
    _ = test(ENV_ID)
