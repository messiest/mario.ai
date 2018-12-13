import uuid
import argparse

import torch
import torch.multiprocessing as _mp

from utils.roster import fetch_name


def get_args():
    # Command Line Interface
    parser = argparse.ArgumentParser(description='mario.ai')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards (default: 0.9)')
    parser.add_argument('--tau', type=float, default=1.00, help='parameter for GAE (default: 1.00)')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=250, help='value loss coefficient (default: 250)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=_mp.cpu_count(), help='how many training processes to use (default: 4)')
    parser.add_argument('--num-steps', type=int, default=50, help='number of forward steps in A3C (default: 50)')
    parser.add_argument('--max-episode-length', type=int, default=1000000, help='maximum length of an episode (default: 1000000)')
    parser.add_argument('--env-name', default='SuperMarioBrosNoFrameskip-v0', help='environment to train on (default: SuperMarioBrosNoFrameskip-v0)')
    parser.add_argument('--no-shared', default=False, help='use an optimizer without shared momentum.')
    parser.add_argument('--use-cuda', default=True, help='run on gpu.')
    parser.add_argument('--record', action='store_true', help='record playback of tests')
    parser.add_argument('--save-interval', type=int, default=10, help='model save interval (default: 10)')
    parser.add_argument('--non-sample', type=int, default=2, help='number of non sampling processes (default: 2)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='directory to save checkpoints')
    parser.add_argument('--start-step', type=int, default=0, help='training step on which to start')
    parser.add_argument('--model-id', type=str, default="mario", help='name id for the model')
    parser.add_argument('--load-model', action='store_true', help='load existing model')
    parser.add_argument('--start-fresh', action='store_true', help='start training a new model')
    parser.add_argument('--verbose', action='store_true', help='print actions for debugging')
    parser.add_argument('--debug', action='store_true', help='print versions of essential packages')
    parser.add_argument('--move-set', default='complex', type=str, help='the set of possible actions')
    parser.add_argument('--algorithm', default='A3C', type=str, help='algorithm being used')
    parser.add_argument('--headless', action='store_true', help='use virtual frame buffer')
    parser.add_argument('--reset-delay', type=int, default=60, help='delay between evaluations')
    parser.add_argument('--save-dir', type=str, default='records', help='file to save results to')
    parser.add_argument('--save-file', type=str, default='results.csv', help='file to save results to')
    parser.add_argument('--uuid', type=str, default=str(uuid.uuid4()), help='uuid for session')
    parser.add_argument('--greedy-eps', action='store_true', help='perform uniform random action according to greedy-epsilon schedule')


    args = parser.parse_args()

    args.model_id = fetch_name(args.env_name) if not args.model_id else args.model_id

    return args
