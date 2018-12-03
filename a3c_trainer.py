import os
import time
import argparse
import warnings

import gym
import torch
import torch.multiprocessing as _mp
import torchvision
from xvfbwrapper import Xvfb

from models import ActorCritic
from optimizers import SharedAdam
from mario_wrapper import create_mario_env
from a3c import train, test
from utils import FontColor, fetch_name, debug, restore_checkpoint
from mario_actions import ACTIONS


# Command Line Interface
parser = argparse.ArgumentParser(description='A3C')
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
parser.add_argument('--non-sample', type=int, default=int(_mp.cpu_count() / 2), help='number of non sampling processes (default: 2)')
parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='directory to save checkpoints')
parser.add_argument('--start-step', type=int, default=0, help='training step on which to start')
parser.add_argument('--model-id', type=str, default=fetch_name(), help='name id for the model')
parser.add_argument('--start-fresh', action='store_true', help='start training a new model')
parser.add_argument('--load-model', default=None, type=str, help='model name to restore')
parser.add_argument('--verbose', action='store_true', help='print actions for debugging')
parser.add_argument('--debug', action='store_true', help='print versions of essential packages')
parser.add_argument('--move-set', default='complex', type=str, help='the set of possible actions')
parser.add_argument('--algorithm', default='A3C', type=str, help='algorithm being used')
parser.add_argument('--headless', action='store_true', help='use virtual frame buffer')
parser.add_argument('--reset-delay', type=int, default=60, help='delay between evaluations')

args = parser.parse_args()


# multiprocessing
mp = _mp.get_context('spawn')


def main(args):
    if args.debug:
        debug.packages()
    os.environ['OMP_NUM_THREADS'] = "1"
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        devices = ",".join([str(i) for i in range(torch.cuda.device_count())])
        os.environ["CUDA_VISIBLE_DEVICES"] = devices

    env = create_mario_env(args.env_name, ACTIONS[args.move_set])

    shared_model = ActorCritic(env.observation_space.shape[0], env.action_space.n)

    if torch.cuda.is_available():
        shared_model = shared_model.cuda()

    shared_model.share_memory()

    optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
    optimizer.share_memory()

    if args.load_model:  # TODO Load model before initializing optimizer
        checkpoint_file = f"{args.env_name}_{args.model_id}_a3c_params.tar"
        checkpoint = restore_checkpoint(checkpoint_file)
        assert args.env_name == checkpoint['env'], \
            "Checkpoint is for different environment"
        args.model_id = checkpoint['id']
        args.start_step = checkpoint['step']
        print("Loading model from checkpoint...")
        print(f"Environment: {args.env_name}")
        print(f"      Agent: {args.model_id}")
        print(f"      Moves: {args.move_set}")
        print(f"      Start: Step {args.start_step}")
        shared_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    else:
        print(f"Environment: {args.env_name}")
        print(f"      Agent: {args.model_id}")
        print(f"      Moves: {args.move_set}")

    torch.manual_seed(args.seed)

    print(
        FontColor.BLUE + \
        f"CPUs:    {mp.cpu_count(): 3d} | " + \
        f"GPUs: {None if not torch.cuda.is_available() else torch.cuda.device_count()}" + \
        FontColor.END
    )

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    # Queue test process
    p = mp.Process(
        target=test,
        args=(args.num_processes, args, shared_model, counter, 0)
    )

    p.start()
    processes.append(p)

    # Queue training processes
    num_processes = args.num_processes
    no_sample = args.non_sample  # count of non-sampling processes

    if args.num_processes > 1:
        num_processes = args.num_processes - 1

    samplers = num_processes - no_sample

    for rank in range(0, num_processes):
        device = 'cpu'
        if torch.cuda.is_available():
            device = 0  # TODO: Need to move to distributed to handle multigpu
        if rank < samplers:  # random action
            p = mp.Process(
                target=train,
                args=(rank, args, shared_model, counter, lock, optimizer, device),
            )
        else:  # best action
            p = mp.Process(
                target=train,
                args=(rank, args, shared_model, counter, lock, optimizer, device, False),
            )
        p.start()
        time.sleep(1.)
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    try:
        if not os.getenv('DISPLAY'):
            print('Running Headless')
            vdisplay = Xvfb()
            vdisplay.start()
        _ = main(args)
    except KeyboardInterrupt:
        if not os.getenv('DISPLAY'):
            vdisplay.stop()
    finally:
        print("Done.")
