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
from utils import FontColor, fetch_name, debug, restore_checkpoint, cli
from mario_actions import ACTIONS


# multiprocessing
mp = _mp.get_context('spawn')

# command line arguments
args = cli.get_args()


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

    # Queue test process
    p = mp.Process(
        target=test,
        args=(args.num_processes, args, shared_model, counter, 0)
    )

    p.start()
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
