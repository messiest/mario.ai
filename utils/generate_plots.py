import os
import sys
import argparse
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

from parsers import parse_loss_logs, parse_result_logs


parser = argparse.ArgumentParser('Mario.ai Plotting')
parser.add_argument('--env-name', type=str, default='SuperMarioBrosNoFrameskip-v0', help='environment name to generate plots for')
parser.add_argument('--model-id', type=str, default='murder_log')
parser.add_argument('--log-dir', type=str, default='logs/')
args = parser.parse_args()


def plot_loss(args):
    print(f"Plotting {args.model_id}'s loss...")
    log_dir = os.path.join(args.log_dir, args.env_name)
    assert os.path.exists(log_dir), 'File not found'

    data = parse_loss_logs(args.model_id, args.env_name, args.log_dir)

    for session in data:
        # saving plots
        save_dir = os.path.join(log_dir, args.model_id, session, 'plots')
        os.makedirs(save_dir, exist_ok=True)

        df = pd.DataFrame().from_dict(data[session])

        plt.figure(figsize=(20, 12), dpi=256)
        for rank in df['rank'].unique():
            df_rank = df[df['rank'] == rank].copy()

            df_rank['log_time'] = pd.to_datetime(df_rank['log_time'])
            idx = pd.date_range(df_rank['log_time'].min(), df_rank['log_time'].max(), freq='T')
            df_rank = df_rank.set_index('log_time')
            df_rank = df_rank.reindex(idx, method='pad')

            df_rank.index = df_rank.index - df_rank.index[0]

            plt.plot(
                df_rank.index / pd.Timedelta(hours=1),
                df_rank['loss'].rolling(60).mean(),
                label=f"Process: {rank}",
            )


        plt.plot([], [], ' ', label=f"Session ID: {session}")

        plt.suptitle(args.env_name, fontsize=18, y=.95)
        plt.title(args.model_id, fontsize=14)
        plt.ylabel('Loss')
        plt.xlabel('Elapsed Time\n(hours)')

        plt.legend();

        plt.savefig(os.path.join(save_dir, 'loss.png'))


def plot_distance(args):
    print(f"Plotting {args.model_id}'s distance...")

    log_dir = os.path.join(args.log_dir, args.env_name)
    assert os.path.exists(log_dir), 'File not found'

    data = parse_result_logs(args.model_id, args.env_name, args.log_dir)

    for session in data:
        # saving plots
        save_dir = os.path.join(log_dir, args.model_id, session, 'plots')
        os.makedirs(save_dir, exist_ok=True)

        df = pd.DataFrame().from_dict(data[session])

        df['log_time'] = pd.to_datetime(df['log_time'])
        idx = pd.date_range(df['log_time'].min(), df['log_time'].max(), freq='T')

        df = df.set_index('log_time')
        df = df.reindex(idx, method='pad')

        df.index = df.index - df.index[0]

        plt.figure(figsize=(20, 12), dpi=256)

        plt.plot(
            df.index / pd.Timedelta(hours=1),
            df['x_position'].rolling(60).mean(),
            label=f"Session ID: {session}",
        )

        plt.suptitle(args.env_name, fontsize=18, y=.95)
        plt.title(args.model_id, fontsize=14, x=.5)
        plt.ylabel('Distance')
        plt.xlabel('Elapsed Time\n(hours)')
        plt.legend()

        plt.savefig(os.path.join(save_dir, 'distance.png'))


def plot_reward(args):
    print(f"Plotting {args.model_id}'s rewards...")

    log_dir = os.path.join(args.log_dir, args.env_name)
    assert os.path.exists(log_dir), 'File not found'

    data = parse_result_logs(args.model_id, args.env_name, args.log_dir)

    for session in data:
        # saving plots
        save_dir = os.path.join(log_dir, args.model_id, session, 'plots')
        os.makedirs(save_dir, exist_ok=True)

        df = pd.DataFrame().from_dict(data[session])

        df['log_time'] = pd.to_datetime(df['log_time'])
        idx = pd.date_range(df['log_time'].min(), df['log_time'].max(), freq='T')

        df = df.set_index('log_time')
        df = df.reindex(idx, method='pad')

        df.index = df.index - df.index[0]

        plt.figure(figsize=(20, 12), dpi=256)

        plt.plot(
            df.index / pd.Timedelta(hours=1),
            df['reward'].rolling(60).mean(),
            label=f"Session ID: {session}",
        )

        plt.suptitle(args.env_name, fontsize=18, y=.95)
        plt.title(args.model_id, fontsize=14, x=.5)
        plt.ylabel('Reward')
        plt.xlabel('Elapsed Time\n(hours)')
        plt.legend()

        plt.savefig(os.path.join(save_dir, 'reward.png'))


def plot_environment_rewards(args):
    def _combine_data(data, master):
        assert isinstance(data, dict)
        for session, session_data in data.items():
            for k, v in session_data.items():
                master[k] += v
        master['session'] += [session] * len(v)
        return master

    log_dir = os.path.join(args.log_dir, args.env_name)
    assert os.path.exists(log_dir), 'File not found'

    # saving plots
    save_dir = os.path.join('assets', args.env_name, 'plots')
    os.makedirs(save_dir, exist_ok=True)

    models = [m for m in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, m))]

    data_store = defaultdict(list)

    for i, model in enumerate(models):
        print(f"{i+1}/{len(models)} | {model}")
        args.model_id = model
        plot_reward(args)
        plot_loss(args)
        plot_distance(args)
        data = parse_result_logs(args.model_id, args.env_name, args.log_dir)

        data_store = _combine_data(data, data_store)

    data_store = dict(data_store)

    _ = data_store.pop('greedy-eps', None)
    _ = data_store.pop('episode', None)

    df_master = pd.DataFrame().from_dict(dict(data_store))
    print(df_master.head(10))
    print(df_master.columns)
    plt.figure(figsize=(20, 12), dpi=256)
    for i, session in enumerate(df_master['session'].unique()):
        df = df_master[df_master['session'] == session].copy()

        df['log_time'] = pd.to_datetime(df['log_time'])
        idx = pd.date_range(df['log_time'].min(), df['log_time'].max(), freq='T')
        df = df.set_index('log_time')
        df = df.reindex(idx, method='pad')

        df.index = df.index - df.index[0]

        plt.plot(
            df.index / pd.Timedelta(hours=1),
            df['reward'].rolling(60).mean(),
            label=f"{i} | {df['id'].iloc[0]}",
        )

        del df

    plt.title(args.env_name, fontsize=18)
    plt.ylabel('Reward')
    plt.xlabel('Elapsed Time\n(hours)')
    plt.legend();

    plt.savefig(os.path.join(save_dir, f'{args.env_name}.reward.png'))


if __name__ == "__main__":
    # _ = plot_loss(args)
    # _ = plot_reward(args)
    _ = plot_environment_rewards(args)
