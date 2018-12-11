import os
import sys
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from parsers import log_parser, csv_parser


parser = argparse.ArgumentParser('Mario.ai Plotting')
parser.add_argument('--env-name', type=str, default='SuperMarioBrosNoFrameskip-1-1-v0', help='environment name to generate plots for')
args = parser.parse_args()

# ENV_NAME = 'SuperMarioBrosNoFrameskip-1-2-v0'


def main(env):
    log_dir = os.path.join('logs', env)
    assert os.path.exists(log_dir), 'File not found'

    log_files = [f for f in os.listdir(log_dir) if f != '.DS_Store']

    plt.figure(figsize=(20, 12), dpi=256)
    for log in log_files:
        print(log, ': reward')
        log_path = os.path.join(log_dir, log)
        df = log_parser(log_path)

        model_id = df['id'].iloc[0]
        level_complete = df['flag_get'].any()
        episodes = df['done'].sum()

        idx = pd.date_range(df['log_time'].min(), df['log_time'].max(), freq='T')

        df = df.set_index('log_time')
        df = df.reindex(idx, method='pad')
        df.index = df.index - df.index[0]

        plt.plot(
            df.index / pd.Timedelta(hours=1),
            df['reward'].rolling(15).mean(),
            label=model_id,
        )


    plt.title(env)
    plt.ylabel('Reward')
    plt.xlabel('Elapsed Time\n(hours)')
    plt.legend();

    plt.savefig(f'assets/{env}_rewards.png')

    plt.figure(figsize=(20, 12), dpi=256)
    for log in log_files:
        print(log, ': distance')

        log_path = os.path.join(log_dir, log)
        df = log_parser(log_path)

        model_id = df['id'].iloc[0]
        level_complete = df['flag_get'].any()
        episodes = df['done'].sum()

        idx = pd.date_range(df['log_time'].min(), df['log_time'].max(), freq='T')

        df = df.set_index('log_time')
        df = df.reindex(idx, method='pad')
        df.index = df.index - df.index[0]

        plt.plot(
            df.index / pd.Timedelta(hours=1),
            (df['x_position'].rolling(30).mean() / 3225),
            label=f"{model_id} | complete: {level_complete}",
        )

    plt.title(env)
    plt.ylabel('Distance\n(percent)')
    plt.xlabel('Elapsed Time\n(hours)')
    plt.legend();

    plt.savefig(f'assets/{env}_distance.png')


if __name__ == "__main__":
    _ = main(args.env_name)
