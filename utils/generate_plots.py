import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

from parsers import log_parser, csv_parser


ENV_NAME = 'SuperMarioBrosNoFrameskip-1-1-v0'


def main(env):
    log_dir = os.path.join('logs', ENV_NAME)
    assert os.path.exists(log_dir), 'File not found'

    log_files = [f for f in os.listdir(log_dir) if f != '.DS_Store']

    plt.figure(figsize=(20, 12), dpi=256)
    for log in log_files:
        log_path = os.path.join(log_dir, log)
        df = log_parser(log_path)

        model_id = df['id'].iloc[0]
        level_complete = df['flag_get'].any()
        episodes = df['done'].sum()

        df = df.set_index('log_time')
        df.index = (df.index - df.index[0]).seconds

        plt.plot(
            (df.index / 3600),
            df['reward'].rolling(60).mean(),
            label=model_id,
        )


    plt.title(env)
    plt.ylabel('Reward')
    plt.xlabel('Elapsed Time\n(hours)')
    plt.legend();

    plt.savefig(f'assets/{env}_rewards.png')

    plt.figure(figsize=(20, 12), dpi=256)
    for log in log_files:
        log_path = os.path.join(log_dir, log)
        df = log_parser(log_path)

        model_id = df['id'].iloc[0]
        level_complete = df['flag_get'].any()
        episodes = df['done'].sum()

        df = df.set_index('log_time')
        df.index = (df.index - df.index[0]).seconds

        plt.plot(
            (df.index / 3600),
            (df['x_position'].rolling(60).mean() / 3225),
            label=f"{model_id} | complete: {level_complete}",
        )


    # plt.ylim(0, 3225)
    plt.title(env)
    plt.ylabel('Distance\n(percent)')
    plt.xlabel('Elapsed Time\n(hours)')
    plt.legend();

    plt.savefig(f'assets/{env}_distance.png')


if __name__ == "__main__":
    _ = main(ENV_NAME)
