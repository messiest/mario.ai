import os
import sys
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from parsers import log_parser, csv_parser


parser = argparse.ArgumentParser('Mario.ai Plotting')
parser.add_argument('--env-name', type=str, default='SuperMarioBrosNoFrameskip-1-1-v0', help='environment name to generate plots for')
args = parser.parse_args()


sns.set_style('whitegrid')


def main(env):
    d = defaultdict(list)

    log_dir = os.path.join('logs', env)
    assert os.path.exists(log_dir), 'File not found'

    log_files = [f for f in os.listdir(log_dir) if os.path.splitext(f)[1] == '.log']


    for i, log in enumerate(log_files):
        print(f"{i+1}/{len(log_files)}:", log)
        log_path = os.path.join(log_dir, log)
        df = log_parser(log_path)

        d['id'] += df['id'].values.tolist()
        d['reward'] += df['reward'].values.tolist()
        d['log_time'] += (df['log_time'] - df['log_time'][0]).values.tolist()

        if i == 2:
            break

    print("Plotting...")

    df = pd.DataFrame().from_dict(d)
    # df['log_time'] = pd.to_datetime(df['log_time'])
    # idx = pd.date_range(df['log_time'].min(), df['log_time'].max(), freq='T')

    # df = df.groupby(pd.Grouper(freq='60min')).agg({"reward": [np.mean, np.min, np.max]})
    df = df.groupby('log_time').agg({"reward": [np.mean, np.min, np.max]})
    # df = df.set_index('log_time')

    print(df)
    # quit()


    df.columns = list(map('-'.join, df.columns.values))

    # df = df.reindex(idx, method='pad')
    # df.index = df.index - df.index[0]

    plt.figure(figsize=(10, 6), dpi=256)
    plt.plot(
        df.index,
        df['reward-mean'].rolling(int(3.6e5)).mean(),
        label='mean',
    )

    plt.fill_between(df.index, df['reward-amax'].rolling(int(3.6e5)).mean(), df['reward-amin'].rolling(int(3.6e5)).mean(), alpha=0.25)

    plt.title(env)
    plt.ylabel('Reward\n(percent)')
    plt.xlabel('Elapsed Time\n(hours)')
    plt.legend()

    plt.savefig(f'assets/{env}_average_distance.png')


if __name__ == "__main__":
    _ = main(args.env_name)
