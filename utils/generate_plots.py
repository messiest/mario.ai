import os
import sys
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from parsers import parse_loss_logs


parser = argparse.ArgumentParser('Mario.ai Plotting')
parser.add_argument('--env-name', type=str, default='SuperMarioBrosNoFrameskip-1-1-v0', help='environment name to generate plots for')
parser.add_argument('--model-id', type=str, default='big_mess')
parser.add_argument('--log-dir', type=str, default='logs/')
args = parser.parse_args()


def plot_loss(args):
    log_dir = os.path.join(args.log_dir, args.env_name)
    assert os.path.exists(log_dir), 'File not found'

    data = parse_loss_logs(args.model_id, args.env_name, args.log_dir)

    for session in data:
        print(session)

    # plt.figure(figsize=(20, 12), dpi=256)
    for session in data:
        df = pd.DataFrame().from_dict(data[session])

        # df['session'] = session

        # df['log_time'] = pd.to_datetime(df['log_time'])

        # idx = pd.date_range(df['log_time'].min(), df['log_time'].max(), freq='T')


        df = df.set_index('log_time')

        df.index = df.index - df.index[0]



        for rank in df['rank'].unique():
            df_rank = df[df['rank'] == rank]
            # df_rank = df_rank.reindex(idx, method='pad')

            plt.plot(
                df_rank.index / pd.Timedelta(hours=1),
                df_rank['loss'].rolling(15).mean(),
                label=rank,
            )

        plt.title(args.env_name + "\n" + session)
        plt.ylabel('Loss')
        plt.xlabel('Elapsed Time\n(hours)')
        plt.legend();

        plt.show()


        # plt.savefig(f'assets/{env}_rewards.png')


if __name__ == "__main__":
    _ = plot_loss(args)
