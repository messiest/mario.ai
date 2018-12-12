import os
from ast import literal_eval
from pprint import pprint
from collections import defaultdict

import pandas as pd


TEST_FILE = os.path.join('..', 'logs', 'SuperMarioBrosNoFrameskip-1-1-v0', 'sleepy_goomba.info.log')
TEST_FILE_2 = os.path.join('..', 'records', 'SuperMarioBrosNoFrameskip-1-1-v0', 'results.csv')

def log_parser(log_file=TEST_FILE):
    store = defaultdict(list)
    with open(log_file, 'r') as file:
        for line in file.readlines():
            data_idx = line.index('{')
            time = pd.to_datetime(line[:data_idx - 2])
            data = literal_eval(line[data_idx:])
            store['log_time'] += [time]
            for k, v in data.items():
                store[k] += [v]

    df = pd.DataFrame().from_dict(store)
    df['log_time'] = pd.to_datetime(df['log_time'])

    return df


def csv_parser(csv_file=TEST_FILE_2):
    df = pd.read_csv(csv_file)

    return df


if __name__ == "__main__":
    _ = csv_parser()
