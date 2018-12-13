import os
import fnmatch
from ast import literal_eval
from pprint import pprint
from collections import defaultdict

import pandas as pd


def _parse_log(log_file):
    data_store = defaultdict(list)
    with open(log_file, 'r') as file:
        for line in file.readlines():
            data_idx = line.index('{')
            time = pd.to_datetime(line[:data_idx - 2])
            data = literal_eval(line[data_idx:])
            data_store['log_time'] += [time]
            for k, v in data.items():
                data_store[k] += [v]

    return dict(data_store)


def _parse_logs(log_type, model_id, env, log_dir):
    env_logs = os.path.join(log_dir, env, model_id)
    assert os.path.exists(env_logs), f"log files not found at {env_logs}"
    data = {}

    for session in os.listdir(env_logs):
        session_dir = os.path.join(env_logs, session)
        if os.path.isdir(session_dir):
            log = fnmatch.filter(os.listdir(session_dir), f'{log_type}.log')[0]
            log_file = os.path.join(session_dir, log)

            data[session] = _parse_log(log_file)

    return data


def parse_loss_logs(model_id, env, log_dir='logs/'):
    data = _parse_logs('loss', model_id, env, log_dir)

    return data


def parse_info_logs(model_id, env, log_dir='logs/'):
    data = _parse_logs('info', model_id, env, log_dir)

    return data


def parse_result_logs(model_id, env, log_dir='logs/'):
    data = _parse_logs('results', model_id, env, log_dir)

    return data


def parse_action_logs(model_id, env, log_dir='logs/'):
    data = _parse_logs('actions', model_id, env, log_dir)

    return data


if __name__ == "__main__":
    data = parse_result_logs('danger_noodle', 'SuperMarioBrosNoFrameskip-1-1-v0')
    pprint(data)
