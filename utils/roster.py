import os
import csv
import time
import requests

import numpy as np
import pandas as pd


def fetch_name(env_name, roster_file='assets/roster.csv'):
    assert os.path.exists('assets/roster.csv'), "roster file not found."
    ids = np.genfromtxt('assets/roster.csv', delimiter='\n', dtype=str)
    try:
        results_file = os.path.join('save', env_name, 'results.csv')
        existing_ids = np.genfromtxt(results_file, delimiter=',', dtype=str)[:, 2]
    except:
        existing_ids = []

    roster = [id for id in ids if id not in existing_ids]

    sample = np.random.choice(roster, 1, replace=False)
    return sample[0].strip()

if __name__ == "__main__":
    print(fetch_name())
