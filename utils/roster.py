import os
import csv
import time
import requests

import numpy as np
import pandas as pd


assert os.path.exists('assets/roster.csv'), "Roster file not found."
ids = np.genfromtxt('assets/roster.csv', delimiter='\n', dtype=str)
try:
    existing_ids = np.genfromtxt('save/results.csv', delimiter=',', dtype=str)[:, 2]
except:
    existing_ids = []

ROSTER = [id for id in ids if id not in existing_ids]


def fetch_name(roster_file='assets/roster.csv'):
    global ROSTER
    sample = np.random.choice(ROSTER, 1, replace=False)
    return sample[0].strip()

if __name__ == "__main__":
    print(fetch_name())
