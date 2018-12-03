import os
import csv
import time
import requests

import numpy as np


assert os.path.exists('assets/roster.csv'), "Roster file not found."
ROSTER = np.genfromtxt('assets/roster.csv', delimiter='\n', dtype=str)


def fetch_name(roster_file='assets/roster.csv'):
    global ROSTER
    sample = np.random.choice(ROSTER, 1, replace=False)
    return sample[0].strip()
