import os
import csv
import time
import requests

import numpy as np


if not os.path.exists('assets/roster.csv'):
    print("Building Roster")
    build_roster(n=128, save_file=roster_file)
ROSTER = np.genfromtxt('assets/roster.csv', delimiter='\n', dtype=str)


def fetch_name(roster_file='assets/roster.csv'):
    global ROSTER
    sample = np.random.choice(ROSTER, 1, replace=False)
    return sample[0].strip()

def build_roster(n=128, save_file='assets/roster.csv'):
    def random_name():
        URL = r"https://frightanic.com/goodies_content/docker-names.php"
        r = requests.get(URL)
        time.sleep(0.1)
        return r.text

    if not os.path.exists(save_file):
        headers = ['ID']
        with open(save_file, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    for i in range(n):
        with open(save_file, 'a', newline='') as file:
            writer = csv.writer(file)
            agent_id = random_name().strip()
            print(f"{i+1:3d} | {agent_id}")
            writer.writerow([agent_id])

    return True
