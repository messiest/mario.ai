import os
import csv
import time
import requests

import numpy as np


if not os.path.exists('assets/roster.csv'):
    print("Building Roster")
    build_roster(n=128, save_file=roster_file)
ROSTER = np.genfromtxt('assets/roster.csv', delimiter='\n', dtype=str)


class FontColor:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    ALL_COLORS = [PURPLE, CYAN, DARKCYAN, BLUE, GREEN, YELLOW, RED]


def fetch_name(roster_file='assets/roster.csv'):
    global ROSTER
    sample = np.random.choice(roster, 1, replace=False)
    return sample[0]

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

    return



if __name__ == "__main__":
    name = fetch_name()
    print(name)
