import math

import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style('whitegrid')


def get_epsilon(step, eps_end=0.01, eps_start=0.1, eps_decay=2000):
    return eps_end + (eps_start - eps_end) * math.exp(-1 * step / eps_decay)

def plot_epsilon_schedule(episodes=1_000_000):
    eps = [get_epsilon(e) for e in range(episodes)]
    plt.figure(figsize=(10, 6), dpi=256)
    plt.plot(eps)

    plt.xlim(0, episodes)
    plt.ylim(0, 1.0)
    plt.title('$\\varepsilon$-greedy Schedule')
    plt.xlabel('Episode')
    plt.ylabel('$\\varepsilon$')
    plt.savefig('assets/epsilon-schedule.png')
    print("Done")


if __name__ == "__main__":
    _ = plot_epsilon_schedule()
