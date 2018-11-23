import math


def get_epsilon(step, eps_end=0.5, eps_start=0.9, eps_decay=200):
    return eps_end + (eps_start - eps_end) * math.exp(-1 * step / eps_decay)
