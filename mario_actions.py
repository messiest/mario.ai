import numpy as np
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT

ACTIONS = np.array([
    [0, 0, 0, 0, 0, 0],  # null action must come first
    [1, 0, 0, 0, 0, 0],  # up
    [0, 1, 0, 0, 0, 0],  # down
    [0, 0, 1, 0, 0, 0],  # left
    [0, 0, 0, 1, 0, 0],  # right
    [0, 0, 0, 0, 1, 0],  # a (jump)
    # [0, 0, 0, 0, 0, 1],  # b (run)
    [0, 0, 1, 0, 1, 0],  # left + jump
    [0, 0, 1, 0, 0, 1],  # left + run
    [0, 0, 1, 0, 1, 1],  # left + run + jump
    [0, 0, 0, 1, 1, 0],  # right + jump
    [0, 0, 0, 1, 0, 1],  # right + run
    [0, 0, 0, 1, 1, 1],  # right + run + jump
    # [0, 1, 0, 0, 1, 0],  # down + jump
])


ACTIONS = COMPLEX_MOVEMENT
