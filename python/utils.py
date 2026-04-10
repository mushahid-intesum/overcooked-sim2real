import torch
import numpy as np

CHANNELS = {
    "wall":     0,
    "floor":    1,
    "self":     2,
    "teammate": 3,
    "onion":    4,
    "tomato":   5,
    "dish":     6,
    "soup":     7,
    "pot":      8,
    "serve":    9,
    "delivery": 10,
}

NUM_CHANNELS = len(CHANNELS)

HELD_NONE   = 0
HELD_ONION  = 1
HELD_TOMATO = 2
HELD_DISH   = 3
HELD_SOUP   = 4
NUM_HELD    = 5

DIRECTION_VEC = {
    0: (-1,  0),  # NORTH
    1: ( 0,  1),  # EAST
    2: ( 1,  0),  # SOUTH
    3: ( 0, -1),  # WEST
}

def compute_gae(rewards, values, dones, gamma, lam):
    T, N = rewards.shape
    adv  = torch.zeros(T, N)
    gae  = torch.zeros(N)
    for t in reversed(range(T)):
        delta  = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae    = delta + gamma * lam * (1 - dones[t]) * gae
        adv[t] = gae
    return adv, adv + values[:T]

def build_cone_mask(fov_radius):
    D = 2 * fov_radius + 1
    mask = np.zeros((D, D), dtype=bool)
    side_reach = max(1, int(fov_radius * 0.6))

    for r in range(D):
        for c in range(D):
            dr = r - fov_radius   
            dc = abs(c - fov_radius)

            if dr < 0:
                if dc <= abs(dr):
                    mask[r, c] = True   
                elif dc <= side_reach:
                    mask[r, c] = True     
            elif dr == 0:
                if dc <= side_reach:
                    mask[r, c] = True   

    return mask  

def rotate_crop_to_ego(crop, direction):
    k = (4 - direction) % 4  
    if k == 0:
        return crop
    return np.rot90(crop, k=k, axes=(1, 2))
