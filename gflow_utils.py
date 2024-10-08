import torch
import torch.nn as nn
from gflownet.envs.graph_building_env import *


class FlowModel(nn.Module):
  def __init__(self, num_hid, n_terms):
    super().__init__()
    self.mlp = nn.Sequential(
        nn.Linear(n_terms, num_hid),  # Input state is a binary vector representing feature
                                # presence and/or absence.
        nn.Tanh(),
        nn.Linear(num_hid, n_terms)  # Output #layer numbers for the #layer possible actions (child states).
    )#.to(device)

  def forward(self, x):
    return self.mlp(x).exp()  # Flows must be positive, so we take the exponential.

def calculate_forward_mask_from_state(state, t, lower_bound):
    """We want to mask the sampling to avoid any potential loss of time while training.
    In order to do so, we will have an upper bound on the number of colors used. Additionally,
    we can make the probability of using the same color in 2 neighbors = 0.
    """
    layers=nx.number_of_nodes(state)
    mask = np.ones(layers)  # Allowed actions represented as 1, disallowed actions as 0.
    mask[lower_bound+3:] = 0
    neighbors = list(state.neighbors(t))
    neighbor_colors = [state.nodes[n]['color'] for n in neighbors]
    # Update the mask
    for color in neighbor_colors:
        mask[color] = 0    
    return torch.Tensor(mask).bool()

