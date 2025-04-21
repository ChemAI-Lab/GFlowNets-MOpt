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

class TBModel(nn.Module):
  def __init__(self, num_hid, n_terms):
    super().__init__()
    self.mlp = nn.Sequential(
        nn.Linear(n_terms, num_hid),  # layers (Number of Paulis) input features.
        nn.LeakyReLU(),
        nn.Linear(num_hid, 2*n_terms),  # double outputs: 1/2 for P_F and 1/2 for P_B.
    )
    self.logZ = nn.Parameter(torch.ones(1))  # log Z is just a single number.

  def forward(self, x, n_terms):
    logits = self.mlp(x)
    # Slice the logits into forward and backward policies.
    P_F = logits[..., :n_terms]
    P_B = logits[..., n_terms:]

    return P_F, P_B
  
class TBModel_seq(nn.Module):
  def __init__(self, num_hid, n_terms):
    super().__init__()
    self.mlp = nn.Sequential(
        nn.Linear(n_terms, num_hid),  # layers (Number of Paulis) input features.
        nn.LeakyReLU(),
        nn.Linear(num_hid, n_terms),  # double outputs: 1/2 for P_F and 1/2 for P_B. P_B not necessary since it is sequential. Only 1 parent.
    )
    self.logZ = nn.Parameter(torch.ones(1))  # log Z is just a single number.

  def forward(self, x, n_terms):
    logits = self.mlp(x)
    # Slice the logits into forward and backward policies.
    P_F = logits[..., :n_terms]

    return P_F
  
def calculate_forward_mask_from_state(state, t, lower_bound):
    """We want to mask the sampling to avoid any potential loss of time while training.
    In order to do so, we will have an upper bound on the number of colors used. Additionally,
    we can make the probability of using the same color in 2 neighbors = 0 to ensure validity.
    """
    layers=nx.number_of_nodes(state)
    mask = np.ones(layers)  # Allowed actions represented as 1, disallowed actions as 0.
    mask[lower_bound+1:] = 0
    neighbors = list(state.neighbors(t))
    neighbor_colors = [state.nodes[n]['color'] for n in neighbors]
    # Update the mask
    for color in neighbor_colors:
        mask[color] = 0    
    return torch.Tensor(mask).bool()

def calculate_backward_mask_from_state(state,t,lower_bound):
    """Here, we mask backward actions to only select parent nodes."""
    # This mask should be 1 for any action that could have led to the current state,
    # otherwise it should be zero.
    layers=nx.number_of_nodes(state)
    mask = np.ones(layers)  # Allowed actions represented as 1, disallowed actions as 0.
    mask[lower_bound+1:] = 0
    neighbors = list(state.neighbors(t))
    neighbor_colors = [state.nodes[n]['color'] for n in neighbors]
    # Update the mask
    for color in neighbor_colors:
        mask[color] = 1    
    return torch.Tensor(mask).bool()
