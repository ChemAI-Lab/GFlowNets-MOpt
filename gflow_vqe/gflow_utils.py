import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, GINEConv, GATConv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.data import Data
from networkx import Graph
import networkx as nx
from collections import defaultdict
import torch
import numpy as np


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

class embTBModel(nn.Module):
  def __init__(self, num_hid, n_terms):
    super().__init__()
    num_emb_dim = 16  # Dimension of the embedding layer.
    layer_1d = num_hid
    layer_2d = num_hid 
    layer_3d = num_hid
    emb_layer = nn.Embedding(n_terms, embedding_dim=num_emb_dim)  # Embedding layer to convert input features to embeddings.
    self.emb_layer = emb_layer
    self.encode_layer = nn.Sequential(
        nn.Linear(n_terms*num_emb_dim, layer_1d),  # layers (Number of Paulis) input features.
        nn.LeakyReLU(),
        nn.Linear(layer_1d, layer_2d),
        nn.LeakyReLU(),
        nn.Linear(layer_2d, layer_3d),
        nn.LeakyReLU(),
    )
    self.logits_layer = nn.Linear(layer_3d, 2*n_terms)
    self.logZ = nn.Parameter(torch.ones(1))  # log Z is just a single number.

  def forward(self, x, n_terms):
    x_emb = self.emb_layer(x.long())  # Convert input features to embeddings.
    x_encoded = self.encode_layer(x_emb.flatten().unsqueeze(0))
    logits = self.logits_layer(x_encoded)
    # Slice the logits into forward and backward policies.
    P_F = logits[..., :n_terms]
    P_B = logits[..., n_terms:]

    return P_F, P_B

class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self, dim_h, n_terms, num_emb_dim):
        super(GIN, self).__init__()
        self.n_terms = n_terms # number of colors
        self.num_emb_dim = num_emb_dim # embeding dimension for colors

        self.emb_layer = nn.Embedding(n_terms, embedding_dim=self.num_emb_dim)  # Embedding layer to convert input features to embeddings.

        self.conv1 = GINEConv(
            nn.Sequential(nn.Linear(self.num_emb_dim, dim_h),
                       nn.BatchNorm1d(dim_h), nn.ReLU(),
                       nn.Linear(dim_h, dim_h), nn.ReLU()),
                      edge_dim=1)
        self.conv2 = GINEConv(
            nn.Sequential(nn.Linear(dim_h, dim_h), nn.BatchNorm1d(dim_h), nn.ReLU(),
                       nn.Linear(dim_h, dim_h), nn.ReLU()),
                      edge_dim=1)
        # self.conv3 = GINEConv(
        #     nn.Sequential(nn.Linear(dim_h, dim_h), nn.BatchNorm1d(dim_h), nn.ReLU(),
        #                nn.Linear(dim_h, dim_h), nn.ReLU()),
        #               edge_dim=1)

        self.logZ = nn.Parameter(torch.ones(1))  # log Z is just a single number

        # prediction of forward and backward probablity
        self.lin_logitz_f = nn.Sequential(nn.Linear(dim_h*2, dim_h), nn.ReLU(), nn.Linear(dim_h, self.n_terms))
        self.lin_logitz_b = nn.Sequential(nn.Linear(dim_h*2, dim_h), nn.ReLU(), nn.Linear(dim_h, self.n_terms))

    def forward(self, x, edge_index, edge_attr, batch):
        # x = x.unsqueeze(0)
        # Node embeddings
        x = self.emb_layer(x)
        x = x.squeeze(1)

        h1 = self.conv1(x, edge_index, edge_attr)
        h2 = self.conv2(h1, edge_index, edge_attr)
        #h3 = self.conv3(h2, edge_index, edge_attr)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        # h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2), dim=1)

        # # Classifier
        logits_f = self.lin_logitz_f(h)
        logits_b = self.lin_logitz_b(h)
        return  logits_f, logits_b

class GIN_2GPUParallel(nn.Module):
    def __init__(self, dim_h, n_terms, num_emb_dim):
        super().__init__()
        self.n_terms = n_terms
        self.num_emb_dim = num_emb_dim

        self.emb_layer = nn.Embedding(n_terms, embedding_dim=num_emb_dim).to('cuda:0')

        self.conv1 = GINEConv(
            nn.Sequential(
                nn.Linear(num_emb_dim, dim_h),
                nn.BatchNorm1d(dim_h),
                nn.ReLU(),
                nn.Linear(dim_h, dim_h),
                nn.ReLU()
            ), edge_dim=1).to('cuda:0')

        self.conv2 = GINEConv(
            nn.Sequential(
                nn.Linear(dim_h, dim_h),
                nn.BatchNorm1d(dim_h),
                nn.ReLU(),
                nn.Linear(dim_h, dim_h),
                nn.ReLU()
            ), edge_dim=1).to('cuda:1')  # Second layer on second GPU

        self.lin_logitz_f = nn.Sequential(
            nn.Linear(dim_h * 2, dim_h),
            nn.ReLU(),
            nn.Linear(dim_h, n_terms)
        ).to('cuda:1')

        self.lin_logitz_b = nn.Sequential(
            nn.Linear(dim_h * 2, dim_h),
            nn.ReLU(),
            nn.Linear(dim_h, n_terms)
        ).to('cuda:1')

        self.logZ = nn.Parameter(torch.ones(1).to('cuda:1'))

    def forward(self, x, edge_index, edge_attr, batch):
        # x on cuda:0
        x = x.to('cuda:0')
        edge_index = edge_index.to('cuda:0')
        edge_attr = edge_attr.to('cuda:0')
        batch0 = batch.to('cuda:0')

        x = self.emb_layer(x)
        x = x.squeeze(1)

        h1 = self.conv1(x, edge_index, edge_attr)
        h1 = h1.to('cuda:1')
        edge_index = edge_index.to('cuda:1')
        edge_attr = edge_attr.to('cuda:1')
        batch1 = batch0.to('cuda:1')

        h2 = self.conv2(h1, edge_index, edge_attr)

        h1_pool = global_add_pool(h1, batch1)
        h2_pool = global_add_pool(h2, batch1)

        h = torch.cat((h1_pool, h2_pool), dim=1)

        logits_f = self.lin_logitz_f(h)
        logits_b = self.lin_logitz_b(h)

        return logits_f, logits_b

class GIN_terms(torch.nn.Module):
    """GIN with terms of the Hamiltonian"""
    def __init__(self, dim_h, n_terms, num_emb_dim):
        super(GIN_terms, self).__init__()
        self.n_terms = n_terms # number of colors
        self.num_emb_dim = num_emb_dim # embeding dimension for colors

        self.emb_layer = nn.Embedding(n_terms, embedding_dim=self.num_emb_dim)  # Embedding layer to convert input features to embeddings.

        self.conv1 = GINEConv(
            nn.Sequential(nn.Linear(self.num_emb_dim+1, dim_h),
                       nn.BatchNorm1d(dim_h), nn.ReLU(),
                       nn.Linear(dim_h, dim_h), nn.ReLU()),
                      edge_dim=1)
        self.conv2 = GINEConv(
            nn.Sequential(nn.Linear(dim_h, dim_h), nn.BatchNorm1d(dim_h), nn.ReLU(),
                       nn.Linear(dim_h, dim_h), nn.ReLU()),
                      edge_dim=1)
        # self.conv3 = GINEConv(
        #     nn.Sequential(nn.Linear(dim_h, dim_h), nn.BatchNorm1d(dim_h), nn.ReLU(),
        #                nn.Linear(dim_h, dim_h), nn.ReLU()),
        #               edge_dim=1)

        self.logZ = nn.Parameter(torch.ones(1))  # log Z is just a single number

        # prediction of forward and backward probablity
        self.lin_logitz_f = nn.Sequential(nn.Linear(dim_h*2, dim_h), nn.ReLU(), nn.Linear(dim_h, self.n_terms))
        self.lin_logitz_b = nn.Sequential(nn.Linear(dim_h*2, dim_h), nn.ReLU(), nn.Linear(dim_h, self.n_terms))

    def forward(self, x, y, edge_index, edge_attr, batch):
        # x = x.unsqueeze(0)
        # Node embeddings
        x = self.emb_layer(x)
        x = x.squeeze(1)
        y = y.unsqueeze(1)
        xy = torch.cat((x, y), dim=1)

        h1 = self.conv1(xy, edge_index, edge_attr)
        h2 = self.conv2(h1, edge_index, edge_attr)
        #h3 = self.conv3(h2, edge_index, edge_attr)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        # h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2), dim=1)

        # # Classifier
        logits_f = self.lin_logitz_f(h)
        logits_b = self.lin_logitz_b(h)
        return  logits_f, logits_b

class GAT_terms(torch.nn.Module):
    """GAT with terms of the Hamiltonian"""
    def __init__(self, dim_h, n_terms, num_emb_dim, n_heads=4):
        super(GAT_terms, self).__init__()
        self.n_terms = n_terms  # number of colors
        self.num_emb_dim = num_emb_dim  # embeding dimension for colors

        self.emb_layer = nn.Embedding(n_terms, embedding_dim=self.num_emb_dim)

        self.conv1 = GATConv(self.num_emb_dim + 1, dim_h, heads=n_heads, concat=False, edge_dim=1)
        self.conv2 = GATConv(dim_h, dim_h, heads=n_heads, concat=False, edge_dim=1)

        self.logZ = nn.Parameter(torch.ones(1))  # log Z is just a single number

        # prediction of forward and backward probablity
        self.lin_logitz_f = nn.Sequential(nn.Linear(dim_h * 2, dim_h), nn.ReLU(), nn.Linear(dim_h, self.n_terms))
        self.lin_logitz_b = nn.Sequential(nn.Linear(dim_h * 2, dim_h), nn.ReLU(), nn.Linear(dim_h, self.n_terms))

    def forward(self, x, y, edge_index, edge_attr, batch):
        # Node embeddings
        x = self.emb_layer(x)
        x = x.squeeze(1)
        y = y.unsqueeze(1)
        xy = torch.cat((x, y), dim=1)
        edge_attr = edge_attr.float()

        h1 = self.conv1(xy, edge_index, edge_attr)
        h2 = self.conv2(h1, edge_index, edge_attr)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2), dim=1)

        # Classifier
        logits_f = self.lin_logitz_f(h)
        logits_b = self.lin_logitz_b(h)
        return logits_f, logits_b

class GraphTransformer_terms(torch.nn.Module):
    """Graph Transformer with terms of the Hamiltonian"""
    def __init__(self, dim_h, n_terms, num_emb_dim, n_heads=4):
        super(GraphTransformer_terms, self).__init__()
        self.n_terms = n_terms  # number of colors
        self.num_emb_dim = num_emb_dim  # embeding dimension for colors

        self.emb_layer = nn.Embedding(n_terms, embedding_dim=self.num_emb_dim)

        self.conv1 = TransformerConv(self.num_emb_dim + 1, dim_h, heads=n_heads, concat=False, edge_dim=1)
        self.conv2 = TransformerConv(dim_h, dim_h, heads=n_heads, concat=False, edge_dim=1)

        self.logZ = nn.Parameter(torch.ones(1))  # log Z is just a single number

        # prediction of forward and backward probablity
        self.lin_logitz_f = nn.Sequential(nn.Linear(dim_h * 2, dim_h), nn.ReLU(), nn.Linear(dim_h, self.n_terms))
        self.lin_logitz_b = nn.Sequential(nn.Linear(dim_h * 2, dim_h), nn.ReLU(), nn.Linear(dim_h, self.n_terms))

    def forward(self, x, y, edge_index, edge_attr, batch):
        # Node embeddings
        x = self.emb_layer(x)
        x = x.squeeze(1)
        y = y.unsqueeze(1)
        xy = torch.cat((x, y), dim=1)
        edge_attr = edge_attr.float()

        h1 = self.conv1(xy, edge_index, edge_attr)
        h2 = self.conv2(h1, edge_index, edge_attr)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2), dim=1)

        # Classifier
        logits_f = self.lin_logitz_f(h)
        logits_b = self.lin_logitz_b(h)
        return logits_f, logits_b

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
