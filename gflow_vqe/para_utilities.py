import math
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch_geometric.nn import GATConv, GINEConv, TransformerConv, global_add_pool
from tqdm import trange

from gflow_vqe.gflow_utils import GAT_terms, GIN_terms, GraphTransformer_terms
from gflow_vqe.utils import (
    custom_reward,
    generate_edge_index,
    graph_to_tensor,
    set_seed,
    trajectory_balance_loss,
)


_MODEL_MAP = {
    "gin": GIN_terms,
    "gat": GAT_terms,
    "transformer": GraphTransformer_terms,
}

_MODEL_SUFFIX_MAP = {
    "gin": "_parallel_ginTBmodel.pth",
    "gat": "_parallel_gatTBmodel.pth",
    "transformer": "_parallel_graphTransformerTBmodel.pth",
    "gin_lite": "_parallel_ginLiteTBmodel.pth",
    "gat_lite": "_parallel_gatLiteTBmodel.pth",
    "transformer_lite": "_parallel_graphTransformerLiteTBmodel.pth",
}


# Shared worker state, initialized once per subprocess.
_WORKER: Dict[str, object] = {}

Trajectory = Tuple[Tuple[int, ...], float]


class GIN_terms_lite(torch.nn.Module):
    """Memory-lighter single-conv GINE variant for parallel sampling."""

    def __init__(self, dim_h, n_terms, num_emb_dim):
        super().__init__()
        self.n_terms = n_terms
        self.emb_layer = nn.Embedding(n_terms, embedding_dim=num_emb_dim)
        self.conv1 = GINEConv(
            nn.Sequential(
                nn.Linear(num_emb_dim + 1, dim_h),
                nn.ReLU(),
                nn.Linear(dim_h, dim_h),
                nn.ReLU(),
            ),
            edge_dim=1,
        )
        self.logZ = nn.Parameter(torch.ones(1))
        self.lin_logitz_f = nn.Sequential(nn.Linear(dim_h, dim_h), nn.ReLU(), nn.Linear(dim_h, n_terms))
        self.lin_logitz_b = nn.Sequential(nn.Linear(dim_h, dim_h), nn.ReLU(), nn.Linear(dim_h, n_terms))

    def forward(self, x, y, edge_index, edge_attr, batch):
        x = self.emb_layer(x).squeeze(1)
        xy = torch.cat((x, y.unsqueeze(1)), dim=1)
        h = self.conv1(xy, edge_index, edge_attr)
        h = global_add_pool(h, batch)
        return self.lin_logitz_f(h), self.lin_logitz_b(h)


class GAT_terms_lite(torch.nn.Module):
    """Memory-lighter single-conv GAT variant for parallel sampling."""

    def __init__(self, dim_h, n_terms, num_emb_dim, n_heads=2):
        super().__init__()
        self.n_terms = n_terms
        self.emb_layer = nn.Embedding(n_terms, embedding_dim=num_emb_dim)
        self.conv1 = GATConv(num_emb_dim + 1, dim_h, heads=n_heads, concat=False, edge_dim=1)
        self.logZ = nn.Parameter(torch.ones(1))
        self.lin_logitz_f = nn.Sequential(nn.Linear(dim_h, dim_h), nn.ReLU(), nn.Linear(dim_h, n_terms))
        self.lin_logitz_b = nn.Sequential(nn.Linear(dim_h, dim_h), nn.ReLU(), nn.Linear(dim_h, n_terms))

    def forward(self, x, y, edge_index, edge_attr, batch):
        x = self.emb_layer(x).squeeze(1)
        xy = torch.cat((x, y.unsqueeze(1)), dim=1)
        h = self.conv1(xy, edge_index, edge_attr.float())
        h = global_add_pool(h, batch)
        return self.lin_logitz_f(h), self.lin_logitz_b(h)


class GraphTransformer_terms_lite(torch.nn.Module):
    """Memory-lighter single-conv GraphTransformer variant for parallel sampling."""

    def __init__(self, dim_h, n_terms, num_emb_dim, n_heads=2):
        super().__init__()
        self.n_terms = n_terms
        self.emb_layer = nn.Embedding(n_terms, embedding_dim=num_emb_dim)
        self.conv1 = TransformerConv(num_emb_dim + 1, dim_h, heads=n_heads, concat=False, edge_dim=1)
        self.logZ = nn.Parameter(torch.ones(1))
        self.lin_logitz_f = nn.Sequential(nn.Linear(dim_h, dim_h), nn.ReLU(), nn.Linear(dim_h, n_terms))
        self.lin_logitz_b = nn.Sequential(nn.Linear(dim_h, dim_h), nn.ReLU(), nn.Linear(dim_h, n_terms))

    def forward(self, x, y, edge_index, edge_attr, batch):
        x = self.emb_layer(x).squeeze(1)
        xy = torch.cat((x, y.unsqueeze(1)), dim=1)
        h = self.conv1(xy, edge_index, edge_attr.float())
        h = global_add_pool(h, batch)
        return self.lin_logitz_f(h), self.lin_logitz_b(h)


_MODEL_MAP.update(
    {
        "gin_lite": GIN_terms_lite,
        "gat_lite": GAT_terms_lite,
        "transformer_lite": GraphTransformer_terms_lite,
    }
)


@dataclass
class ParallelTrainingResult:
    sampled_graphs: List[nx.Graph]
    sampled_colorings: List[torch.Tensor]
    losses: List[float]
    mean_rewards: List[float]
    expected_total_samples: int
    actual_total_samples: int
    used_multiprocessing: bool


def model_class_from_name(model_name: str):
    key = model_name.lower()
    if key not in _MODEL_MAP:
        raise ValueError(
            "Unsupported model '{}'. Expected one of: {}".format(
                model_name, ", ".join(sorted(_MODEL_MAP.keys()))
            )
        )
    return _MODEL_MAP[key]


def model_checkpoint_suffix(model_name: str) -> str:
    key = model_name.lower()
    if key not in _MODEL_SUFFIX_MAP:
        raise ValueError(
            "Unsupported model '{}'. Expected one of: {}".format(
                model_name, ", ".join(sorted(_MODEL_SUFFIX_MAP.keys()))
            )
        )
    return _MODEL_SUFFIX_MAP[key]


def expected_sample_count(num_updates: int, num_workers: int, episodes_per_worker: int) -> int:
    return int(num_updates) * int(num_workers) * int(episodes_per_worker)


def _graph_batch_tensors(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    y: torch.Tensor,
    n_nodes: int,
    batch_size: int,
    cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
):
    """Replicate a single graph topology/features into a batched graph."""
    if batch_size in cache:
        return cache[batch_size]

    if batch_size == 1:
        batch = torch.zeros(n_nodes, dtype=torch.long, device=edge_index.device)
        out = (edge_index, edge_attr, y, batch)
        cache[batch_size] = out
        return out

    offsets = (
        torch.arange(batch_size, device=edge_index.device, dtype=edge_index.dtype).view(batch_size, 1, 1) * n_nodes
    )
    edge_index_b = (edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1).contiguous()
    edge_attr_b = edge_attr.repeat(batch_size, 1)
    y_b = y.repeat(batch_size)
    batch = torch.arange(batch_size, device=edge_index.device, dtype=torch.long).repeat_interleave(n_nodes)
    out = (edge_index_b, edge_attr_b, y_b, batch)
    cache[batch_size] = out
    return out


def _forward_mask_from_colors_batch(
    colors_batch: torch.Tensor,
    neighbor_idx: torch.Tensor,
    lower_bound: int,
    n_terms: int,
) -> torch.Tensor:
    bsz = colors_batch.shape[0]
    mask = torch.ones((bsz, n_terms), dtype=torch.bool, device=colors_batch.device)
    if lower_bound + 1 < n_terms:
        mask[:, lower_bound + 1 :] = False
    if neighbor_idx.numel() > 0:
        neighbor_colors = colors_batch.index_select(1, neighbor_idx).long()
        rows = torch.arange(bsz, device=colors_batch.device).unsqueeze(1).expand_as(neighbor_colors)
        mask[rows, neighbor_colors] = False
    return mask


def _backward_mask_from_colors_batch(
    colors_batch: torch.Tensor,
    neighbor_idx: torch.Tensor,
    lower_bound: int,
    n_terms: int,
) -> torch.Tensor:
    # Mirrors current backward-mask behavior in training.py.
    bsz = colors_batch.shape[0]
    mask = torch.ones((bsz, n_terms), dtype=torch.bool, device=colors_batch.device)
    if lower_bound + 1 < n_terms:
        mask[:, lower_bound + 1 :] = False
    if neighbor_idx.numel() > 0:
        neighbor_colors = colors_batch.index_select(1, neighbor_idx).long()
        rows = torch.arange(bsz, device=colors_batch.device).unsqueeze(1).expand_as(neighbor_colors)
        mask[rows, neighbor_colors] = True
    return mask


def _forward_mask_from_colors(
    colors: torch.Tensor,
    neighbor_idx: torch.Tensor,
    lower_bound: int,
    n_terms: int,
) -> torch.Tensor:
    mask = torch.ones(n_terms, dtype=torch.bool, device=colors.device)
    if lower_bound + 1 < n_terms:
        mask[lower_bound + 1 :] = False
    if neighbor_idx.numel() > 0:
        neighbor_colors = colors[neighbor_idx].long()
        mask[neighbor_colors] = False
    return mask


def _backward_mask_from_colors(
    colors: torch.Tensor,
    neighbor_idx: torch.Tensor,
    lower_bound: int,
    n_terms: int,
) -> torch.Tensor:
    # Keep behavior aligned with current training.py mask logic.
    mask = torch.ones(n_terms, dtype=torch.bool, device=colors.device)
    if lower_bound + 1 < n_terms:
        mask[lower_bound + 1 :] = False
    if neighbor_idx.numel() > 0:
        neighbor_colors = colors[neighbor_idx].long()
        mask[neighbor_colors] = True
    return mask


def _set_graph_colors_from_tensor(graph: nx.Graph, colors: torch.Tensor) -> None:
    for i in range(colors.numel()):
        graph.nodes[i]["color"] = int(colors[i].item())


def sampled_colorings_to_graphs(
    base_graph: nx.Graph,
    sampled_colorings: Sequence[torch.Tensor],
) -> List[nx.Graph]:
    sampled_graphs: List[nx.Graph] = []
    for colors in sampled_colorings:
        g = base_graph.copy()
        _set_graph_colors_from_tensor(g, colors)
        sampled_graphs.append(g)
    return sampled_graphs


def _init_worker(
    base_graph: nx.Graph,
    y_cpu: torch.Tensor,
    edge_index_cpu: torch.Tensor,
    edge_attr_cpu: torch.Tensor,
    base_colors_cpu: torch.Tensor,
    n_terms: int,
    n_hid_units: int,
    n_emb: int,
    bound: int,
    seed: int,
    wfn,
    n_q: int,
    l0: float,
    l1: float,
    model_name: str,
    representation: str,
) -> None:
    torch.set_num_threads(1)

    _WORKER["base_graph"] = base_graph
    _WORKER["y_cpu"] = y_cpu
    _WORKER["edge_index_cpu"] = edge_index_cpu
    _WORKER["edge_attr_cpu"] = edge_attr_cpu
    _WORKER["base_colors_cpu"] = base_colors_cpu
    _WORKER["n_terms"] = n_terms
    _WORKER["n_hid_units"] = n_hid_units
    _WORKER["n_emb"] = n_emb
    _WORKER["bound"] = bound
    _WORKER["seed"] = seed
    _WORKER["wfn"] = wfn
    _WORKER["n_q"] = n_q
    _WORKER["l0"] = l0
    _WORKER["l1"] = l1
    _WORKER["model_name"] = model_name
    _WORKER["representation"] = representation

    n_nodes = nx.number_of_nodes(base_graph)
    _WORKER["n_nodes"] = n_nodes
    _WORKER["neighbor_idx"] = [
        torch.tensor(list(base_graph.neighbors(i)), dtype=torch.long) for i in range(n_nodes)
    ]
    _WORKER["batch_cpu"] = torch.zeros(n_nodes, dtype=torch.long)
    model_cls = model_class_from_name(model_name)
    worker_model = model_cls(n_hid_units, n_terms, n_emb)
    worker_model.eval()
    _WORKER["model"] = worker_model


def _worker_load_model(model_state: Dict[str, torch.Tensor]):
    model = _WORKER["model"]
    model.load_state_dict(model_state)
    model.eval()
    return model


def _worker_model_forward(
    model,
    state_colors: torch.Tensor,
    state_graph: Optional[nx.Graph],
    y_cpu: torch.Tensor,
    edge_index_cpu: torch.Tensor,
    edge_attr_cpu: torch.Tensor,
    batch_cpu: torch.Tensor,
    representation: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if representation == "networkx":
        x = graph_to_tensor(state_graph).unsqueeze(1).long()
    else:
        x = state_colors.clone().unsqueeze(1).long()

    p_f, p_b = model(x, y_cpu, edge_index_cpu, edge_attr_cpu, batch_cpu)
    return p_f.squeeze(0), p_b.squeeze(0)


def _worker_model_forward_state_batch(
    model,
    states_batch: torch.Tensor,
    y_cpu: torch.Tensor,
    edge_index_cpu: torch.Tensor,
    edge_attr_cpu: torch.Tensor,
    n_nodes: int,
    cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz = states_batch.shape[0]
    edge_index_b, edge_attr_b, y_b, batch_b = _graph_batch_tensors(
        edge_index_cpu,
        edge_attr_cpu,
        y_cpu,
        n_nodes,
        bsz,
        cache,
    )
    x = states_batch.reshape(-1, 1).long()
    p_f, p_b = model(x, y_b, edge_index_b, edge_attr_b, batch_b)
    return p_f, p_b


def _sample_worker(
    args: Tuple[int, Dict[str, torch.Tensor], int, int],
) -> List[Trajectory]:
    worker_id, model_state, episodes_per_worker, update_idx = args
    set_seed(int(_WORKER["seed"]) + worker_id + 10_000 * update_idx)

    n_nodes = int(_WORKER["n_nodes"])
    n_terms = int(_WORKER["n_terms"])
    bound = int(_WORKER["bound"])
    representation = str(_WORKER["representation"])

    model = _worker_load_model(model_state)

    base_graph = _WORKER["base_graph"]
    y_cpu = _WORKER["y_cpu"]
    edge_index_cpu = _WORKER["edge_index_cpu"]
    edge_attr_cpu = _WORKER["edge_attr_cpu"]
    batch_cpu = _WORKER["batch_cpu"]
    base_colors_cpu = _WORKER["base_colors_cpu"]
    neighbor_idx = _WORKER["neighbor_idx"]

    reward_graph = base_graph.copy()
    trajectories: List[Trajectory] = []

    with torch.no_grad():
        if representation == "state_vector":
            states_batch = base_colors_cpu.unsqueeze(0).repeat(episodes_per_worker, 1)
            batch_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

            for t in range(n_nodes):
                p_f, _ = _worker_model_forward_state_batch(
                    model,
                    states_batch,
                    y_cpu,
                    edge_index_cpu,
                    edge_attr_cpu,
                    n_nodes,
                    batch_cache,
                )
                mask_f = _forward_mask_from_colors_batch(states_batch, neighbor_idx[t], bound, n_terms)
                p_f = torch.where(mask_f, p_f, p_f.new_full(p_f.shape, -100.0))
                actions = Categorical(logits=p_f).sample()
                states_batch[:, t] = actions

            for b in range(episodes_per_worker):
                colors = states_batch[b]
                _set_graph_colors_from_tensor(reward_graph, colors)
                reward = float(custom_reward(reward_graph, _WORKER["wfn"], _WORKER["n_q"], _WORKER["l0"], _WORKER["l1"]))
                trajectories.append((tuple(int(c) for c in colors.tolist()), reward))
        else:
            for _ in range(episodes_per_worker):
                state_colors = base_colors_cpu.clone()
                state_graph = base_graph.copy() if representation == "networkx" else None

                for t in range(n_nodes):
                    p_f, _ = _worker_model_forward(
                        model,
                        state_colors,
                        state_graph,
                        y_cpu,
                        edge_index_cpu,
                        edge_attr_cpu,
                        batch_cpu,
                        representation,
                    )
                    mask_f = _forward_mask_from_colors(state_colors, neighbor_idx[t], bound, n_terms)
                    p_f = torch.where(mask_f, p_f, p_f.new_full(p_f.shape, -100.0))

                    action = Categorical(logits=p_f).sample()
                    action_i = int(action.item())
                    state_colors[t] = action_i

                    if state_graph is not None:
                        state_graph.nodes[t]["color"] = action_i

                _set_graph_colors_from_tensor(reward_graph, state_colors)
                reward = float(custom_reward(reward_graph, _WORKER["wfn"], _WORKER["n_q"], _WORKER["l0"], _WORKER["l1"]))
                trajectories.append((tuple(int(c) for c in state_colors.tolist()), reward))

    return trajectories


def _replay_and_update(
    model,
    optimizer: torch.optim.Optimizer,
    trajectories: Sequence[Trajectory],
    base_graph: nx.Graph,
    y: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    batch: torch.Tensor,
    base_colors: torch.Tensor,
    neighbor_idx: Sequence[torch.Tensor],
    bound: int,
    n_terms: int,
    device: torch.device,
    representation: str,
    replay_batch_size: int,
) -> float:
    if not trajectories:
        return 0.0

    if representation == "state_vector":
        model.train()
        optimizer.zero_grad(set_to_none=True)

        n_nodes = len(neighbor_idx)
        total = len(trajectories)
        actions_all = torch.tensor([t[0] for t in trajectories], dtype=torch.long, device=device)
        rewards_all = torch.tensor([t[1] for t in trajectories], dtype=torch.float, device=device)
        rewards_all = torch.clamp(rewards_all, min=1e-30)

        batch_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        chunk = replay_batch_size if replay_batch_size > 0 else total
        loss_sum = torch.zeros((), device=device)

        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            actions = actions_all[start:end]
            rewards = rewards_all[start:end]
            bsz = actions.shape[0]

            states = base_colors.unsqueeze(0).expand(bsz, -1).clone()
            total_log_pf = torch.zeros(bsz, device=device)
            total_log_pb = torch.zeros(bsz, device=device)

            for t in range(n_nodes):
                edge_index_b, edge_attr_b, y_b, batch_b = _graph_batch_tensors(
                    edge_index,
                    edge_attr,
                    y,
                    n_nodes,
                    bsz,
                    batch_cache,
                )
                # Clone to avoid in-place state updates invalidating embedding indices
                # needed by autograd for previously computed graph outputs.
                x = states.clone().reshape(-1, 1).long()
                p_f, _ = model(x, y_b, edge_index_b, edge_attr_b, batch_b)
                mask_f = _forward_mask_from_colors_batch(states, neighbor_idx[t], bound, n_terms)
                p_f = torch.where(mask_f, p_f, p_f.new_full(p_f.shape, -100.0))

                action_t = actions[:, t]
                total_log_pf = total_log_pf + torch.log_softmax(p_f, dim=-1).gather(1, action_t.unsqueeze(1)).squeeze(1)
                states[:, t] = action_t

                # Same rationale as above: keep index tensors immutable per forward.
                x_next = states.clone().reshape(-1, 1).long()
                _, p_b = model(x_next, y_b, edge_index_b, edge_attr_b, batch_b)
                mask_b = _backward_mask_from_colors_batch(states, neighbor_idx[t], bound, n_terms)
                p_b = torch.where(mask_b, p_b, p_b.new_full(p_b.shape, -100.0))
                total_log_pb = total_log_pb + torch.log_softmax(p_b, dim=-1).gather(1, action_t.unsqueeze(1)).squeeze(1)

            chunk_loss = (model.logZ + total_log_pf + total_log_pb - torch.log(rewards)).pow(2).sum()
            loss_sum = loss_sum + chunk_loss

        loss = loss_sum / total
        loss.backward()
        optimizer.step()
        return float(loss.item())

    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = torch.zeros((), device=device)
    n_nodes = len(neighbor_idx)

    for actions, reward in trajectories:
        state_colors = base_colors.clone().to(device)
        state_graph = base_graph.copy() if representation == "networkx" else None

        total_log_pf = torch.zeros((), device=device)
        total_log_pb = torch.zeros((), device=device)

        for t in range(n_nodes):
            if representation == "networkx":
                x = graph_to_tensor(state_graph).unsqueeze(1).long().to(device)
            else:
                x = state_colors.clone().unsqueeze(1).long()

            p_f, _ = model(x, y, edge_index, edge_attr, batch)
            p_f = p_f.squeeze(0)

            mask_f = _forward_mask_from_colors(state_colors, neighbor_idx[t], bound, n_terms)
            p_f = torch.where(mask_f, p_f, p_f.new_full(p_f.shape, -100.0))

            action_i = int(actions[t])
            total_log_pf = total_log_pf + torch.log_softmax(p_f, dim=-1)[action_i]

            state_colors[t] = action_i
            if state_graph is not None:
                state_graph.nodes[t]["color"] = action_i

            if representation == "networkx":
                x_next = graph_to_tensor(state_graph).unsqueeze(1).long().to(device)
            else:
                x_next = state_colors.clone().unsqueeze(1).long()

            _, p_b = model(x_next, y, edge_index, edge_attr, batch)
            p_b = p_b.squeeze(0)

            mask_b = _backward_mask_from_colors(state_colors, neighbor_idx[t], bound, n_terms)
            p_b = torch.where(mask_b, p_b, p_b.new_full(p_b.shape, -100.0))
            total_log_pb = total_log_pb + torch.log_softmax(p_b, dim=-1)[action_i]

        total_loss = total_loss + trajectory_balance_loss(model.logZ, total_log_pf, total_log_pb, reward)

    total_loss = total_loss / len(trajectories)
    total_loss.backward()
    optimizer.step()
    return float(total_loss.item())


def run_parallel_graph_model_custom_reward_training(
    graph: nx.Graph,
    n_terms: int,
    n_hid_units: int,
    n_emb: int,
    num_updates: int,
    episodes_per_worker: int,
    num_workers: int,
    learning_rate: float,
    seed: int,
    wfn,
    n_q: int,
    l0: float,
    l1: float,
    model_name: str,
    representation: str,
    gpu: bool,
    fig_name: str,
    replay_batch_size: int = 64,
    show_progress: bool = True,
) -> ParallelTrainingResult:
    model_name = model_name.lower()
    representation = representation.lower()
    if representation not in {"networkx", "state_vector"}:
        raise ValueError(
            "Unsupported representation '{}'. Expected one of: networkx, state_vector".format(
                representation
            )
        )

    device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
    set_seed(seed)

    model_cls = model_class_from_name(model_name)
    model = model_cls(n_hid_units, n_terms, n_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    edge_index = generate_edge_index(graph).to(device)
    edge_attr = -1.0 * torch.ones((edge_index.shape[1], 1), dtype=torch.long, device=device)
    batch = torch.zeros(nx.number_of_nodes(graph), dtype=torch.long, device=device)
    y = torch.tensor(
        [graph.nodes[i]["v"].coeff for i in range(nx.number_of_nodes(graph))],
        dtype=torch.float,
        device=device,
    )

    n_nodes = nx.number_of_nodes(graph)
    base_colors = graph_to_tensor(graph).long().to(device)

    neighbor_idx = [
        torch.tensor(list(graph.neighbors(i)), dtype=torch.long, device=device) for i in range(n_nodes)
    ]

    color_map = nx.coloring.greedy_color(graph, strategy="random_sequential")
    bound = max(color_map.values()) + 2

    y_cpu = y.detach().cpu()
    edge_index_cpu = edge_index.detach().cpu()
    edge_attr_cpu = edge_attr.detach().cpu()
    base_colors_cpu = base_colors.detach().cpu()

    expected_total = expected_sample_count(num_updates, num_workers, episodes_per_worker)

    losses: List[float] = []
    mean_rewards: List[float] = []
    sampled_colorings: List[torch.Tensor] = []

    init_args = (
        graph,
        y_cpu,
        edge_index_cpu,
        edge_attr_cpu,
        base_colors_cpu,
        n_terms,
        n_hid_units,
        n_emb,
        bound,
        seed,
        wfn,
        n_q,
        l0,
        l1,
        model_name,
        representation,
    )

    used_pool = num_workers > 1
    pool = None

    if used_pool:
        try:
            ctx_name = "spawn"
            if not gpu and "fork" in mp.get_all_start_methods():
                ctx_name = "fork"
            ctx = mp.get_context(ctx_name)
            pool = ctx.Pool(processes=num_workers, initializer=_init_worker, initargs=init_args)
        except (PermissionError, OSError) as exc:
            print("Multiprocessing unavailable ({}). Falling back to serial worker emulation.".format(exc))
            used_pool = False

    def _run_one_update(update_idx: int, trajectories: Sequence[Trajectory]):
        mean_loss = _replay_and_update(
            model=model,
            optimizer=optimizer,
            trajectories=trajectories,
            base_graph=graph,
            y=y,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            base_colors=base_colors,
            neighbor_idx=neighbor_idx,
            bound=bound,
            n_terms=n_terms,
            device=device,
            representation=representation,
            replay_batch_size=replay_batch_size,
        )
        losses.append(mean_loss)

        if trajectories:
            mean_reward = float(sum(t[1] for t in trajectories) / len(trajectories))
            best_reward = float(max(t[1] for t in trajectories))
        else:
            mean_reward = 0.0
            best_reward = 0.0
        mean_rewards.append(mean_reward)

        for colors, _ in trajectories:
            sampled_colorings.append(torch.tensor(colors, dtype=torch.long))

        if show_progress and (update_idx + 1) % max(1, math.ceil(num_updates / 20)) == 0:
            print(
                "Update {}/{} | loss={:.6f} | best_reward={:.6f} | logZ={:.4f}".format(
                    update_idx + 1,
                    num_updates,
                    mean_loss,
                    best_reward,
                    model.logZ.item(),
                )
            )

    iterator = trange(num_updates, desc="Parallel updates") if show_progress else range(num_updates)

    if used_pool and pool is not None:
        with pool:
            for update_idx in iterator:
                model_state = {
                    k: v.detach().cpu().clone().share_memory_() for k, v in model.state_dict().items()
                }
                worker_args = [
                    (wid, model_state, episodes_per_worker, update_idx) for wid in range(num_workers)
                ]
                worker_batches = pool.map(_sample_worker, worker_args)
                trajectories = [traj for batch in worker_batches for traj in batch]
                _run_one_update(update_idx, trajectories)
    else:
        _init_worker(*init_args)
        for update_idx in iterator:
            model_state = {
                k: v.detach().cpu().clone().share_memory_() for k, v in model.state_dict().items()
            }
            worker_args = [
                (wid, model_state, episodes_per_worker, update_idx) for wid in range(num_workers)
            ]
            worker_batches = [_sample_worker(a) for a in worker_args]
            trajectories = [traj for batch in worker_batches for traj in batch]
            _run_one_update(update_idx, trajectories)

    ckpt_path = fig_name + model_checkpoint_suffix(model_name)
    torch.save(
        {
            "model_name": model_name,
            "representation": representation,
            "model_state_dict": model.cpu().state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": losses,
            "mean_rewards": mean_rewards,
            "expected_total_samples": expected_total,
            "actual_total_samples": len(sampled_colorings),
        },
        ckpt_path,
    )
    model.to(device)

    sampled_graphs = sampled_colorings_to_graphs(graph, sampled_colorings)

    return ParallelTrainingResult(
        sampled_graphs=sampled_graphs,
        sampled_colorings=sampled_colorings,
        losses=losses,
        mean_rewards=mean_rewards,
        expected_total_samples=expected_total,
        actual_total_samples=len(sampled_colorings),
        used_multiprocessing=used_pool,
    )
