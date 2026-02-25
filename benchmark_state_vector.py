import time
import gc
from threading import Event, Thread
from typing import Dict, List, Tuple

import networkx as nx
import psutil
import torch
from openfermion.linalg import get_ground_state, get_sparse_operator
from openfermion.utils import count_qubits
from torch.distributions.categorical import Categorical
from torch_geometric.data import Data
from tqdm import trange

from gflow_vqe.gflow_utils import GIN_terms
from gflow_vqe.hamiltonians import *
from gflow_vqe.training import coeff_GIN_TB_training_custom_reward
from gflow_vqe.utils import (
    BinaryHamiltonian,
    FC_CompMatrix,
    custom_reward,
    generate_edge_index,
    get_groups_measurement,
    get_terms,
    max_color,
    obj_to_comp_graph,
    set_seed,
    trajectory_balance_loss,
)


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
    # Mirrors calculate_backward_mask_from_state in gflow_utils.py
    # (starts with ones, only clips upper-bound colors).
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


def _bytes_to_mib(n_bytes: int) -> float:
    return n_bytes / (1024.0 * 1024.0)


def _run_with_memory_tracking(run_fn, *args, **kwargs):
    """
    Execute a training function while tracking process RSS.
    Returns ((fn_result), elapsed_seconds, start_rss_bytes, peak_rss_bytes).
    """
    process = psutil.Process()
    start_rss = process.memory_info().rss
    peak_rss = start_rss
    stop_event = Event()

    def _sample_rss():
        nonlocal peak_rss
        while not stop_event.is_set():
            try:
                rss = process.memory_info().rss
            except psutil.Error:
                break
            if rss > peak_rss:
                peak_rss = rss
            time.sleep(0.01)

    sampler = Thread(target=_sample_rss, daemon=True)
    sampler.start()
    t0 = time.time()
    try:
        result = run_fn(*args, **kwargs)
    finally:
        elapsed = time.time() - t0
        stop_event.set()
        sampler.join(timeout=0.2)
        try:
            peak_rss = max(peak_rss, process.memory_info().rss)
        except psutil.Error:
            pass
    return result, elapsed, start_rss, peak_rss


def coeff_GIN_TB_training_custom_reward_state_vector(
    graph: nx.Graph,
    n_terms: int,
    n_hid_units: int,
    n_episodes: int,
    learning_rate: float,
    update_freq: int,
    seed: int,
    wfn,
    n_q: int,
    fig_name: str,
    n_emb: int,
    l0: float,
    l1: float,
) -> Tuple[List[torch.Tensor], List[float]]:
    """
    Same model and TB objective as coeff_GIN_TB_training_custom_reward,
    but keeps trajectory state as a color tensor instead of copying networkx
    graphs at every step.
    """
   # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    set_seed(seed)

    model = GIN_terms(n_hid_units, n_terms, n_emb).to(device)
    opt = torch.optim.Adam(model.parameters(), learning_rate)

    edge_index = generate_edge_index(graph).to(device)
    edge_attr = -1.0 * torch.ones((edge_index.shape[1], 1), dtype=torch.long, device=device)
    y = torch.tensor(
        [graph.nodes[i]["v"].coeff for i in range(nx.number_of_nodes(graph))],
        dtype=torch.float,
        device=device,
    )

    n_nodes = nx.number_of_nodes(graph)
    neighbor_idx = [
        torch.tensor(list(graph.neighbors(i)), dtype=torch.long, device=device)
        for i in range(n_nodes)
    ]

    color_map = nx.coloring.greedy_color(graph, strategy="random_sequential")
    bound = max(color_map.values()) + 2

    losses: List[float] = []
    sampled_colorings: List[torch.Tensor] = []
    minibatch_loss = 0

    # Reused for reward evaluation only.
    reward_graph = graph.copy()
    base_state = torch.zeros(n_nodes, dtype=torch.long, device=device)

    tbar = trange(n_episodes, desc="Training iter (state-vector)")
    for episode in tbar:
        state_colors = base_state.clone()
        # Clone to avoid sharing storage with `state_colors`, which is updated
        # in-place along the trajectory and would otherwise break autograd.
        x = state_colors.clone().unsqueeze(1)
        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
        P_F_s, P_B_s = model(data.x, data.y, data.edge_index, data.edge_attr, data.batch)
        total_log_P_F, total_log_P_B = 0, 0

        for t in range(n_nodes):
            mask_f = _forward_mask_from_colors(state_colors, neighbor_idx[t], bound, n_terms)
            P_F_s = torch.where(mask_f, P_F_s, -100)
            categorical = Categorical(logits=P_F_s)
            action = categorical.sample()
            state_colors[t] = action.item()
            total_log_P_F += categorical.log_prob(action)

            if t == n_nodes - 1:
                _set_graph_colors_from_tensor(reward_graph, state_colors)
                reward = custom_reward(reward_graph, wfn, n_q, l0, l1)

            # Keep each forward input immutable for autograd.
            x = state_colors.clone().unsqueeze(1)
            data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
            P_F_s, P_B_s = model(data.x, data.y, data.edge_index, data.edge_attr, data.batch)

            mask_b = _backward_mask_from_colors(state_colors, neighbor_idx[t], bound, n_terms)
            P_B_s = torch.where(mask_b, P_B_s, -100)
            total_log_P_B += Categorical(logits=P_B_s).log_prob(action)

        minibatch_loss += trajectory_balance_loss(
            model.logZ,
            total_log_P_F,
            total_log_P_B,
            reward,
        )

        sampled_colorings.append(state_colors.detach().cpu().clone())

        if episode % update_freq == 0:
            losses.append(minibatch_loss.item())
            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            minibatch_loss = 0
            torch.save(
                {
                    "epoch": episode,
                    "model_state_dict": model.cpu().state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "loss": losses,
                },
                fig_name + "_statevec_ginTBmodel.pth",
            )
            model.to(device)

    return sampled_colorings, losses


def _top_k_lines_from_graphs(
    sampled_graphs: List[nx.Graph],
    wfn,
    n_q: int,
    l0: float,
    l1: float,
    k: int = 5,
) -> List[str]:
    unique_graphs: Dict[Tuple[int, ...], nx.Graph] = {}
    for graph in sampled_graphs:
        colors = tuple(int(graph.nodes[i]["color"]) for i in range(nx.number_of_nodes(graph)))
        if colors not in unique_graphs:
            unique_graphs[colors] = graph

    ranked = sorted(
        unique_graphs.values(),
        key=lambda g: custom_reward(g, wfn, n_q, l0, l1),
        reverse=True,
    )

    lines: List[str] = []
    for graph in ranked[:k]:
        eps2_m = get_groups_measurement(graph, wfn, n_q)
        reward = custom_reward(graph, wfn, n_q, l0, l1)
        lines.append(
            "eps^2 M={} and max color {}. Reward= {}".format(
                eps2_m, max_color(graph), reward
            )
        )
    return lines


def _top_k_lines_from_colorings(
    base_graph: nx.Graph,
    sampled_colorings: List[torch.Tensor],
    wfn,
    n_q: int,
    l0: float,
    l1: float,
    k: int = 5,
) -> List[str]:
    unique_colorings: Dict[Tuple[int, ...], None] = {}
    for colors in sampled_colorings:
        key = tuple(int(c) for c in colors.tolist())
        unique_colorings[key] = None

    eval_graph = base_graph.copy()
    scored: List[Tuple[float, float, int]] = []
    for key in unique_colorings:
        for i, color in enumerate(key):
            eval_graph.nodes[i]["color"] = color
        reward = custom_reward(eval_graph, wfn, n_q, l0, l1)
        eps2_m = get_groups_measurement(eval_graph, wfn, n_q)
        scored.append((reward, eps2_m, max_color(eval_graph)))

    scored.sort(key=lambda x: x[0], reverse=True)
    lines = []
    for reward, eps2_m, n_color in scored[:k]:
        lines.append(
            "eps^2 M={} and max color {}. Reward= {}".format(
                eps2_m, n_color, reward
            )
        )
    return lines


def build_problem(molecule_builder):
    mol, H, _, n_paulis, Hq = molecule_builder()
    sparse_hamiltonian = get_sparse_operator(Hq)
    energy, fci_wfn = get_ground_state(sparse_hamiltonian)
    n_q = count_qubits(Hq)

    binary_H = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
    terms = get_terms(binary_H)
    comp_matrix = FC_CompMatrix(terms)
    graph = obj_to_comp_graph(terms, comp_matrix)
    n_terms = nx.number_of_nodes(graph)
    return graph, n_terms, fci_wfn, n_q, energy, n_paulis


def main():
    # Keep hyperparameters aligned with the existing driver defaults.
    n_hid_units = 64
    n_episodes = 1000
    learning_rate = 3e-4
    update_freq = 10
    seed = 45
    n_emb_dim = 2
    l0 = 1000
    l1 = 0

    molecule = parser()
    if molecule is None:
        raise ValueError("Unknown molecule name. Pass a valid molecule function (e.g., H2, H4, LiH, BeH2, N2).")
    molecule_name = molecule.__name__

    graph, n_terms, fci_wfn, n_q, energy, n_paulis = build_problem(molecule)
    print("{} problem loaded".format(molecule_name))
    print("Energy={}".format(energy))
    print("Number of Pauli products to measure: {}".format(n_paulis))
    print("Number of terms in the Hamiltonian: {}".format(n_terms))

    statevec_graph = graph.copy()
    baseline_graph = graph.copy()

    (sampled_colorings, _), statevec_time, statevec_start_mem, statevec_peak_mem = _run_with_memory_tracking(
        coeff_GIN_TB_training_custom_reward_state_vector,
        statevec_graph,
        n_terms,
        n_hid_units,
        n_episodes,
        learning_rate,
        update_freq,
        seed,
        fci_wfn,
        n_q,
        "{}_statevec_benchmark".format(molecule_name),
        n_emb_dim,
        l0,
        l1,
    )
    statevec_lines = _top_k_lines_from_colorings(statevec_graph, sampled_colorings, fci_wfn, n_q, l0, l1, k=5)
    del sampled_colorings
    gc.collect()

    (sampled_graphs, _), baseline_time, baseline_start_mem, baseline_peak_mem = _run_with_memory_tracking(
        coeff_GIN_TB_training_custom_reward,
        baseline_graph,
        n_terms,
        n_hid_units,
        n_episodes,
        learning_rate,
        update_freq,
        seed,
        fci_wfn,
        n_q,
        "{}_baseline_benchmark".format(molecule_name),
        n_emb_dim,
        l0,
        l1,
    )
    baseline_lines = _top_k_lines_from_graphs(sampled_graphs, fci_wfn, n_q, l0, l1, k=5)
    del sampled_graphs
    gc.collect()

    print("")
    print("State-vector version (in-place colors)")
    print("Training time: {:.2f} seconds".format(statevec_time))
    print(
        "Memory usage: start {:.2f} MiB, peak {:.2f} MiB, delta +{:.2f} MiB".format(
            _bytes_to_mib(statevec_start_mem),
            _bytes_to_mib(statevec_peak_mem),
            _bytes_to_mib(statevec_peak_mem - statevec_start_mem),
        )
    )
    for line in statevec_lines:
        print(line)

    print("")
    print("Baseline (networkx state copies)")
    print("Training time: {:.2f} seconds".format(baseline_time))
    print(
        "Memory usage: start {:.2f} MiB, peak {:.2f} MiB, delta +{:.2f} MiB".format(
            _bytes_to_mib(baseline_start_mem),
            _bytes_to_mib(baseline_peak_mem),
            _bytes_to_mib(baseline_peak_mem - baseline_start_mem),
        )
    )
    for line in baseline_lines:
        print(line)


if __name__ == "__main__":
    main()
