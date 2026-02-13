from typing import Optional, Sequence, Tuple

import networkx as nx
import torch
from torch.distributions.categorical import Categorical
from torch_geometric.data import Data
from tqdm import trange

from gflow_vqe.advanced_losses import (
    gafn_joint_trajectory_loss,
    nabla_detailed_balance_loss,
)
from gflow_vqe.covariance_rewards import (
    CovarianceRewardData,
    CovarianceRewardEngine,
    build_covariance_reward_data,
)
from gflow_vqe.gflow_utils import GAT_terms, GIN_terms, GraphTransformer_terms
from gflow_vqe.training import get_training_device
from gflow_vqe.utils import generate_edge_index, graph_to_tensor, set_seed


_MODEL_MAP = {
    "gin": GIN_terms,
    "gat": GAT_terms,
    "transformer": GraphTransformer_terms,
}

_CHECKPOINT_PREFIX = {
    "gin": "_statevec_gin",
    "gat": "_statevec_gat",
    "transformer": "_statevec_graphTransformer",
}

_OBJECTIVE_SUFFIX = {
    "gafn": "GAFNmodel.pth",
    "nabla_db": "NablaDBmodel.pth",
}


def _checkpoint_suffix(model_name: str, objective: str) -> str:
    return _CHECKPOINT_PREFIX[model_name] + _OBJECTIVE_SUFFIX[objective]


def _forward_mask_from_color_tensor(
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


def _backward_mask_from_color_tensor(
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
        mask[neighbor_colors] = True
    return mask


def _resolve_covariance_data(
    graph: nx.Graph,
    covariance_data: Optional[CovarianceRewardData],
    wfn,
    n_q: Optional[int],
) -> CovarianceRewardData:
    if covariance_data is not None:
        return covariance_data
    if wfn is None or n_q is None:
        raise ValueError("Pass either covariance_data or both (wfn, n_q) to build covariance rewards.")
    return build_covariance_reward_data(graph=graph, wfn=wfn, n_qubit=n_q)


def run_graph_model_covariance_objective_training_state_vector(
    graph: nx.Graph,
    n_terms: int,
    n_hid_units: int,
    n_episodes: int,
    learning_rate: float,
    update_freq: int,
    seed: int,
    fig_name: str,
    n_emb: int,
    l0: float,
    l1: float,
    objective: str,
    model_name: str = "gin",
    covariance_data: Optional[CovarianceRewardData] = None,
    wfn=None,
    n_q: Optional[int] = None,
    alpha_edge: float = 1.0,
    alpha_terminal: float = 1.0,
    beta_nabla: float = 1.0,
    show_progress: bool = True,
) -> Tuple[Sequence[torch.Tensor], Sequence[float]]:
    """
    State-vector training with covariance-based rewards and one of:
      - objective='gafn'      (Generative Augmented Flow Networks)
      - objective='nabla_db'  (gradient-informed detailed balance)
    """
    model_name = model_name.lower()
    objective = objective.lower()
    if model_name not in _MODEL_MAP:
        raise ValueError("Unsupported model '{}'. Use one of: {}".format(model_name, sorted(_MODEL_MAP.keys())))
    if objective not in _OBJECTIVE_SUFFIX:
        raise ValueError(
            "Unsupported objective '{}'. Use one of: {}".format(objective, sorted(_OBJECTIVE_SUFFIX.keys()))
        )
    if update_freq <= 0:
        raise ValueError("update_freq must be > 0.")

    device = get_training_device()
    set_seed(seed)

    reward_data = _resolve_covariance_data(graph, covariance_data, wfn, n_q)
    reward_engine = CovarianceRewardEngine(graph=graph, reward_data=reward_data, l0=l0, l1=l1)

    n_nodes = nx.number_of_nodes(graph)
    if n_terms != n_nodes:
        raise ValueError("n_terms ({}) must match graph nodes ({}).".format(n_terms, n_nodes))
    if reward_data.n_terms != n_nodes:
        raise ValueError(
            "Covariance data n_terms ({}) must match graph nodes ({}).".format(reward_data.n_terms, n_nodes)
        )

    model = _MODEL_MAP[model_name](n_hid_units, n_terms, n_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    edge_index = generate_edge_index(graph).to(device)
    edge_attr = -1.0 * torch.ones((edge_index.shape[1], 1), dtype=torch.long, device=device)
    y = torch.tensor(
        [graph.nodes[i]["v"].coeff for i in range(n_nodes)],
        dtype=torch.float,
        device=device,
    )

    neighbor_idx = [torch.tensor(list(graph.neighbors(i)), dtype=torch.long, device=device) for i in range(n_nodes)]
    color_map = nx.coloring.greedy_color(graph, strategy="random_sequential")
    bound = max(color_map.values()) + 2

    base_state = graph_to_tensor(graph).long().to(device)
    base_reward = reward_engine.reward(base_state)

    losses = []
    sampled_colorings = []
    minibatch_loss = torch.zeros((), device=device)
    pending_episodes = 0
    tiny = reward_data.tiny

    iterator = trange(n_episodes, desc="Training {}".format(objective)) if show_progress else range(n_episodes)
    for episode in iterator:
        state_colors = base_state.clone()
        # Clone index tensors per forward to avoid autograd version errors when state_colors is updated in-place.
        x = state_colors.clone().unsqueeze(1)
        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
        p_f, p_b = model(data.x, data.y, data.edge_index, data.edge_attr, data.batch)
        p_f = p_f.squeeze(0)
        p_b = p_b.squeeze(0)

        total_log_p_f = torch.zeros((), device=device)
        log_p_f_terms = []
        log_p_b_terms = []
        reward_direction_terms = []
        log_augmented_backward_terms = []

        reward_prev = base_reward

        for t in range(n_nodes):
            mask_f = _forward_mask_from_color_tensor(state_colors, neighbor_idx[t], bound, n_terms)
            p_f_masked = torch.where(mask_f, p_f, p_f.new_full(p_f.shape, -100.0))
            dist_f = Categorical(logits=p_f_masked)
            action = dist_f.sample()
            action_i = int(action.item())

            log_p_f = dist_f.log_prob(action)
            total_log_p_f = total_log_p_f + log_p_f
            log_p_f_terms.append(log_p_f)

            state_colors[t] = action_i

            # Same rationale: keep index tensors immutable for previously built graph outputs.
            x_next = state_colors.clone().unsqueeze(1)
            data_next = Data(x=x_next, y=y, edge_index=edge_index, edge_attr=edge_attr)
            p_f, p_b = model(data_next.x, data_next.y, data_next.edge_index, data_next.edge_attr, data_next.batch)
            p_f = p_f.squeeze(0)
            p_b = p_b.squeeze(0)

            mask_b = _backward_mask_from_color_tensor(state_colors, neighbor_idx[t], bound, n_terms)
            p_b_masked = torch.where(mask_b, p_b, p_b.new_full(p_b.shape, -100.0))
            dist_b = Categorical(logits=p_b_masked)
            log_p_b = dist_b.log_prob(action)
            log_p_b_terms.append(log_p_b)

            reward_next = reward_engine.reward(state_colors)

            if objective == "gafn":
                r_edge = reward_engine.edge_intermediate_reward(
                    reward_prev=reward_prev,
                    reward_next=reward_next,
                    scale=alpha_edge,
                    clip_negative=True,
                )
                f_next = max(reward_next, tiny)
                pb_plus_aug = torch.exp(log_p_b) + (r_edge / f_next)
                log_augmented_backward_terms.append(torch.log(torch.clamp(pb_plus_aug, min=tiny)))

            if objective == "nabla_db":
                directional_gain = reward_engine.directional_log_reward_gain(
                    reward_prev=reward_prev,
                    reward_next=reward_next,
                )
                reward_direction_terms.append(
                    torch.tensor(directional_gain, device=device, dtype=log_p_f.dtype)
                )

            reward_prev = reward_next

        terminal_reward = reward_prev

        if objective == "gafn":
            gafn_loss = gafn_joint_trajectory_loss(
                log_z=model.logZ.squeeze(),
                total_log_p_f=total_log_p_f,
                terminal_reward=terminal_reward,
                terminal_intermediate_reward=float(alpha_terminal) * terminal_reward,
                log_augmented_backward_terms=log_augmented_backward_terms,
                tiny=tiny,
            )
        else:
            gafn_loss = torch.zeros((), device=device)

        if objective == "nabla_db":
            nabla_loss = nabla_detailed_balance_loss(
                log_p_f_terms=log_p_f_terms,
                log_p_b_terms=log_p_b_terms,
                reward_direction_terms=reward_direction_terms,
                beta=beta_nabla,
                reduction="mean",
            )
        else:
            nabla_loss = torch.zeros((), device=device)

        if objective == "gafn":
            trajectory_loss = gafn_loss
        elif objective == "nabla_db":
            trajectory_loss = nabla_loss
        else:
            raise ValueError("Unsupported objective '{}'.".format(objective))

        minibatch_loss = minibatch_loss + trajectory_loss
        pending_episodes += 1
        sampled_colorings.append(state_colors.detach().cpu().clone())

        should_step = ((episode + 1) % update_freq == 0) or (episode == n_episodes - 1)
        if should_step:
            mean_loss = minibatch_loss / float(pending_episodes)
            losses.append(float(mean_loss.item()))
            mean_loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            minibatch_loss = torch.zeros((), device=device)
            pending_episodes = 0
            torch.save(
                {
                    "epoch": episode,
                    "model_name": model_name,
                    "objective": objective,
                    "model_state_dict": model.cpu().state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": losses,
                    "alpha_edge": alpha_edge,
                    "alpha_terminal": alpha_terminal,
                    "beta_nabla": beta_nabla,
                },
                fig_name + _checkpoint_suffix(model_name, objective),
            )
            model.to(device)

    return sampled_colorings, losses


def _run_named_objective(
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
    model_name: str,
    objective: str,
    covariance_data: Optional[CovarianceRewardData] = None,
    alpha_edge: float = 1.0,
    alpha_terminal: float = 1.0,
    beta_nabla: float = 1.0,
    show_progress: bool = True,
):
    return run_graph_model_covariance_objective_training_state_vector(
        graph=graph,
        n_terms=n_terms,
        n_hid_units=n_hid_units,
        n_episodes=n_episodes,
        learning_rate=learning_rate,
        update_freq=update_freq,
        seed=seed,
        fig_name=fig_name,
        n_emb=n_emb,
        l0=l0,
        l1=l1,
        objective=objective,
        model_name=model_name,
        covariance_data=covariance_data,
        wfn=wfn,
        n_q=n_q,
        alpha_edge=alpha_edge,
        alpha_terminal=alpha_terminal,
        beta_nabla=beta_nabla,
        show_progress=show_progress,
    )


def coeff_GIN_GAFN_training_cov_reward_state_vector(
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
    covariance_data: Optional[CovarianceRewardData] = None,
    alpha_edge: float = 1.0,
    alpha_terminal: float = 1.0,
    show_progress: bool = True,
):
    return _run_named_objective(
        graph=graph,
        n_terms=n_terms,
        n_hid_units=n_hid_units,
        n_episodes=n_episodes,
        learning_rate=learning_rate,
        update_freq=update_freq,
        seed=seed,
        wfn=wfn,
        n_q=n_q,
        fig_name=fig_name,
        n_emb=n_emb,
        l0=l0,
        l1=l1,
        model_name="gin",
        objective="gafn",
        covariance_data=covariance_data,
        alpha_edge=alpha_edge,
        alpha_terminal=alpha_terminal,
        show_progress=show_progress,
    )


def coeff_GIN_nablaDB_training_cov_reward_state_vector(
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
    covariance_data: Optional[CovarianceRewardData] = None,
    beta_nabla: float = 1.0,
    show_progress: bool = True,
):
    return _run_named_objective(
        graph=graph,
        n_terms=n_terms,
        n_hid_units=n_hid_units,
        n_episodes=n_episodes,
        learning_rate=learning_rate,
        update_freq=update_freq,
        seed=seed,
        wfn=wfn,
        n_q=n_q,
        fig_name=fig_name,
        n_emb=n_emb,
        l0=l0,
        l1=l1,
        model_name="gin",
        objective="nabla_db",
        covariance_data=covariance_data,
        beta_nabla=beta_nabla,
        show_progress=show_progress,
    )


def coeff_GAT_GAFN_training_cov_reward_state_vector(
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
    covariance_data: Optional[CovarianceRewardData] = None,
    alpha_edge: float = 1.0,
    alpha_terminal: float = 1.0,
    show_progress: bool = True,
):
    return _run_named_objective(
        graph=graph,
        n_terms=n_terms,
        n_hid_units=n_hid_units,
        n_episodes=n_episodes,
        learning_rate=learning_rate,
        update_freq=update_freq,
        seed=seed,
        wfn=wfn,
        n_q=n_q,
        fig_name=fig_name,
        n_emb=n_emb,
        l0=l0,
        l1=l1,
        model_name="gat",
        objective="gafn",
        covariance_data=covariance_data,
        alpha_edge=alpha_edge,
        alpha_terminal=alpha_terminal,
        show_progress=show_progress,
    )


def coeff_GAT_nablaDB_training_cov_reward_state_vector(
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
    covariance_data: Optional[CovarianceRewardData] = None,
    beta_nabla: float = 1.0,
    show_progress: bool = True,
):
    return _run_named_objective(
        graph=graph,
        n_terms=n_terms,
        n_hid_units=n_hid_units,
        n_episodes=n_episodes,
        learning_rate=learning_rate,
        update_freq=update_freq,
        seed=seed,
        wfn=wfn,
        n_q=n_q,
        fig_name=fig_name,
        n_emb=n_emb,
        l0=l0,
        l1=l1,
        model_name="gat",
        objective="nabla_db",
        covariance_data=covariance_data,
        beta_nabla=beta_nabla,
        show_progress=show_progress,
    )


def coeff_Transformer_GAFN_training_cov_reward_state_vector(
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
    covariance_data: Optional[CovarianceRewardData] = None,
    alpha_edge: float = 1.0,
    alpha_terminal: float = 1.0,
    show_progress: bool = True,
):
    return _run_named_objective(
        graph=graph,
        n_terms=n_terms,
        n_hid_units=n_hid_units,
        n_episodes=n_episodes,
        learning_rate=learning_rate,
        update_freq=update_freq,
        seed=seed,
        wfn=wfn,
        n_q=n_q,
        fig_name=fig_name,
        n_emb=n_emb,
        l0=l0,
        l1=l1,
        model_name="transformer",
        objective="gafn",
        covariance_data=covariance_data,
        alpha_edge=alpha_edge,
        alpha_terminal=alpha_terminal,
        show_progress=show_progress,
    )


def coeff_Transformer_nablaDB_training_cov_reward_state_vector(
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
    covariance_data: Optional[CovarianceRewardData] = None,
    beta_nabla: float = 1.0,
    show_progress: bool = True,
):
    return _run_named_objective(
        graph=graph,
        n_terms=n_terms,
        n_hid_units=n_hid_units,
        n_episodes=n_episodes,
        learning_rate=learning_rate,
        update_freq=update_freq,
        seed=seed,
        wfn=wfn,
        n_q=n_q,
        fig_name=fig_name,
        n_emb=n_emb,
        l0=l0,
        l1=l1,
        model_name="transformer",
        objective="nabla_db",
        covariance_data=covariance_data,
        beta_nabla=beta_nabla,
        show_progress=show_progress,
    )

