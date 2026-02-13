from typing import Sequence

import torch


def _as_scalar_tensor(value, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(float(value), device=device, dtype=dtype)


def gafn_joint_trajectory_loss(
    log_z: torch.Tensor,
    total_log_p_f: torch.Tensor,
    terminal_reward: float,
    terminal_intermediate_reward: float,
    log_augmented_backward_terms: Sequence[torch.Tensor],
    tiny: float = 1e-30,
) -> torch.Tensor:
    """
    Joint GAFN-style trajectory loss with augmented edge and terminal rewards.
    Implements a log-space form of:
        (logZ + log P_F(tau) - log((R(x) + r(x)) * prod_t [P_B + r_t/F]))^2
    """
    device = total_log_p_f.device
    dtype = total_log_p_f.dtype

    augmented_terminal = max(float(terminal_reward) + float(terminal_intermediate_reward), tiny)
    log_rhs = torch.log(torch.tensor(augmented_terminal, device=device, dtype=dtype))
    if log_augmented_backward_terms:
        log_rhs = log_rhs + torch.stack(list(log_augmented_backward_terms)).sum()

    return (log_z + total_log_p_f - log_rhs).pow(2)


def nabla_detailed_balance_loss(
    log_p_f_terms: Sequence[torch.Tensor],
    log_p_b_terms: Sequence[torch.Tensor],
    reward_direction_terms: Sequence[torch.Tensor],
    beta: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Discrete gradient-informed detailed balance residual:
        (log P_F - log P_B - beta * Delta(log R))^2
    """
    if not log_p_f_terms:
        raise ValueError("Expected at least one transition for nabla detailed balance loss.")
    if not (len(log_p_f_terms) == len(log_p_b_terms) == len(reward_direction_terms)):
        raise ValueError("log_p_f_terms, log_p_b_terms and reward_direction_terms must have same length.")

    losses = []
    for log_pf, log_pb, reward_dir in zip(log_p_f_terms, log_p_b_terms, reward_direction_terms):
        target = _as_scalar_tensor(reward_dir, device=log_pf.device, dtype=log_pf.dtype)
        losses.append((log_pf - log_pb - float(beta) * target).pow(2))

    stacked = torch.stack(losses)
    if reduction == "sum":
        return stacked.sum()
    if reduction == "mean":
        return stacked.mean()
    raise ValueError("Unsupported reduction '{}'. Use 'mean' or 'sum'.".format(reduction))


def combined_gafn_nabla_loss(
    gafn_loss: torch.Tensor,
    nabla_loss: torch.Tensor,
    lambda_gafn: float = 1.0,
    lambda_nabla: float = 1.0,
) -> torch.Tensor:
    return float(lambda_gafn) * gafn_loss + float(lambda_nabla) * nabla_loss

