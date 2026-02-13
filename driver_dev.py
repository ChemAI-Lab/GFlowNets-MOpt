import argparse
import csv
import pickle
import time

import networkx as nx
import numpy as np
import torch
from openfermion.linalg import get_ground_state, get_sparse_operator
from openfermion.utils import count_qubits
from tequila.grouping.binary_rep import BinaryHamiltonian

from gflow_vqe.advanced_training import (
    run_graph_model_covariance_objective_training_state_vector,
)
from gflow_vqe.covariance_rewards import (
    CovarianceRewardEngine,
    build_covariance_reward_data,
)
from gflow_vqe.hamiltonians import *
from gflow_vqe.result_analysis import histogram_all_fci, plot_loss_curve
from gflow_vqe.training import set_training_device, state_vector_colorings_to_graphs
from gflow_vqe.utils import *


MOLECULES = {
    "H2": H2,
    "H4": H4,
    "H6": H6,
    "LiH": LiH,
    "BeH2": BeH2,
    "H2O": H2O,
    "N2": N2,
    "NH3": NH3,
    "H2bk": H2bk,
    "H4bk": H4bk,
    "H6bk": H6bk,
    "LiHbk": LiHbk,
    "BeH2bk": BeH2bk,
    "H2Obk": H2Obk,
    "N2bk": N2bk,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Development driver for covariance-based GFlowNets "
            "(GAFN and nabla-DB)."
        )
    )
    parser.add_argument("molecule", choices=sorted(MOLECULES.keys()))
    parser.add_argument(
        "--objective",
        choices=["gafn", "nabla_db", "all"],
        default="all",
    )
    parser.add_argument("--model", choices=["gin", "gat", "transformer"], default="gin")
    parser.add_argument("--gpu", action="store_true", help="Use cuda:0 when available.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--hid-units", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--update-freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--emb-dim", type=int, default=2)
    parser.add_argument("--l0", type=float, default=1000.0)
    parser.add_argument("--l1", type=float, default=0.0)
    parser.add_argument("--alpha-edge", type=float, default=1.0)
    parser.add_argument("--alpha-terminal", type=float, default=1.0)
    parser.add_argument("--beta-nabla", type=float, default=1.0)
    parser.add_argument("--prefix", type=str, default="", help="Output prefix. Defaults to molecule name.")
    parser.add_argument("--plot", action="store_true", help="Generate loss and histogram plots per run.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def summarize_run(
    sampled_colorings,
    sampled_graphs,
    losses,
    reward_engine,
    wfn,
    n_q,
    l0,
    l1,
    train_time_s,
):
    if len(sampled_graphs) == 0:
        return {
            "num_samples": 0,
            "num_unique_colorings": 0,
            "valid_fraction": 0.0,
            "mean_cov_reward": 0.0,
            "best_cov_reward": 0.0,
            "best_exact_reward": 0.0,
            "best_eps2m": float("inf"),
            "best_num_groups": -1,
            "last_loss": float("nan"),
            "train_time_s": train_time_s,
        }

    cov_rewards = [reward_engine.reward(c) for c in sampled_colorings]
    mean_cov_reward = float(np.mean(cov_rewards))
    best_idx = int(np.argmax(cov_rewards))

    best_graph = sampled_graphs[best_idx]
    best_exact_reward = float(custom_reward(best_graph, wfn, n_q, l0, l1))
    best_eps2m = float(get_groups_measurement(best_graph, wfn, n_q))
    best_num_groups = int(max_color(best_graph))

    valid_fraction = float(sum(color_reward(g) > 0 for g in sampled_graphs) / len(sampled_graphs))
    unique_colorings = len({tuple(int(v) for v in c.tolist()) for c in sampled_colorings})

    return {
        "num_samples": len(sampled_graphs),
        "num_unique_colorings": unique_colorings,
        "valid_fraction": valid_fraction,
        "mean_cov_reward": mean_cov_reward,
        "best_cov_reward": float(np.max(cov_rewards)),
        "best_exact_reward": best_exact_reward,
        "best_eps2m": best_eps2m,
        "best_num_groups": best_num_groups,
        "last_loss": float(losses[-1]) if len(losses) > 0 else float("nan"),
        "train_time_s": train_time_s,
    }


def main():
    args = parse_args()

    device = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")
    set_training_device(device)
    print("Training device={}".format(device))

    t0 = time.time()
    molecule_fn = MOLECULES[args.molecule]
    mol, H, Hferm, n_paulis, Hq = molecule_fn()
    print("Number of Pauli products to measure: {}".format(n_paulis))

    sparse_hamiltonian = get_sparse_operator(Hq)
    energy, fci_wfn = get_ground_state(sparse_hamiltonian)
    n_q = count_qubits(Hq)
    print("Energy={}".format(energy))
    print("Number of Qubits={}".format(n_q))

    binary_H = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
    terms = get_terms(binary_H)
    comp_matrix = FC_CompMatrix(terms)
    Gc = obj_to_comp_graph(terms, comp_matrix)
    n_terms = nx.number_of_nodes(Gc)
    print("Number of terms in the Hamiltonian: {}".format(n_terms))

    objectives = (
        ["gafn", "nabla_db"]
        if args.objective == "all"
        else [args.objective]
    )
    fig_prefix = args.prefix if args.prefix else args.molecule

    print("For all experiments, hyperparameters are:")
    print("    + model={}".format(args.model))
    print("    + objectives={}".format(objectives))
    print("    + n_hid_units={}".format(args.hid_units))
    print("    + n_episodes={}".format(args.episodes))
    print("    + learning_rate={}".format(args.lr))
    print("    + update_freq={}".format(args.update_freq))
    print("    + seed={}".format(args.seed))
    print("    + n_emb_dim={}".format(args.emb_dim))
    print("    + l0={}, l1={}".format(args.l0, args.l1))
    print("    + alpha_edge={}, alpha_terminal={}".format(args.alpha_edge, args.alpha_terminal))
    print("    + beta_nabla={}".format(args.beta_nabla))

    cov_t0 = time.time()
    covariance_data = build_covariance_reward_data(Gc, fci_wfn, n_q)
    cov_build_time = time.time() - cov_t0
    print("Covariance dictionary precomputed in {:.2f} seconds.".format(cov_build_time))

    summary_rows = []

    for objective in objectives:
        run_name = "{}_{}_{}".format(fig_prefix, args.model, objective)
        print("\n=== Running objective: {} | output prefix: {} ===".format(objective, run_name))

        train_t0 = time.time()
        sampled_colorings, losses = run_graph_model_covariance_objective_training_state_vector(
            graph=Gc,
            n_terms=n_terms,
            n_hid_units=args.hid_units,
            n_episodes=args.episodes,
            learning_rate=args.lr,
            update_freq=args.update_freq,
            seed=args.seed,
            fig_name=run_name,
            n_emb=args.emb_dim,
            l0=args.l0,
            l1=args.l1,
            objective=objective,
            model_name=args.model,
            covariance_data=covariance_data,
            alpha_edge=args.alpha_edge,
            alpha_terminal=args.alpha_terminal,
            beta_nabla=args.beta_nabla,
            show_progress=not args.no_progress,
        )
        train_time_s = time.time() - train_t0

        sampled_graphs = state_vector_colorings_to_graphs(Gc, sampled_colorings)
        reward_engine = CovarianceRewardEngine(Gc, covariance_data, args.l0, args.l1)
        stats = summarize_run(
            sampled_colorings=sampled_colorings,
            sampled_graphs=sampled_graphs,
            losses=losses,
            reward_engine=reward_engine,
            wfn=fci_wfn,
            n_q=n_q,
            l0=args.l0,
            l1=args.l1,
            train_time_s=train_time_s,
        )

        with open(run_name + "_sampled_graphs.p", "wb") as f:
            pickle.dump(sampled_graphs, f, pickle.HIGHEST_PROTOCOL)
        print("Saved sampled graphs to {}".format(run_name + "_sampled_graphs.p"))

        if args.plot:
            plot_loss_curve(run_name, losses, title="Loss - {}".format(objective))
            valid_count = sum(color_reward(g) > 0 for g in sampled_graphs)
            if valid_count > 0:
                histogram_all_fci(run_name, sampled_graphs, fci_wfn, n_q)
                print("Saved plots for {}".format(run_name))
            else:
                print("Skipped histogram for {} (no valid sampled graphs).".format(run_name))

        print(
            "Summary [{}]: train_time={:.2f}s, valid_fraction={:.3f}, best_cov_reward={:.6f}, best_eps2m={:.6f}, best_groups={}".format(
                objective,
                stats["train_time_s"],
                stats["valid_fraction"],
                stats["best_cov_reward"],
                stats["best_eps2m"],
                stats["best_num_groups"],
            )
        )

        row = {
            "molecule": args.molecule,
            "model": args.model,
            "objective": objective,
            "num_samples": stats["num_samples"],
            "num_unique_colorings": stats["num_unique_colorings"],
            "valid_fraction": stats["valid_fraction"],
            "mean_cov_reward": stats["mean_cov_reward"],
            "best_cov_reward": stats["best_cov_reward"],
            "best_exact_reward": stats["best_exact_reward"],
            "best_eps2m": stats["best_eps2m"],
            "best_num_groups": stats["best_num_groups"],
            "last_loss": stats["last_loss"],
            "train_time_s": stats["train_time_s"],
        }
        summary_rows.append(row)

    summary_path = "{}_{}_dev_summary.csv".format(fig_prefix, args.model)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nSaved summary CSV: {}".format(summary_path))
    print("\n=== Comparison (sorted by best_exact_reward, descending) ===")
    for row in sorted(summary_rows, key=lambda r: r["best_exact_reward"], reverse=True):
        print(
            "{} | best_exact_reward={:.6f} | best_eps2m={:.6f} | best_groups={} | valid_fraction={:.3f} | train_time={:.2f}s".format(
                row["objective"],
                row["best_exact_reward"],
                row["best_eps2m"],
                row["best_num_groups"],
                row["valid_fraction"],
                row["train_time_s"],
            )
        )

    t1 = time.time()
    print("\nTotal elapsed time: {:.2f} seconds".format(t1 - t0))


if __name__ == "__main__":
    main()
