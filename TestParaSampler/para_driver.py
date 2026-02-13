import argparse
import pickle
import time

import torch
from openfermion.linalg import get_ground_state, get_sparse_operator
from openfermion.utils import count_qubits

from gflow_vqe.hamiltonians import *
from gflow_vqe.para_utilities import run_parallel_graph_model_custom_reward_training
from gflow_vqe.result_analysis import (
    check_sampled_graphs_fci_plot,
    histogram_all_fci,
    plot_loss_curve,
)
from gflow_vqe.training import set_training_device
from gflow_vqe.utils import BinaryHamiltonian, FC_CompMatrix, get_terms, obj_to_comp_graph


def _build_problem(molecule_name: str):
    molecule_fn = globals().get(molecule_name)
    if molecule_fn is None:
        raise ValueError(
            "Unknown molecule '{}'. Expected one of the molecule functions in gflow_vqe.hamiltonians.".format(
                molecule_name
            )
        )

    mol, H, _, n_paulis, Hq = molecule_fn()
    sparse_hamiltonian = get_sparse_operator(Hq)
    energy, fci_wfn = get_ground_state(sparse_hamiltonian)
    n_q = count_qubits(Hq)

    binary_h = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
    terms = get_terms(binary_h)
    comp_matrix = FC_CompMatrix(terms)
    graph = obj_to_comp_graph(terms, comp_matrix)
    n_terms = graph.number_of_nodes()

    return {
        "molecule": mol,
        "graph": graph,
        "n_terms": n_terms,
        "fci_wfn": fci_wfn,
        "n_q": n_q,
        "energy": energy,
        "n_paulis": n_paulis,
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Parallel single-model GFlowNet training with custom reward for "
            "GIN/GAT/GraphTransformer, compatible with networkx/state_vector flows."
        )
    )
    parser.add_argument("molecule", type=str, help="Molecule function name (e.g., H2, H4, LiH, BeH2, N2).")
    parser.add_argument("--gpu", action="store_true", help="Use cuda:0 for model updates if available.")

    parser.add_argument(
        "--model",
        choices=["gin", "gat", "transformer", "gin_lite", "gat_lite", "transformer_lite"],
        default="gin_lite",
    )
    parser.add_argument("--representation", choices=["networkx", "state_vector"], default="state_vector")

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-updates", type=int, default=100)
    parser.add_argument("--episodes-per-worker", type=int, default=4)

    parser.add_argument("--n-hid-units", type=int, default=64)
    parser.add_argument("--n-emb-dim", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--replay-batch-size",
        type=int,
        default=64,
        help="Mini-batch size for replay/update in state_vector mode (0 = full batch).",
    )
    parser.add_argument("--seed", type=int, default=45)

    parser.add_argument("--l0", type=float, default=1000.0)
    parser.add_argument("--l1", type=float, default=0.0)

    parser.add_argument(
        "--fig-name",
        type=str,
        default=None,
        help="Output prefix. If omitted, uses the molecule name.",
    )
    parser.add_argument("--save-sampled", action="store_true", help="Save sampled graphs as pickle.")
    parser.add_argument(
        "--save-colorings",
        action="store_true",
        help="Save sampled colorings (state vectors) as pickle.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip plotting and analysis helpers.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    device = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")
    set_training_device(device)

    t0 = time.time()
    problem = _build_problem(args.molecule)

    expected_total = args.num_updates * args.num_workers * args.episodes_per_worker
    fig_name = args.fig_name if args.fig_name else args.molecule

    print("Training device={}".format(device))
    print("Molecule={}".format(args.molecule))
    print("Energy={}".format(problem["energy"]))
    print("Number of Qubits={}".format(problem["n_q"]))
    print("Number of Pauli products to measure={}".format(problem["n_paulis"]))
    print("Number of terms in the Hamiltonian={}".format(problem["n_terms"]))

    print("Parallel hyperparameters:")
    print("    + model={}".format(args.model))
    print("    + representation={}".format(args.representation))
    print("    + n_hid_units={}".format(args.n_hid_units))
    print("    + n_emb_dim={}".format(args.n_emb_dim))
    print("    + learning_rate={}".format(args.learning_rate))
    print("    + replay_batch_size={}".format(args.replay_batch_size))
    print("    + seed={}".format(args.seed))
    print("    + num_workers={}".format(args.num_workers))
    print("    + num_updates={}".format(args.num_updates))
    print("    + episodes_per_worker={}".format(args.episodes_per_worker))
    print("    + l0={}".format(args.l0))
    print("    + l1={}".format(args.l1))
    print("Expected total sampled graphs={}".format(expected_total))
    if args.num_workers > 1 and args.episodes_per_worker < 8:
        print(
            "Note: low episodes_per_worker increases model-sync overhead. "
            "For same sample count, try fewer updates and larger episodes_per_worker."
        )

    result = run_parallel_graph_model_custom_reward_training(
        graph=problem["graph"],
        n_terms=problem["n_terms"],
        n_hid_units=args.n_hid_units,
        n_emb=args.n_emb_dim,
        num_updates=args.num_updates,
        episodes_per_worker=args.episodes_per_worker,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        seed=args.seed,
        wfn=problem["fci_wfn"],
        n_q=problem["n_q"],
        l0=args.l0,
        l1=args.l1,
        model_name=args.model,
        representation=args.representation,
        gpu=args.gpu,
        fig_name=fig_name,
        replay_batch_size=args.replay_batch_size,
        show_progress=True,
    )

    t1 = time.time()
    print("Used multiprocessing={}".format(result.used_multiprocessing))
    print("Actual total sampled graphs={}".format(result.actual_total_samples))
    print("Expected total sampled graphs={}".format(result.expected_total_samples))
    print("Training time: {:.2f} seconds".format(t1 - t0))

    sampled_graphs = result.sampled_graphs

    if args.save_sampled:
        sampled_graphs_file = fig_name + "_sampled_graphs.p"
        with open(sampled_graphs_file, "wb") as f:
            pickle.dump(sampled_graphs, f, pickle.HIGHEST_PROTOCOL)
        print("Saved sampled graphs: {}".format(sampled_graphs_file))

    if args.save_colorings:
        sampled_colorings_file = fig_name + "_sampled_colorings.p"
        with open(sampled_colorings_file, "wb") as f:
            pickle.dump(result.sampled_colorings, f, pickle.HIGHEST_PROTOCOL)
        print("Saved sampled colorings: {}".format(sampled_colorings_file))

    if not args.skip_analysis:
        if len(sampled_graphs) < 16:
            print(
                "Skipping graph analysis plots because sampled_graphs has {} items (<16).".format(
                    len(sampled_graphs)
                )
            )
        else:
            check_sampled_graphs_fci_plot(fig_name, sampled_graphs, problem["fci_wfn"], problem["n_q"])
            plot_loss_curve(fig_name, result.losses, title="Loss over Parallel Training Updates")
            histogram_all_fci(fig_name, sampled_graphs, problem["fci_wfn"], problem["n_q"])


if __name__ == "__main__":
    main()
