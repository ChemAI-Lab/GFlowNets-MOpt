import argparse
import os
import pickle

import numpy as np
import seaborn as sns
import matplotlib.ticker as mticker

from gflow_vqe.circuit_helpers import get_groups_2qgates
from gflow_vqe.hamiltonians import *
from gflow_vqe.gflow_utils import *
from gflow_vqe.result_analysis import *
from gflow_vqe.training import *
from gflow_vqe.utils import *
from openfermion import count_qubits
from openfermion import get_sparse_operator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pareto plot by epsilon^2 M against number of groups or 2-qubit gates."
    )
    parser.add_argument(
        "func_name",
        type=str,
        help="Molecule helper from gflow_vqe.hamiltonians (for example H2, LiH, BeH2, N2).",
    )
    parser.add_argument(
        "--wfn",
        type=lambda s: str(s).upper(),
        default="FCI",
        choices=("FCI", "HF", "CISD"),
        help="Wavefunction used for eps^2 M evaluation (default: FCI).",
    )
    parser.add_argument(
        "--y-axis",
        dest="y_axis",
        type=str,
        default="groups",
        choices=("groups", "two-qubit"),
        help="Metric to plot on the y-axis. 'groups' for number of groups, 'two-qubit' for N2q.",
    )
    args = parser.parse_args()
    args.func = globals().get(args.func_name)
    if args.func is None:
        raise ValueError("Unknown molecule '{}'".format(args.func_name))
    return args


def pareto_front_min(pts):
    if len(pts) == 0:
        return np.array([], dtype=bool)
    is_pareto = np.ones(pts.shape[0], dtype=bool)
    for i, p in enumerate(pts):
        if not is_pareto[i]:
            continue
        dominated = np.any(np.all(pts <= p, axis=1) & np.any(pts < p, axis=1))
        if dominated:
            is_pareto[i] = False
    return is_pareto


def build_metrics(sampled_graphs, wfn, n_qubits):
    measurements = []
    group_counts = []
    n2q_gates = []

    for graph in sampled_graphs:
        measurements.append(float(get_groups_measurement(graph, wfn, n_qubits)))
        group_counts.append(int(max_color(graph)))
        n2q_gates.append(int(get_groups_2qgates(graph)))

    return {
        "measurements": np.asarray(measurements, dtype=float),
        "num_groups": np.asarray(group_counts, dtype=int),
        "two_qubit_gates": np.asarray(n2q_gates, dtype=int),
    }


def load_or_build_metrics(metrics_path, sampled_graphs, wfn, n_qubits):
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "rb") as handle:
                cached = pickle.load(handle)
            if (
                isinstance(cached, dict)
                and all(key in cached for key in ("measurements", "num_groups", "two_qubit_gates"))
                and isinstance(cached["measurements"], np.ndarray)
                and cached["measurements"].shape[0] == len(sampled_graphs)
            ):
                print("Loaded metrics from {}".format(metrics_path))
                return cached
            print("Cached metrics file '{}' does not match expected format; rebuilding.".format(metrics_path))
        except Exception as error:
            print("Could not load metrics cache '{}': {}".format(metrics_path, error))

    metrics = build_metrics(sampled_graphs, wfn, n_qubits)
    with open(metrics_path, "wb") as handle:
        pickle.dump(metrics, handle, pickle.HIGHEST_PROTOCOL)
    print("Computed and saved metrics to {}".format(metrics_path))
    return metrics


def main():
    args = parse_args()
    fig_name = args.func_name
    mol, H, Hferm, n_paulis, Hq = args.func()
    print("Number of Pauli products to measure: {}".format(n_paulis))

    sparse_hamiltonian = get_sparse_operator(Hq)
    _, wfn = get_variance_wavefunction(mol, Hq, method=args.wfn, sparse_hamiltonian=sparse_hamiltonian)
    n_qubits = count_qubits(Hq)

    sampled_graphs_path = fig_name + "_sampled_graphs.p"
    metrics_path = fig_name + "_sampled_graphs_metrics.p"
    output_path = fig_name + "_pareto_joint_all.svg" if args.y_axis == "groups" else fig_name + "_pareto_joint_all_2qubit.svg"

    with open(sampled_graphs_path, "rb") as handle:
        sampled_graphs = pickle.load(handle)

    sampled_graphs = [g for g in sampled_graphs if color_reward(g) > 0]
    if len(sampled_graphs) == 0:
        raise RuntimeError("No valid sampled graphs were found in '{}'.".format(sampled_graphs_path))

    print("Number of valid graphs in file: {}".format(len(sampled_graphs)))
    metrics = load_or_build_metrics(metrics_path, sampled_graphs, wfn, n_qubits)

    if args.y_axis == "groups":
        y = metrics["num_groups"]
        y_label = "$N_G(x)$"
        y_ticks_integer = True
        y_min, y_max = 0.0, max(1.0, float(np.max(y) + 1.0))
    else:
        y = metrics["two_qubit_gates"]
        y_label = "$N_{2q}(x)$"
        y_ticks_integer = True
        y_min, y_max = float(np.min(y)), float(np.max(y))
        if y_max < y_min:
            y_min, y_max = 0.0, 1.0
        if y_min == y_max:
            y_max = y_min + 1.0

    points = np.column_stack((metrics["measurements"], y.astype(float)))

    x = points[:, 0]
    mask = pareto_front_min(points)
    pareto_sorted = points[mask][np.argsort(points[mask][:, 0])]

    sns.set_theme(style="whitegrid")
    g = sns.JointGrid(x=x, y=points[:, 1], height=7.5, space=0)

    sns.scatterplot(x=x, y=points[:, 1], ax=g.ax_joint, alpha=0.5, s=30, edgecolor=None)
    if len(pareto_sorted) > 0:
        g.ax_joint.plot(
            pareto_sorted[:, 0],
            pareto_sorted[:, 1],
            color="orange",
            marker="o",
            markersize=8,
            linewidth=1.5,
            label="Pareto front",
        )

    g.set_axis_labels("$\\epsilon^2M(x)$", y_label, fontsize=14)
    sns.kdeplot(x=x, ax=g.ax_marg_x, fill=True, color="purple")
    sns.kdeplot(y=points[:, 1], ax=g.ax_marg_y, fill=True, color="green")
    g.ax_marg_x.set_ylabel("")
    g.ax_marg_y.set_xlabel("")
    g.ax_marg_x.tick_params(axis="x", labelbottom=False)
    g.ax_marg_y.tick_params(axis="y", labelleft=False)

    g.ax_joint.set_xlim(0.55, 1.8)
    if args.y_axis == "groups":
        g.ax_joint.set_ylim(9, 30)
    else:
        g.ax_joint.set_ylim(max(0.0, y_min - 0.5), y_max + 0.5)
    if y_ticks_integer:
        g.ax_joint.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    g.ax_joint.legend(loc="best")
    g.figure.savefig(output_path, format="svg", dpi=600, bbox_inches="tight")
    print("Saved Pareto plot to {}".format(output_path))


if __name__ == "__main__":
    main()
