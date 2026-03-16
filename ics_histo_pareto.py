import argparse
import os
import pickle

import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from openfermion import get_sparse_operator
from tequila.grouping.binary_rep import BinaryHamiltonian

import gflow_vqe.hamiltonians as hamlib
from gflow_vqe.circuit_helpers import grouping_circuit_stats_tequila
from gflow_vqe.overlapping_helpers import (
    as_tequila_wavefunction,
    get_opt_sample_size,
    iterative_coefficient_splitting_from_gflow_grouping,
    iterative_coefficient_splitting_from_groups,
    prepare_cov_dict,
)
from gflow_vqe.utils import color_reward, get_variance_wavefunction


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value, got '{}'.".format(value))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Pareto plot by epsilon^2 M against number of groups or 2-qubit gates, "
            "with sorted-insertion and optional ICS overlays."
        )
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
        help=(
            "Wavefunction used for SI/ICS evaluation. This should match the wavefunction "
            "used to generate the cached metrics file (default: FCI)."
        ),
    )
    parser.add_argument(
        "--y-axis",
        dest="y_axis",
        type=str,
        default="groups",
        choices=("groups", "two-qubit"),
        help="Metric to plot on the y-axis. 'groups' for number of groups, 'two-qubit' for N2q.",
    )
    parser.add_argument(
        "--ics",
        action="store_true",
        help="Also compute and plot ICS results starting from each Pareto-front GFlowNet graph and from SI.",
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const=True,
        default=False,
        type=str_to_bool,
        help="Save the best n sampled graphs according to a custom reward (default: False).",
    )
    parser.add_argument(
        "--n_save",
        type=int,
        default=10,
        help="Number of top graphs to save when --save is enabled (default: 10).",
    )
    parser.add_argument(
        "--l0",
        type=float,
        default=None,
        help="Custom reward coefficient for 1/eps^2M. Required only when --save is enabled.",
    )
    parser.add_argument(
        "--l1",
        type=float,
        default=None,
        help="Custom reward coefficient for the color reward. Required only when --save is enabled.",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=None,
        help="Custom reward coefficient for 1/N_{2q}. Required only when --save is enabled.",
    )
    args = parser.parse_args(argv)
    args.func = getattr(hamlib, args.func_name, None)
    if args.func is None:
        raise ValueError("Unknown molecule '{}'".format(args.func_name))
    if args.save and args.n_save < 1:
        parser.error("--n_save must be at least 1.")
    if args.save and any(value is None for value in (args.l0, args.l1, args.l2)):
        parser.error("--l0, --l1, and --l2 are required when --save is enabled.")
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


def load_sampled_graphs(sampled_graphs_path):
    if not os.path.exists(sampled_graphs_path):
        raise FileNotFoundError("Could not find sampled graphs file '{}'.".format(sampled_graphs_path))

    with open(sampled_graphs_path, "rb") as handle:
        sampled_graphs = pickle.load(handle)

    if not isinstance(sampled_graphs, list):
        sampled_graphs = list(sampled_graphs)

    valid_graphs = [graph for graph in sampled_graphs if color_reward(graph) > 0]
    if not valid_graphs:
        raise RuntimeError("No valid sampled graphs were found in '{}'.".format(sampled_graphs_path))
    return valid_graphs


def load_cached_metrics(metrics_path, expected_length):
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            "Could not find metrics file '{}'. Run metrics_histo_pareto.py first.".format(metrics_path)
        )

    with open(metrics_path, "rb") as handle:
        cached = pickle.load(handle)

    required_keys = ("measurements", "num_groups", "two_qubit_gates")
    if not isinstance(cached, dict) or any(key not in cached for key in required_keys):
        raise ValueError("Metrics file '{}' does not match the expected cache format.".format(metrics_path))

    metrics = {key: np.asarray(cached[key]) for key in required_keys}
    if metrics["measurements"].shape[0] != expected_length:
        raise ValueError(
            "Metrics file '{}' contains {} entries, but {} valid sampled graphs were loaded.".format(
                metrics_path,
                metrics["measurements"].shape[0],
                expected_length,
            )
        )
    return metrics


def _to_real_if_close(value, tiny=1e-12):
    if hasattr(value, "imag") and abs(value.imag) < tiny:
        return float(value.real)
    return value


def optimal_allocation_metric(commuting_parts, suggested_sample_size, wfn, tiny=1e-12):
    measurement_metric = 0.0
    wf = as_tequila_wavefunction(wfn)

    for idx, part in enumerate(commuting_parts):
        op = part.to_qubit_hamiltonian()
        var_part = wf.inner((op * op)(wf)) - wf.inner(op(wf)) ** 2
        if hasattr(var_part, "imag") and abs(var_part.imag) < tiny:
            var_part = var_part.real
        measurement_metric += var_part / suggested_sample_size[idx]

    return float(_to_real_if_close(measurement_metric, tiny=tiny))


def is_identity_term(term):
    return not np.any(term.get_binary())


def normalize_binary_groups(groups):
    normalized_groups = []
    for group in groups:
        terms = list(group.binary_terms) if isinstance(group, BinaryHamiltonian) else list(group)
        terms = [term for term in terms if not is_identity_term(term)]
        if terms:
            normalized_groups.append(BinaryHamiltonian(terms))
    if not normalized_groups:
        raise ValueError("No measurable terms were found in the provided groups.")
    return normalized_groups


def compute_group_metrics(groups, cov_dict, wfn):
    normalized_groups = normalize_binary_groups(groups)
    sample_size = get_opt_sample_size([group.binary_terms for group in normalized_groups], cov_dict)
    measurement = optimal_allocation_metric(normalized_groups, sample_size, wfn)
    group_mapping = {idx: list(group.binary_terms) for idx, group in enumerate(normalized_groups)}
    two_qubit_gates = int(grouping_circuit_stats_tequila(group_mapping).total_two_qubit_gates)
    return {
        "measurement": measurement,
        "num_groups": len(normalized_groups),
        "two_qubit_gates": two_qubit_gates,
        "sample_size": sample_size,
        "groups": normalized_groups,
    }


def metric_point(metric_dict, y_axis):
    if y_axis == "groups":
        return float(metric_dict["measurement"]), float(metric_dict["num_groups"])
    return float(metric_dict["measurement"]), float(metric_dict["two_qubit_gates"])


def format_metric_triplet(metric_dict):
    return "eps^2M(x)={:.10g}, N_G(x)={}, N_{{2q}}(x)={}".format(
        float(metric_dict["measurement"]),
        int(metric_dict["num_groups"]),
        int(metric_dict["two_qubit_gates"]),
    )


def plot_marginal(values, ax, axis, color):
    values = np.asarray(values, dtype=float)
    if values.size > 1 and np.unique(values).size > 1:
        if axis == "x":
            sns.kdeplot(x=values, ax=ax, fill=True, color=color, warn_singular=False)
        else:
            sns.kdeplot(y=values, ax=ax, fill=True, color=color, warn_singular=False)
        return

    if axis == "x":
        ax.hist(values, bins=1, color=color, alpha=0.4)
    else:
        ax.hist(values, bins=1, color=color, alpha=0.4, orientation="horizontal")


def padded_limits(values, pad_fraction=0.05, min_pad=0.05):
    values = np.asarray(values, dtype=float)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        pad = max(min_pad, 0.05 * max(1.0, abs(vmin)))
        return vmin - pad, vmax + pad
    pad = max(min_pad, pad_fraction * (vmax - vmin))
    return vmin - pad, vmax + pad


def custom_reward_from_cached_metrics(graph, measurement, num_groups, two_qubit_gates, l0, l1, l2):
    reward = 0.0

    if l0 != 0:
        if measurement == 0:
            raise ZeroDivisionError("Encountered eps^2M(x)=0 while computing the custom reward.")
        reward += l0 / measurement

    if l1 != 0:
        reward += l1 * (graph.number_of_nodes() - num_groups)

    if l2 != 0:
        if two_qubit_gates == 0:
            raise ZeroDivisionError("Encountered N_{2q}(x)=0 while computing the custom reward.")
        reward += l2 / two_qubit_gates

    return float(reward)


def save_top_graphs(fig_name, sampled_graphs, metrics, n_save, l0, l1, l2):
    scored_graphs = []
    for idx, graph in enumerate(sampled_graphs):
        reward = custom_reward_from_cached_metrics(
            graph,
            float(metrics["measurements"][idx]),
            int(metrics["num_groups"][idx]),
            int(metrics["two_qubit_gates"][idx]),
            l0,
            l1,
            l2,
        )
        scored_graphs.append((reward, idx))

    scored_graphs.sort(key=lambda item: item[0], reverse=True)
    n_to_save = min(n_save, len(scored_graphs))
    top_indices = [idx for _, idx in scored_graphs[:n_to_save]]
    output_path = "{}_top_{}_custom_reward_l0_{}_l1_{}_l2_{}.p".format(
        fig_name,
        n_to_save,
        "{:g}".format(l0),
        "{:g}".format(l1),
        "{:g}".format(l2),
    )

    with open(output_path, "wb") as handle:
        pickle.dump([sampled_graphs[idx] for idx in top_indices], handle, pickle.HIGHEST_PROTOCOL)

    print("Saved top {} graphs by custom reward to {}".format(n_to_save, output_path))
    for rank, (reward, idx) in enumerate(scored_graphs[:n_to_save], start=1):
        metric_dict = {
            "measurement": float(metrics["measurements"][idx]),
            "num_groups": int(metrics["num_groups"][idx]),
            "two_qubit_gates": int(metrics["two_qubit_gates"][idx]),
        }
        print("  [{}] reward={:.10g}, {}".format(rank, reward, format_metric_triplet(metric_dict)))


def make_output_path(fig_name, y_axis):
    if y_axis == "groups":
        return fig_name + "_ics_pareto_joint_all.svg"
    return fig_name + "_ics_pareto_joint_all_2qubit.svg"


def main(argv=None):
    args = parse_args(argv)
    fig_name = args.func_name
    sampled_graphs_path = fig_name + "_sampled_graphs.p"
    metrics_path = fig_name + "_sampled_graphs_metrics.p"

    sampled_graphs = load_sampled_graphs(sampled_graphs_path)
    metrics = load_cached_metrics(metrics_path, len(sampled_graphs))
    output_path = make_output_path(fig_name, args.y_axis)

    print("Loaded {} valid sampled graphs from {}".format(len(sampled_graphs), sampled_graphs_path))
    print("Loaded metrics from {}".format(metrics_path))

    if args.save:
        save_top_graphs(fig_name, sampled_graphs, metrics, args.n_save, args.l0, args.l1, args.l2)

    mol, H, _, n_paulis, Hq = args.func()
    print("Number of Pauli products to measure: {}".format(n_paulis))

    sparse_hamiltonian = get_sparse_operator(Hq)
    _, variance_wfn = get_variance_wavefunction(mol, Hq, method=args.wfn, sparse_hamiltonian=sparse_hamiltonian)
    binary_hamiltonian = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
    cov_dict = prepare_cov_dict(binary_hamiltonian, variance_wfn)

    si_groups, _ = binary_hamiltonian.commuting_groups(
        options={"method": "si", "condition": "fc", "cov_dict": cov_dict}
    )
    si_metrics = compute_group_metrics(si_groups, cov_dict, variance_wfn)

    points = np.column_stack((metrics["measurements"].astype(float), metrics["num_groups"].astype(float)))
    if args.y_axis == "two-qubit":
        points[:, 1] = metrics["two_qubit_gates"].astype(float)

    x = points[:, 0]
    mask = pareto_front_min(points)
    pareto_indices = np.flatnonzero(mask)
    pareto_indices = pareto_indices[np.argsort(points[pareto_indices, 0])]
    pareto_sorted = points[pareto_indices]

    print("")
    print("Sorted insertion:")
    print("  Before ICS: {}".format(format_metric_triplet(si_metrics)))

    si_ics_metrics = None
    pareto_ics_results = []
    if args.ics:
        si_ics_groups, _ = iterative_coefficient_splitting_from_groups(si_groups, cov_dict, condition="fc")
        si_ics_metrics = compute_group_metrics(si_ics_groups, cov_dict, variance_wfn)

        print("  After  ICS: {}".format(format_metric_triplet(si_ics_metrics)))
        print("")
        print("Pareto-front GFlowNet graphs:")
        for display_idx, graph_idx in enumerate(pareto_indices):
            before_metrics = {
                "measurement": float(metrics["measurements"][graph_idx]),
                "num_groups": int(metrics["num_groups"][graph_idx]),
                "two_qubit_gates": int(metrics["two_qubit_gates"][graph_idx]),
            }
            gflow_ics_groups, _ = iterative_coefficient_splitting_from_gflow_grouping(
                binary_hamiltonian,
                sampled_graphs[graph_idx],
                cov_dict,
                condition="fc",
            )
            after_metrics = compute_group_metrics(gflow_ics_groups, cov_dict, variance_wfn)
            pareto_ics_results.append(
                {
                    "index": graph_idx,
                    "before": before_metrics,
                    "after": after_metrics,
                }
            )
            print("  [{}] Before ICS: {}".format(display_idx, format_metric_triplet(before_metrics)))
            print("      After  ICS: {}".format(format_metric_triplet(after_metrics)))

    print("")
    print("Number of Pareto-front graphs: {}".format(len(pareto_indices)))

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
            zorder=3,
        )

    si_x, si_y = metric_point(si_metrics, args.y_axis)
    g.ax_joint.plot(
        si_x,
        si_y,
        marker="D",
        color="red",
        markersize=7,
        linestyle="None",
        label="SI",
        zorder=5,
    )

    point_label_used = False
    if args.ics and si_ics_metrics is not None:
        si_ics_x, si_ics_y = metric_point(si_ics_metrics, args.y_axis)
        g.ax_joint.plot(
            [si_x, si_ics_x],
            [si_y, si_ics_y],
            color="black",
            linestyle=":",
            linewidth=1.2,
            label=None,
            zorder=1,
        )
        g.ax_joint.plot(
            si_ics_x,
            si_ics_y,
            marker="*",
            color="#cc7a00",
            markersize=10,
            linestyle="None",
            label="SI-ICS",
            zorder=5,
        )

        for result in pareto_ics_results:
            before_x, before_y = metric_point(result["before"], args.y_axis)
            after_x, after_y = metric_point(result["after"], args.y_axis)
            g.ax_joint.plot(
                [before_x, after_x],
                [before_y, after_y],
                color="black",
                linestyle=":",
                linewidth=2.0,
                label=None,
                zorder=1,
            )
            g.ax_joint.plot(
                after_x,
                after_y,
                marker="^",
                color="#4b2e83",
                markersize=8,
                linestyle="None",
                label=None if point_label_used else "GFN-ICS",
                zorder=5,
            )
            point_label_used = True

    if args.y_axis == "groups":
        y_label = "$N_G(x)$"
    else:
        y_label = "$N_{2q}(x)$"
    g.set_axis_labels("$\\epsilon^2M(x)$", y_label, fontsize=14)

    plot_marginal(x, g.ax_marg_x, axis="x", color="purple")
    plot_marginal(points[:, 1], g.ax_marg_y, axis="y", color="green")
    g.ax_marg_x.set_ylabel("")
    g.ax_marg_y.set_xlabel("")
    g.ax_marg_x.tick_params(axis="x", labelbottom=False)
    g.ax_marg_y.tick_params(axis="y", labelleft=False)

    overlay_points = [metric_point(si_metrics, args.y_axis)]
    if args.ics and si_ics_metrics is not None:
        overlay_points.append(metric_point(si_ics_metrics, args.y_axis))
        overlay_points.extend(metric_point(result["after"], args.y_axis) for result in pareto_ics_results)

    all_x = np.concatenate((x, np.asarray([point[0] for point in overlay_points], dtype=float)))
    all_y = np.concatenate((points[:, 1], np.asarray([point[1] for point in overlay_points], dtype=float)))
    x_min, x_max = padded_limits(all_x, pad_fraction=0.06, min_pad=0.05)

    if args.y_axis == "groups":
        y_min = max(0.0, float(np.floor(np.min(all_y)) - 1.0))
        y_max = float(np.ceil(np.max(all_y)) + 1.0)
    else:
        y_min, y_max = padded_limits(all_y, pad_fraction=0.04, min_pad=0.5)

    g.ax_joint.set_xlim(x_min, x_max)
    g.ax_joint.set_ylim(y_min, y_max)
    g.ax_joint.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    g.ax_joint.legend(loc="best")
    g.figure.savefig(output_path, format="svg", dpi=600, bbox_inches="tight")
    print("Saved Pareto plot to {}".format(output_path))


if __name__ == "__main__":
    main()
