"""Quick SI/ICS analysis for one coefficient-ordered GFlowNet graph.

The preferred input is a saved top-graphs pickle.  When no such file is
available, the graph with the smallest cached ``eps^2 M`` value is selected
from the valid entries in ``*_sampled_graphs.p`` and
``*_sampled_graphs_metrics.p``.  Only that graph, standard sorted insertion
(SI), and SI initialized ICS are evaluated.

The numerical backend is shared with ``ics_histo_pareto_fast_o_v2.py`` so this
driver evaluates only the covariance moments required by those two ICS
initializations and preserves the ordered graph's embedded-Pauli mapping.
"""

import argparse
import glob
import hashlib
import os
import pickle
import time

import numpy as np
from openfermion import get_sparse_operator
from openfermion.utils import count_qubits
from tequila.grouping.binary_rep import BinaryHamiltonian
from threadpoolctl import threadpool_limits

import gflow_vqe.hamiltonians as hamlib
from gflow_vqe.overlapping_helpers import OverlappingAuxiliary, get_opt_sample_size
from gflow_vqe.utils import get_variance_wavefunction

if __package__:
    from . import ics_histo_pareto_fast_o_v2 as fast_ics
else:
    import ics_histo_pareto_fast_o_v2 as fast_ics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run SI, SI-ICS, and ICS for only the best saved coefficient-ordered "
            "GFlowNet graph."
        )
    )
    parser.add_argument(
        "func_name",
        help="Molecule helper from gflow_vqe.hamiltonians (for example H2, LiH, BeH2, N2).",
    )
    parser.add_argument(
        "--wfn",
        type=lambda value: str(value).upper(),
        default="FCI",
        choices=("FCI", "HF", "CISD"),
        help=(
            "Wavefunction used for SI/ICS (default: FCI). When the metrics cache "
            "is used, this must match the wavefunction used to create that cache."
        ),
    )
    parser.add_argument(
        "--qwc",
        action="store_true",
        help="Use qubit-wise commuting groupings instead of fully commuting groupings.",
    )
    parser.add_argument(
        "--top-graphs",
        default=None,
        help=(
            "Saved top-graphs pickle. If omitted, a matching <molecule>_top_*.p "
            "file is discovered in the current working directory before falling "
            "back to sampled graphs."
        ),
    )
    parser.add_argument(
        "--sampled-graphs",
        default=None,
        help=(
            "Sampled-graphs pickle (default: <molecule>_sampled_graphs.p in the "
            "current working directory)."
        ),
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help=(
            "Cached metrics pickle (default: <molecule>_sampled_graphs_metrics.p "
            "in the current working directory)."
        ),
    )
    parser.add_argument(
        "--cov-workers",
        type=int,
        default=fast_ics.default_cov_workers(),
        help="Requested worker processes for selected Pauli moments (default: up to 8).",
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=1,
        help="BLAS threads used by each numerical worker (default: 1).",
    )
    parser.add_argument(
        "--cache-dir",
        default=fast_ics.DEFAULT_CACHE_DIRECTORY,
        help=(
            "Reusable moment and graph-ICS cache directory (default: "
            ".pareto_fast_cache in the current working directory)."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reading and writing reusable covariance/ICS caches.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore compatible covariance/ICS cache entries and recompute them.",
    )

    args = parser.parse_args(argv)
    if args.cov_workers < 1:
        parser.error("--cov-workers must be at least 1.")
    if args.blas_threads < 1:
        parser.error("--blas-threads must be at least 1.")
    args.func = getattr(hamlib, args.func_name, None)
    if args.func is None:
        parser.error("Unknown molecule helper '{}'.".format(args.func_name))
    return args


def discover_top_graphs_path(func_name, explicit_path=None, search_directory=None):
    """Return the preferred top-graphs path, or ``None`` when none exists."""

    search_directory = os.path.abspath(search_directory or os.getcwd())
    if explicit_path is not None:
        explicit_path = os.path.abspath(explicit_path)
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError(
                "Could not find explicitly requested top-graphs file '{}'.".format(
                    explicit_path
                )
            )
        return explicit_path

    conventional_path = os.path.join(
        search_directory,
        "{}_top_graphs.p".format(func_name),
    )
    if os.path.isfile(conventional_path):
        return conventional_path

    patterns = (
        os.path.join(
            search_directory,
            "{}_top_*_custom_reward_l0_*_l1_*_l2_*.p".format(func_name),
        ),
        os.path.join(search_directory, "{}_top_*.p".format(func_name)),
    )
    candidates = sorted(
        {
            path
            for pattern in patterns
            for path in glob.glob(pattern)
            if os.path.isfile(path)
        }
    )
    if len(candidates) > 1:
        raise RuntimeError(
            "Found multiple matching top-graphs files: {}. Select one with "
            "--top-graphs PATH.".format(", ".join(candidates))
        )
    return candidates[0] if candidates else None


def load_first_top_graph(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)

    if hasattr(payload, "nodes") and hasattr(payload, "edges"):
        return payload

    try:
        graphs = list(payload)
    except TypeError as error:
        raise ValueError(
            "Top-graphs file '{}' must contain a graph or an iterable of graphs.".format(
                path
            )
        ) from error
    if not graphs:
        raise ValueError("Top-graphs file '{}' is empty.".format(path))
    graph = graphs[0]
    if not hasattr(graph, "nodes") or not hasattr(graph, "edges"):
        raise ValueError(
            "The first entry in top-graphs file '{}' is not a graph.".format(path)
        )
    return graph


def validate_metrics(metrics, expected_length, path):
    validated = {}
    dtypes = {
        "measurements": float,
        "num_groups": int,
        "two_qubit_gates": int,
    }
    for key, dtype in dtypes.items():
        values = np.asarray(metrics[key], dtype=dtype)
        if values.ndim != 1 or len(values) != expected_length:
            raise ValueError(
                "Metrics entry '{}' in '{}' must be a one-dimensional array of "
                "length {}, got shape {}.".format(
                    key,
                    path,
                    expected_length,
                    values.shape,
                )
            )
        validated[key] = values

    if not np.all(np.isfinite(validated["measurements"])):
        raise ValueError("Metrics file '{}' contains non-finite measurements.".format(path))
    return validated


def select_graph_inputs(args):
    """Choose the graph using top-file precedence and return cached metadata."""

    run_directory = os.getcwd()
    top_graphs_path = discover_top_graphs_path(
        args.func_name,
        args.top_graphs,
        search_directory=run_directory,
    )
    if top_graphs_path is not None:
        return {
            "graph": load_first_top_graph(top_graphs_path),
            "source": "first-ranked graph in '{}'".format(top_graphs_path),
            "cached_before": None,
        }

    sampled_path = args.sampled_graphs or os.path.join(
        run_directory,
        "{}_sampled_graphs.p".format(args.func_name),
    )
    metrics_path = args.metrics or os.path.join(
        run_directory,
        "{}_sampled_graphs_metrics.p".format(args.func_name),
    )
    sampled_graphs = fast_ics.load_sampled_graphs(sampled_path)
    metrics = fast_ics.load_cached_metrics(metrics_path, len(sampled_graphs))
    metrics = validate_metrics(metrics, len(sampled_graphs), metrics_path)

    best_index = int(np.argmin(metrics["measurements"]))
    cached_before = {
        "measurement": float(metrics["measurements"][best_index]),
        "num_groups": int(metrics["num_groups"][best_index]),
        "two_qubit_gates": int(metrics["two_qubit_gates"][best_index]),
    }
    return {
        "graph": sampled_graphs[best_index],
        "source": (
            "valid graph index {} from '{}' (selected by '{}')".format(
                best_index,
                sampled_path,
                metrics_path,
            )
        ),
        "cached_before": cached_before,
    }


def cache_paths(args, grouping_condition, fingerprint):
    if args.no_cache:
        return None, None, None

    moment_path = os.path.join(
        args.cache_dir,
        "{}_{}_moments_v2.p".format(args.func_name, args.wfn.lower()),
    )
    graph_ics_path = os.path.join(
        args.cache_dir,
        "{}_{}_{}_quick_ics_metrics_v2.p".format(
            args.func_name,
            args.wfn.lower(),
            grouping_condition,
        ),
    )
    graph_ics_fingerprint = hashlib.sha256(
        "{}|{}|quick-ics|sparse-tequila-report-v1".format(
            fingerprint,
            grouping_condition,
        ).encode("ascii")
    ).hexdigest()
    return moment_path, graph_ics_path, graph_ics_fingerprint


def compute_quick_metrics(groups, covariances, reporter):
    """Return only the measurement quantities needed by this quick driver."""

    normalized_groups = fast_ics.normalize_binary_groups(groups)
    sample_size = get_opt_sample_size(
        [group.binary_terms for group in normalized_groups],
        covariances,
    )
    measurement = reporter.optimal_allocation_metric(
        normalized_groups,
        sample_size,
    )
    return {
        "measurement": float(measurement),
        "num_groups": len(normalized_groups),
    }


def compact_quick_metrics(metrics):
    return {
        "measurement": float(metrics["measurement"]),
        "num_groups": int(metrics["num_groups"]),
    }


def format_quick_metrics(metrics):
    return "eps^2M(x)={:.10g}, N_G(x)={}".format(
        float(metrics["measurement"]),
        int(metrics["num_groups"]),
    )


def optimize_plan(plan, covariances, reporter, blas_threads):
    with threadpool_limits(limits=blas_threads):
        groups = plan.overlapping.optimal_overlapping_groups(
            OverlappingAuxiliary(covariances)
        )
    binary_groups = [BinaryHamiltonian(group) for group in groups]
    return compute_quick_metrics(
        binary_groups,
        covariances,
        reporter,
    )


def main(argv=None):
    total_start = time.perf_counter()
    args = parse_args(argv)
    selected = select_graph_inputs(args)
    print("Selected {}.".format(selected["source"]), flush=True)

    chemistry_start = time.perf_counter()
    mol, hamiltonian, fermion_hamiltonian, n_paulis, qubit_operator = args.func()
    n_qubits = int(count_qubits(qubit_operator))
    binary_hamiltonian = BinaryHamiltonian.init_from_qubit_hamiltonian(hamiltonian)
    binary_terms = fast_ics.ordered_measurable_terms(binary_hamiltonian)
    if len(binary_terms) != n_paulis:
        raise ValueError(
            "Hamiltonian reports {} measurable terms, but {} were found.".format(
                n_paulis,
                len(binary_terms),
            )
        )
    print(
        "Hamiltonian: qubits={}, measurable Pauli terms={}, build_s={:.3f}.".format(
            n_qubits,
            n_paulis,
            time.perf_counter() - chemistry_start,
        ),
        flush=True,
    )

    state_start = time.perf_counter()
    sparse_hamiltonian = get_sparse_operator(qubit_operator)
    energy, variance_wfn = get_variance_wavefunction(
        mol,
        qubit_operator,
        method=args.wfn,
        sparse_hamiltonian=sparse_hamiltonian,
    )
    variance_wfn = np.asarray(variance_wfn, dtype=complex).reshape(-1)
    print(
        "{} energy={:.12g}; state build_s={:.3f}.".format(
            args.wfn,
            energy,
            time.perf_counter() - state_start,
        ),
        flush=True,
    )
    del sparse_hamiltonian, qubit_operator, hamiltonian, fermion_hamiltonian, mol

    grouping_condition = "qwc" if args.qwc else "fc"
    print(
        "Grouping condition={}.".format(grouping_condition.upper()),
        flush=True,
    )
    position_by_key = {
        term.binary_tuple(): position
        for position, term in enumerate(binary_terms)
    }
    compatibility = fast_ics.build_compatibility_matrix(
        binary_terms,
        grouping_condition,
    )

    terms_by_support = {}
    for term in binary_terms:
        support = fast_ics._pauli_support_key(term.to_pauli_strings())
        if support in terms_by_support:
            raise ValueError(
                "Hamiltonian contains duplicate measurable Pauli term {}.".format(
                    support
                )
            )
        terms_by_support[support] = term
    gflow_initial_groups = fast_ics.resolve_ordered_graph_groups(
        selected["graph"],
        terms_by_support,
        set(terms_by_support),
        grouping_condition,
        position_by_key=position_by_key,
        compatibility=compatibility,
    )
    si_groups, _ = binary_hamiltonian.commuting_groups(
        options={"method": "si", "condition": grouping_condition}
    )

    si_plan, si_pairs = fast_ics.build_fast_ics_plan(
        si_groups,
        grouping_condition,
        position_by_key,
        binary_terms,
        compatibility,
        signature="__sorted_insertion__",
    )
    graph_signature = fast_ics.ics_input_signature(
        gflow_initial_groups,
        position_by_key,
    )
    graph_plan, graph_pairs = fast_ics.build_fast_ics_plan(
        gflow_initial_groups,
        grouping_condition,
        position_by_key,
        binary_terms,
        compatibility,
        signature=graph_signature,
    )
    required_pairs = set(si_pairs)
    required_pairs.update(graph_pairs)

    fingerprint = fast_ics.hamiltonian_state_fingerprint(
        binary_terms,
        variance_wfn,
        n_qubits,
        args.wfn,
    )
    moment_path, graph_ics_path, graph_ics_fingerprint = cache_paths(
        args,
        grouping_condition,
        fingerprint,
    )
    covariance_start = time.perf_counter()
    covariances = fast_ics.build_selected_covariance(
        binary_terms,
        required_pairs,
        n_qubits,
        variance_wfn,
        args.cov_workers,
        args.blas_threads,
        moment_cache_path=moment_path,
        fingerprint=fingerprint,
        rebuild_cache=args.rebuild_cache,
    )
    print(
        "Selected covariance entries={}; runtime_s={:.3f}.".format(
            len(covariances),
            time.perf_counter() - covariance_start,
        ),
        flush=True,
    )

    reporter = fast_ics.SparseTequilaReporter(variance_wfn)
    si_metrics = compute_quick_metrics(
        si_groups,
        covariances,
        reporter,
    )
    si_ics_metrics = optimize_plan(
        si_plan,
        covariances,
        reporter,
        args.blas_threads,
    )

    graph_cache = fast_ics.load_versioned_cache(
        graph_ics_path,
        graph_ics_fingerprint,
        rebuild=args.rebuild_cache,
    )
    graph_results = dict(graph_cache.get("results", {})) if graph_cache else {}
    if graph_signature in graph_results:
        graph_ics_metrics = graph_results[graph_signature]
        print("Loaded selected graph ICS result from cache.", flush=True)
    else:
        graph_ics_metrics = optimize_plan(
            graph_plan,
            covariances,
            reporter,
            args.blas_threads,
        )
        graph_results[graph_signature] = compact_quick_metrics(graph_ics_metrics)
        fast_ics.save_versioned_cache(
            graph_ics_path,
            graph_ics_fingerprint,
            results=graph_results,
        )

    if selected["cached_before"] is None:
        graph_before_metrics = compute_quick_metrics(
            gflow_initial_groups,
            covariances,
            reporter,
        )
        graph_before_source = "computed from the top-graphs entry"
    else:
        graph_before_metrics = selected["cached_before"]
        graph_before_source = "loaded from the metrics cache"

    print("")
    print("Sorted insertion:")
    print("  SI:     {}".format(format_quick_metrics(si_metrics)))
    print("  SI-ICS: {}".format(format_quick_metrics(si_ics_metrics)))
    print("")
    print("Best ordered GFlowNet graph:")
    print(
        "  Before ICS ({}): {}".format(
            graph_before_source,
            format_quick_metrics(graph_before_metrics),
        )
    )
    print(
        "  After  ICS: {}".format(
            format_quick_metrics(graph_ics_metrics)
        )
    )
    print("")
    print("Total runtime_s={:.3f}.".format(time.perf_counter() - total_start))


if __name__ == "__main__":
    main()
