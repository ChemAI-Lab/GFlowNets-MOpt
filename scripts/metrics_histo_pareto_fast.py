"""Fast Pareto-metric analysis for standard and coefficient-ordered graphs.

The saved graphs carry their Pauli term in each node's ``v`` attribute.  This
script therefore maps by that metadata instead of assuming that graph node
numbers follow either the native or coefficient-sorted Hamiltonian order.

Measurement metrics are evaluated from the Pauli products that are actually
needed by the sampled partitions.  Circuit metrics remain Tequila's circuit
counts, but color-label-equivalent partitions are compiled only once and can
be distributed over worker processes.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import gc
import math
import multiprocessing as mp
import os
import pickle
import time

# Process-level parallelism is used explicitly below.  Prevent each worker
# from starting another large BLAS/OpenMP pool as well.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from openfermion import count_qubits, get_sparse_operator
from threadpoolctl import threadpool_limits

import gflow_vqe.hamiltonians as hamlib
from gflow_vqe.circuit_helpers import grouping_circuit_stats_tequila
from gflow_vqe.utils import get_variance_wavefunction


VARIANCE_TINY = 1.0e-10
REAL_TOLERANCE = 1.0e-8
MOMENT_WORKER_CAP = 32


@dataclass(frozen=True)
class PauliTerm:
    index: int
    pauli_tuple: tuple
    coefficient: complex
    x_mask: int
    z_mask: int
    n_y: int

    @property
    def moment_key(self):
        return self.x_mask, self.z_mask


_MOMENT_STATE = None
_MOMENT_BASIS = None
_MOMENT_PARITY = None
_MOMENT_SUPPORT = None
_WORKER_THREAD_LIMIT = None
_CIRCUIT_TERMS = None


def default_workers():
    return max(1, os.cpu_count() or 1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Fast Pareto plot by epsilon^2 M against number of groups or "
            "2-qubit gates. Saved graphs from driver_sv.py and "
            "driver_sv_ordered.py are both supported."
        )
    )
    parser.add_argument(
        "func_name",
        type=str,
        help="Molecule helper from gflow_vqe.hamiltonians (for example H2, LiH, BeH2, N2).",
    )
    parser.add_argument(
        "--wfn",
        type=lambda value: str(value).upper(),
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
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help=(
            "Maximum worker processes (default: all available CPUs). Circuit "
            "counts can use the full request; the memory-bound Pauli-moment "
            "stage adaptively caps it at 32."
        ),
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1.")
    args.func = getattr(hamlib, args.func_name, None)
    if args.func is None:
        raise ValueError("Unknown molecule '{}'".format(args.func_name))
    return args


def pareto_front_min(pts):
    if len(pts) == 0:
        return np.array([], dtype=bool)
    is_pareto = np.ones(pts.shape[0], dtype=bool)
    for index, point in enumerate(pts):
        if not is_pareto[index]:
            continue
        dominated = np.any(np.all(pts <= point, axis=1) & np.any(pts < point, axis=1))
        if dominated:
            is_pareto[index] = False
    return is_pareto


def _progress(label, completed, total, next_fraction):
    if total < 10:
        return next_fraction
    fraction = completed / total
    if fraction + 1.0e-12 >= next_fraction:
        print("{}: {}/{} ({:.0%})".format(label, completed, total, fraction))
        return next_fraction + 0.1
    return next_fraction


def _color_groups_if_valid(graph):
    """Return color groups in graph-node order, or None for reward-zero graphs."""

    nodes = tuple(graph.nodes())
    if not nodes:
        return None

    color_to_nodes = {}
    for node in nodes:
        data = graph.nodes[node]
        if "color" not in data:
            raise ValueError(
                "Graph node {} is missing the 'color' attribute.".format(node)
            )
        color_to_nodes.setdefault(data["color"], []).append(node)

    # color_reward is zero when every term has its own color.
    if len(color_to_nodes) >= len(nodes):
        return None

    # Checking only pairs within a proposed group is equivalent to scanning all
    # conflict edges, and is much cheaper for the many-group large-molecule case.
    for group_nodes in color_to_nodes.values():
        for left_position, left_node in enumerate(group_nodes):
            neighbors = graph[left_node]
            if left_node in neighbors:
                return None
            for right_node in group_nodes[left_position + 1 :]:
                if right_node in neighbors:
                    return None

    return tuple(tuple(group) for group in color_to_nodes.values())


def load_valid_graph_records(sampled_graphs_path):
    if not os.path.exists(sampled_graphs_path):
        raise FileNotFoundError(
            "Could not find sampled graphs file '{}'.".format(sampled_graphs_path)
        )

    with open(sampled_graphs_path, "rb") as handle:
        sampled_graphs = pickle.load(handle)
    if not isinstance(sampled_graphs, list):
        sampled_graphs = list(sampled_graphs)

    start = time.perf_counter()
    records = []
    next_fraction = 0.1
    for graph_index, graph in enumerate(sampled_graphs, start=1):
        node_groups = _color_groups_if_valid(graph)
        if node_groups is not None:
            records.append((graph, node_groups))
        next_fraction = _progress(
            "Filtering sampled graphs",
            graph_index,
            len(sampled_graphs),
            next_fraction,
        )

    if not records:
        raise RuntimeError(
            "No valid sampled graphs were found in '{}'.".format(sampled_graphs_path)
        )
    print(
        "Graph filtering: {:.3f} s ({} valid of {})".format(
            time.perf_counter() - start,
            len(records),
            len(sampled_graphs),
        )
    )
    return records


def load_cached_metrics(metrics_path, expected_length):
    if not os.path.exists(metrics_path):
        return None
    try:
        with open(metrics_path, "rb") as handle:
            cached = pickle.load(handle)
    except Exception as error:
        print("Could not load metrics cache '{}': {}".format(metrics_path, error))
        return None

    required = ("measurements", "num_groups", "two_qubit_gates")
    valid = isinstance(cached, dict) and all(key in cached for key in required)
    if valid:
        arrays = {key: np.asarray(cached[key]) for key in required}
        valid = all(
            array.ndim == 1 and len(array) == expected_length
            for array in arrays.values()
        )
    if not valid:
        print(
            "Cached metrics file '{}' does not match expected format; rebuilding.".format(
                metrics_path
            )
        )
        return None

    print("Loaded metrics from {}".format(metrics_path))
    return cached


def _support_key(pauli_string):
    if not hasattr(pauli_string, "items"):
        raise TypeError(
            "Expected graph node attribute 'v' to be a Tequila PauliString, got {}.".format(
                type(pauli_string).__name__
            )
        )
    return tuple(
        sorted(
            (int(qubit), str(pauli).upper()) for qubit, pauli in pauli_string.items()
        )
    )


def make_pauli_terms(qubit_operator, n_qubits):
    terms = []
    for pauli_tuple, coefficient in qubit_operator.terms.items():
        if not pauli_tuple:
            continue
        normalized_tuple = tuple(
            (int(qubit), str(pauli).upper()) for qubit, pauli in pauli_tuple
        )
        x_mask = 0
        z_mask = 0
        n_y = 0
        for qubit, pauli in normalized_tuple:
            bit = 1 << (n_qubits - 1 - qubit)
            if pauli in ("X", "Y"):
                x_mask |= bit
            if pauli in ("Z", "Y"):
                z_mask |= bit
            if pauli == "Y":
                n_y += 1
        terms.append(
            PauliTerm(
                index=len(terms),
                pauli_tuple=normalized_tuple,
                coefficient=complex(coefficient),
                x_mask=x_mask,
                z_mask=z_mask,
                n_y=n_y,
            )
        )
    return terms


def map_graph_partitions(records, terms):
    terms_by_support = {term.pauli_tuple: term.index for term in terms}
    if len(terms_by_support) != len(terms):
        raise ValueError(
            "The Hamiltonian contains duplicate measurable Pauli supports."
        )

    graph_signatures = []
    representatives = {}
    circuit_terms = [None] * len(terms)
    expected_term_ids = set(range(len(terms)))
    start = time.perf_counter()
    next_fraction = 0.1

    for graph_index, (graph, node_groups) in enumerate(records, start=1):
        term_ids_by_node = {}
        seen_term_ids = set()
        for node, data in graph.nodes(data=True):
            if "v" not in data:
                raise ValueError(
                    "Graph node {} is missing the embedded Pauli-term attribute 'v'.".format(
                        node
                    )
                )
            support = _support_key(data["v"])
            if support not in terms_by_support:
                raise ValueError(
                    "Graph node {} contains a Pauli term absent from the current Hamiltonian: {}.".format(
                        node,
                        support,
                    )
                )
            term_index = terms_by_support[support]
            if term_index in seen_term_ids:
                raise ValueError(
                    "A sampled graph contains a duplicate Pauli term {}.".format(
                        support
                    )
                )
            seen_term_ids.add(term_index)
            term_ids_by_node[node] = term_index

            graph_coefficient = complex(getattr(data["v"], "coeff", 1.0))
            if not np.isclose(
                graph_coefficient,
                terms[term_index].coefficient,
                rtol=1.0e-9,
                atol=1.0e-11,
            ):
                raise ValueError(
                    "Coefficient mismatch for Pauli term {}: graph={}, Hamiltonian={}.".format(
                        support,
                        graph_coefficient,
                        terms[term_index].coefficient,
                    )
                )
            if circuit_terms[term_index] is None:
                circuit_terms[term_index] = data["v"]

        if seen_term_ids != expected_term_ids:
            missing = sorted(expected_term_ids - seen_term_ids)
            raise ValueError(
                "Sampled graph {} does not contain the current Hamiltonian's full measurable "
                "term set; missing term indices {}.".format(graph_index - 1, missing)
            )

        ordered_groups = tuple(
            tuple(term_ids_by_node[node] for node in group_nodes)
            for group_nodes in node_groups
        )
        signature = tuple(sorted(tuple(sorted(group)) for group in ordered_groups))
        graph_signatures.append(signature)
        representatives.setdefault(signature, ordered_groups)

        next_fraction = _progress(
            "Mapping embedded Pauli terms",
            graph_index,
            len(records),
            next_fraction,
        )

    if any(term is None for term in circuit_terms):
        raise ValueError(
            "Could not recover every measurable Pauli term from graph metadata."
        )

    print(
        "Partition mapping: {:.3f} s ({} graphs -> {} unique color-label-independent partitions)".format(
            time.perf_counter() - start,
            len(graph_signatures),
            len(representatives),
        )
    )
    return graph_signatures, representatives, tuple(circuit_terms)


def collect_required_pairs(partition_signatures, n_terms):
    unique_groups = set()
    for signature in partition_signatures:
        unique_groups.update(signature)

    pair_codes = set()
    for group in unique_groups:
        for left_position, left_index in enumerate(group):
            for right_index in group[left_position:]:
                pair_codes.add(left_index * n_terms + right_index)
    return pair_codes, unique_groups


def _product_key_and_phase(left, right):
    x_mask = left.x_mask ^ right.x_mask
    z_mask = left.z_mask ^ right.z_mask
    n_y_product = (x_mask & z_mask).bit_count()
    phase_power = (left.n_y + right.n_y - n_y_product) % 4
    swap_sign = -1 if (left.z_mask & right.x_mask).bit_count() % 2 else 1
    phase = swap_sign * (1j**phase_power)
    return (x_mask, z_mask), phase


def _init_moment_worker(state_vector):
    global _MOMENT_STATE
    global _MOMENT_BASIS
    global _MOMENT_PARITY
    global _MOMENT_SUPPORT
    global _WORKER_THREAD_LIMIT

    _WORKER_THREAD_LIMIT = threadpool_limits(limits=1)
    _MOMENT_STATE = np.asarray(state_vector, dtype=complex).reshape(-1)
    _MOMENT_BASIS = np.arange(_MOMENT_STATE.size, dtype=np.uint64)
    _MOMENT_PARITY = np.fromiter(
        (index.bit_count() & 1 for index in range(_MOMENT_STATE.size)),
        dtype=np.int8,
        count=_MOMENT_STATE.size,
    )
    # This is exact sparsity only: no numerical amplitude threshold is used.
    _MOMENT_SUPPORT = np.flatnonzero(_MOMENT_STATE != 0).astype(np.uint64)


def _fwht_inplace(values):
    width = 1
    size = values.size
    while width < size:
        blocks = values.reshape(-1, 2 * width)
        left = blocks[:, :width].copy()
        right = blocks[:, width:].copy()
        blocks[:, :width] = left + right
        blocks[:, width:] = left - right
        width *= 2


def _moments_for_x(job):
    x_mask, z_masks = job
    dimension = _MOMENT_STATE.size
    transform_work = dimension * int(math.log2(dimension))
    direct_work = len(z_masks) * len(_MOMENT_SUPPORT)
    use_fwht = direct_work > transform_work

    results = []
    if use_fwht:
        target = np.bitwise_xor(_MOMENT_BASIS, np.uint64(x_mask))
        values = np.conjugate(_MOMENT_STATE[target]) * _MOMENT_STATE
        _fwht_inplace(values)
        for z_mask in z_masks:
            phase = 1j ** ((x_mask & z_mask).bit_count() % 4)
            results.append((z_mask, complex(phase * values[z_mask])))
    else:
        target = np.bitwise_xor(_MOMENT_SUPPORT, np.uint64(x_mask))
        values = np.conjugate(_MOMENT_STATE[target]) * _MOMENT_STATE[_MOMENT_SUPPORT]
        for z_mask in z_masks:
            signs = (
                1.0
                - 2.0
                * _MOMENT_PARITY[np.bitwise_and(_MOMENT_SUPPORT, np.uint64(z_mask))]
            )
            phase = 1j ** ((x_mask & z_mask).bit_count() % 4)
            results.append((z_mask, complex(phase * np.dot(signs, values))))
    return x_mask, results


def _process_context():
    if os.name == "posix" and "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return mp.get_context()


def build_required_moments(required_products, state_vector, workers):
    products_by_x = defaultdict(set)
    for x_mask, z_mask in required_products:
        products_by_x[x_mask].add(z_mask)
    jobs = [
        (x_mask, tuple(sorted(z_masks)))
        for x_mask, z_masks in sorted(products_by_x.items())
    ]

    moments = {}
    start = time.perf_counter()
    # These jobs repeatedly stream the same statevector and become
    # memory-bandwidth-bound well before large server core counts.  Keep the
    # user's full --workers value for the much heavier circuit-compilation
    # stage below, while avoiding counterproductive moment oversubscription.
    effective_workers = min(workers, len(jobs), MOMENT_WORKER_CAP)
    print(
        "Pauli moments: {} products in {} X-mask batches using {} worker(s); "
        "full-state exact support={}/{}".format(
            len(required_products),
            len(jobs),
            effective_workers,
            int(np.count_nonzero(np.asarray(state_vector) != 0)),
            int(np.asarray(state_vector).size),
        )
    )

    if effective_workers == 1:
        _init_moment_worker(state_vector)
        iterator = map(_moments_for_x, jobs)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=_process_context(),
            initializer=_init_moment_worker,
            initargs=(np.asarray(state_vector, dtype=complex),),
        )
        iterator = executor.map(_moments_for_x, jobs, chunksize=1)

    next_fraction = 0.1
    try:
        for completed, (x_mask, values) in enumerate(iterator, start=1):
            for z_mask, value in values:
                moments[(x_mask, z_mask)] = value
            next_fraction = _progress(
                "Pauli moments",
                completed,
                len(jobs),
                next_fraction,
            )
    finally:
        if executor is not None:
            executor.shutdown()

    print("Pauli moments completed in {:.3f} s".format(time.perf_counter() - start))
    return moments


def _variance_as_real(value, group):
    value = complex(value)
    if abs(value.imag) > REAL_TOLERANCE:
        raise ValueError(
            "Group variance has a non-negligible imaginary part {} for term indices {}.".format(
                value.imag,
                group,
            )
        )
    variance = value.real
    if variance < VARIANCE_TINY:
        variance = 0.0
    return variance


def build_partition_measurements(
    partition_representatives,
    terms,
    pair_codes,
    unique_groups,
    state_vector,
    workers,
):
    n_terms = len(terms)
    required_products = {term.moment_key for term in terms}
    pair_product_data = {}
    for pair_code in pair_codes:
        left_index, right_index = divmod(pair_code, n_terms)
        product_key, phase = _product_key_and_phase(
            terms[left_index],
            terms[right_index],
        )
        pair_product_data[pair_code] = (product_key, phase)
        required_products.add(product_key)

    moments = build_required_moments(required_products, state_vector, workers)

    pair_covariances = {}
    for pair_code, (product_key, phase) in pair_product_data.items():
        left_index, right_index = divmod(pair_code, n_terms)
        left = terms[left_index]
        right = terms[right_index]
        unit_covariance = (
            phase * moments[product_key]
            - moments[left.moment_key] * moments[right.moment_key]
        )
        pair_covariances[pair_code] = (
            left.coefficient * right.coefficient * unit_covariance
        )

    group_sqrt_variances = {}
    for group in unique_groups:
        variance = 0.0j
        for left_position, left_index in enumerate(group):
            diagonal_code = left_index * n_terms + left_index
            variance += pair_covariances[diagonal_code]
            for right_index in group[left_position + 1 :]:
                pair_code = left_index * n_terms + right_index
                variance += 2.0 * pair_covariances[pair_code]
        group_sqrt_variances[group] = math.sqrt(_variance_as_real(variance, group))

    measurements = {}
    for signature, representative_groups in partition_representatives.items():
        sqrt_variance_sum = sum(
            group_sqrt_variances[tuple(sorted(group))]
            for group in representative_groups
        )
        measurements[signature] = float(sqrt_variance_sum**2)
    return measurements


def _init_circuit_worker(circuit_terms):
    global _CIRCUIT_TERMS
    global _WORKER_THREAD_LIMIT
    _WORKER_THREAD_LIMIT = threadpool_limits(limits=1)
    _CIRCUIT_TERMS = circuit_terms


def _partition_two_qubit_gates(job):
    partition_index, representative_groups = job
    grouping = {
        color: [_CIRCUIT_TERMS[term_index] for term_index in group]
        for color, group in enumerate(representative_groups)
    }
    count = grouping_circuit_stats_tequila(grouping).total_two_qubit_gates
    return partition_index, int(count)


def build_partition_gate_counts(partition_representatives, circuit_terms, workers):
    signatures = list(partition_representatives)
    jobs = [
        (index, partition_representatives[signature])
        for index, signature in enumerate(signatures)
    ]
    effective_workers = min(workers, len(jobs))
    print(
        "Circuit metrics: {} unique partitions using {} worker(s)".format(
            len(jobs),
            effective_workers,
        )
    )
    start = time.perf_counter()

    if effective_workers == 1:
        _init_circuit_worker(circuit_terms)
        iterator = map(_partition_two_qubit_gates, jobs)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=_process_context(),
            initializer=_init_circuit_worker,
            initargs=(circuit_terms,),
        )
        iterator = executor.map(_partition_two_qubit_gates, jobs, chunksize=1)

    counts = [None] * len(jobs)
    next_fraction = 0.1
    try:
        for completed, (partition_index, count) in enumerate(iterator, start=1):
            counts[partition_index] = count
            next_fraction = _progress(
                "Circuit metrics",
                completed,
                len(jobs),
                next_fraction,
            )
    finally:
        if executor is not None:
            executor.shutdown()

    print("Circuit metrics completed in {:.3f} s".format(time.perf_counter() - start))
    return {signature: counts[index] for index, signature in enumerate(signatures)}


def build_metrics_fast(records, func, wfn_method, workers):
    setup_start = time.perf_counter()
    mol, _, _, n_paulis, qubit_operator = func()
    print("Number of Pauli products to measure: {}".format(n_paulis))
    sparse_hamiltonian = get_sparse_operator(qubit_operator)
    _, state_vector = get_variance_wavefunction(
        mol,
        qubit_operator,
        method=wfn_method,
        sparse_hamiltonian=sparse_hamiltonian,
    )
    del sparse_hamiltonian
    n_qubits = int(count_qubits(qubit_operator))
    state_vector = np.asarray(state_vector, dtype=complex).reshape(-1)
    expected_dimension = 2**n_qubits
    if state_vector.size != expected_dimension:
        raise ValueError(
            "Expected a {}-qubit statevector of size {}, got {}.".format(
                n_qubits,
                expected_dimension,
                state_vector.size,
            )
        )
    print(
        "Hamiltonian and {} wavefunction: {:.3f} s".format(
            wfn_method,
            time.perf_counter() - setup_start,
        )
    )

    terms = make_pauli_terms(qubit_operator, n_qubits)
    graph_signatures, representatives, circuit_terms = map_graph_partitions(
        records, terms
    )
    del mol, qubit_operator
    # Worker jobs use compact integer partitions and the retained PauliString
    # tuple, not the very large NetworkX objects.  Release graph storage before
    # forking the metric pools so it is not carried in every worker's address
    # space.
    if hasattr(records, "clear"):
        records.clear()
        gc.collect()
    pair_start = time.perf_counter()
    pair_codes, unique_groups = collect_required_pairs(representatives, len(terms))
    print(
        "Selected covariance pairs: {} across {} unique groups ({:.3f} s)".format(
            len(pair_codes),
            len(unique_groups),
            time.perf_counter() - pair_start,
        )
    )

    measurement_start = time.perf_counter()
    partition_measurements = build_partition_measurements(
        representatives,
        terms,
        pair_codes,
        unique_groups,
        state_vector,
        workers,
    )
    print(
        "Measurement metric assembly: {:.3f} s".format(
            time.perf_counter() - measurement_start
        )
    )
    partition_gate_counts = build_partition_gate_counts(
        representatives,
        circuit_terms,
        workers,
    )

    return {
        "measurements": np.asarray(
            [partition_measurements[signature] for signature in graph_signatures],
            dtype=float,
        ),
        "num_groups": np.asarray(
            [len(signature) for signature in graph_signatures],
            dtype=int,
        ),
        "two_qubit_gates": np.asarray(
            [partition_gate_counts[signature] for signature in graph_signatures],
            dtype=int,
        ),
    }


def padded_limits(values, pad_fraction=0.05, min_pad=0.05):
    values = np.asarray(values, dtype=float)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if np.isclose(minimum, maximum):
        pad = max(min_pad, 0.05 * max(1.0, abs(minimum)))
    else:
        pad = max(min_pad, pad_fraction * (maximum - minimum))
    return minimum - pad, maximum + pad


def plot_marginal(values, axis, orientation, color):
    values = np.asarray(values, dtype=float)
    if values.size > 1 and np.unique(values).size > 1:
        if orientation == "x":
            sns.kdeplot(x=values, ax=axis, fill=True, color=color, warn_singular=False)
        else:
            sns.kdeplot(y=values, ax=axis, fill=True, color=color, warn_singular=False)
        return
    if orientation == "x":
        axis.hist(values, bins=1, color=color, alpha=0.4)
    else:
        axis.hist(values, bins=1, color=color, alpha=0.4, orientation="horizontal")


def plot_metrics(metrics, y_axis, output_path):
    if y_axis == "groups":
        y = np.asarray(metrics["num_groups"], dtype=int)
        y_label = "$N_G(x)$"
        legacy_y_limits = (9.0, 30.0)
    else:
        y = np.asarray(metrics["two_qubit_gates"], dtype=int)
        y_label = "$N_{2q}(x)$"
        legacy_y_limits = None

    points = np.column_stack(
        (np.asarray(metrics["measurements"], dtype=float), y.astype(float))
    )
    x = points[:, 0]
    mask = pareto_front_min(points)
    pareto_sorted = points[mask][np.argsort(points[mask][:, 0])]

    sns.set_theme(style="whitegrid")
    grid = sns.JointGrid(x=x, y=points[:, 1], height=7.5, space=0)
    sns.scatterplot(
        x=x,
        y=points[:, 1],
        ax=grid.ax_joint,
        alpha=0.5,
        s=30,
        edgecolor=None,
    )
    if len(pareto_sorted) > 0:
        grid.ax_joint.plot(
            pareto_sorted[:, 0],
            pareto_sorted[:, 1],
            color="orange",
            marker="o",
            markersize=8,
            linewidth=1.5,
            label="Pareto front",
        )

    grid.set_axis_labels("$\\epsilon^2M(x)$", y_label, fontsize=14)
    plot_marginal(x, grid.ax_marg_x, "x", "purple")
    plot_marginal(points[:, 1], grid.ax_marg_y, "y", "green")
    grid.ax_marg_x.set_ylabel("")
    grid.ax_marg_y.set_xlabel("")
    grid.ax_marg_x.tick_params(axis="x", labelbottom=False)
    grid.ax_marg_y.tick_params(axis="y", labelleft=False)

    legacy_x_limits = (0.55, 1.8)
    if np.min(x) >= legacy_x_limits[0] and np.max(x) <= legacy_x_limits[1]:
        grid.ax_joint.set_xlim(*legacy_x_limits)
    else:
        grid.ax_joint.set_xlim(*padded_limits(x))

    if (
        legacy_y_limits is not None
        and np.min(y) >= legacy_y_limits[0]
        and np.max(y) <= legacy_y_limits[1]
    ):
        grid.ax_joint.set_ylim(*legacy_y_limits)
    else:
        y_min, y_max = padded_limits(y, min_pad=0.5)
        grid.ax_joint.set_ylim(max(0.0, y_min), y_max)
    grid.ax_joint.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    grid.ax_joint.legend(loc="best")
    grid.figure.savefig(output_path, format="svg", dpi=600, bbox_inches="tight")
    plt.close(grid.figure)
    print("Saved Pareto plot to {}".format(output_path))


def main(argv=None):
    total_start = time.perf_counter()
    args = parse_args(argv)
    fig_name = args.func_name
    sampled_graphs_path = fig_name + "_sampled_graphs.p"
    metrics_path = fig_name + "_sampled_graphs_metrics.p"
    if args.y_axis == "groups":
        output_path = fig_name + "_pareto_joint_all.svg"
    else:
        output_path = fig_name + "_pareto_joint_all_2qubit.svg"

    records = load_valid_graph_records(sampled_graphs_path)
    print("Number of valid graphs in file: {}".format(len(records)))
    metrics = load_cached_metrics(metrics_path, len(records))
    if metrics is None:
        build_start = time.perf_counter()
        metrics = build_metrics_fast(records, args.func, args.wfn, args.workers)
        with open(metrics_path, "wb") as handle:
            pickle.dump(metrics, handle, pickle.HIGHEST_PROTOCOL)
        print(
            "Computed and saved metrics to {} in {:.3f} s".format(
                metrics_path,
                time.perf_counter() - build_start,
            )
        )
    else:
        records.clear()
        gc.collect()

    plot_metrics(metrics, args.y_axis, output_path)
    print("Total analysis time: {:.3f} s".format(time.perf_counter() - total_start))


if __name__ == "__main__":
    main()
