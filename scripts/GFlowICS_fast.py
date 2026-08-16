import argparse
import math
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import networkx as nx
import numpy as np
import tequila as tq
from openfermion import QubitOperator
from openfermion.linalg import get_sparse_operator
from openfermion.utils import count_qubits
from tequila.grouping.binary_rep import BinaryHamiltonian
from tequila.hamiltonian import QubitHamiltonian

from gflow_vqe import hamiltonians as hamlib
from gflow_vqe.overlapping_helpers import (
    as_tequila_wavefunction,
    extract_measurable_terms,
    get_opt_sample_size,
    groups_from_gflow_grouping,
    iterative_coefficient_splitting_from_gflow_grouping,
    iterative_coefficient_splitting_from_groups,
)
from gflow_vqe.utils import (
    FC_CompMatrix,
    color_reward,
    get_groups_measurement,
    get_terms,
    get_variance_wavefunction,
    obj_to_comp_graph,
)


DEFAULT_COVARIANCE_CHUNKSIZE = 128


@dataclass(frozen=True)
class PauliTerm:
    index: int
    pauli_tuple: tuple[tuple[int, str], ...]
    ops: tuple[str, ...]
    coefficient: complex
    word: str
    source_order: int


_ACTION_STATE = None
_ACTION_N_QUBITS = None
_ACTION_TERMS = None


def default_cov_workers():
    return max(1, min(8, os.cpu_count() or 1))


def clean_complex(value, tiny=1.0e-12):
    value = complex(value)
    # Small real components can contribute to the Greedy objective when many
    # covariances are accumulated.  Preserve them; only discard imaginary
    # roundoff from quantities that should be real.
    real = value.real
    imag = 0.0 if abs(value.imag) < tiny else value.imag
    return complex(real, imag)


def pauli_word(pauli_tuple):
    if not pauli_tuple:
        return "I"
    return " ".join("{}{}".format(pauli, qubit) for qubit, pauli in pauli_tuple)


def make_terms(qubit_operator, n_qubits):
    source_items = list(qubit_operator.terms.items())
    source_order = {
        pauli_tuple: position
        for position, (pauli_tuple, _) in enumerate(source_items)
    }
    items = sorted(source_items, key=lambda item: (bool(item[0]), item[0]))
    terms = []
    for index, (pauli_tuple_value, coefficient) in enumerate(items):
        pauli_tuple_value = tuple(
            (int(qubit), str(pauli)) for qubit, pauli in pauli_tuple_value
        )
        pauli_by_qubit = dict(pauli_tuple_value)
        terms.append(
            PauliTerm(
                index=index,
                pauli_tuple=pauli_tuple_value,
                ops=tuple(
                    pauli_by_qubit.get(qubit, "I") for qubit in range(n_qubits)
                ),
                coefficient=clean_complex(coefficient),
                word=pauli_word(pauli_tuple_value),
                source_order=source_order[pauli_tuple_value],
            )
        )
    return terms


def terms_fully_commute(term1, term2):
    anticommutes = sum(
        op1 != "I" and op2 != "I" and op1 != op2
        for op1, op2 in zip(term1.ops, term2.ops)
    )
    return anticommutes % 2 == 0


def tequila_wavefunction_from_array(state_vector):
    return tq.QubitWaveFunction.from_array(np.asarray(state_vector, dtype=complex))


def pauli_hamiltonian_for_term(term):
    return QubitHamiltonian.from_openfermion(
        QubitOperator(term.pauli_tuple, 1.0)
    )


def wavefunction_array(wfn, dimension):
    array = np.asarray(wfn.to_array(), dtype=complex).reshape(-1)
    if array.size != dimension:
        raise ValueError(
            "Expected wavefunction array of size {}, got {}.".format(
                dimension,
                array.size,
            )
        )
    return array


def action_row_for_term(term, reference_wfn, dimension):
    return wavefunction_array(
        pauli_hamiltonian_for_term(term)(reference_wfn),
        dimension,
    )


def _init_action_worker(state_vector, n_qubits, terms):
    global _ACTION_STATE
    global _ACTION_N_QUBITS
    global _ACTION_TERMS
    _ACTION_STATE = tequila_wavefunction_from_array(state_vector)
    _ACTION_N_QUBITS = int(n_qubits)
    _ACTION_TERMS = list(terms)


def _action_rows_chunk(term_positions):
    dimension = 2**_ACTION_N_QUBITS
    return [
        (
            position,
            action_row_for_term(_ACTION_TERMS[position], _ACTION_STATE, dimension),
        )
        for position in term_positions
    ]


def iter_index_chunks(n_items, chunksize):
    for start in range(0, n_items, chunksize):
        yield list(range(start, min(start + chunksize, n_items)))


def build_action_matrix(terms, state_vector, n_qubits, max_workers, chunksize):
    dimension = 2**n_qubits
    state_vector = np.asarray(state_vector, dtype=complex).reshape(-1)
    if state_vector.size != dimension:
        raise ValueError(
            "Expected statevector size {}, got {}.".format(
                dimension,
                state_vector.size,
            )
        )

    actions = np.empty((len(terms), dimension), dtype=complex)
    if max_workers == 1:
        reference_wfn = tequila_wavefunction_from_array(state_vector)
        for position, term in enumerate(terms):
            actions[position] = action_row_for_term(
                term,
                reference_wfn,
                dimension,
            )
        return actions

    automatic_chunksize = max(1, math.ceil(len(terms) / (4 * max_workers)))
    task_chunksize = min(chunksize, automatic_chunksize)
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_action_worker,
        initargs=(state_vector, n_qubits, terms),
    ) as executor:
        chunks = iter_index_chunks(len(terms), task_chunksize)
        for chunk_rows in executor.map(_action_rows_chunk, chunks):
            for position, row in chunk_rows:
                actions[position] = row
    return actions


def build_covariance_dictionary(
    terms,
    state_vector,
    n_qubits,
    max_workers,
    chunksize,
):
    """Build all commuting covariances from one Pauli-action matrix."""

    state_vector = np.asarray(state_vector, dtype=complex).reshape(-1)
    actions = build_action_matrix(
        terms,
        state_vector,
        n_qubits,
        max_workers,
        chunksize,
    )
    single_values = actions.dot(state_vector.conjugate())
    gram = actions.conjugate().dot(actions.T)
    single_expectations = {
        term.index: clean_complex(single_values[position])
        for position, term in enumerate(terms)
    }

    covariances = {}
    for left_position, left in enumerate(terms):
        for right_position in range(left_position, len(terms)):
            right = terms[right_position]
            if not terms_fully_commute(left, right):
                continue
            covariance = clean_complex(
                gram[left_position, right_position]
                - single_expectations[left.index]
                * single_expectations[right.index]
            )
            covariances[(left.index, right.index)] = covariance
    return covariances, single_expectations


def binary_tuple_for_term(term):
    n_qubits = len(term.ops)
    x_bits = [0.0] * n_qubits
    z_bits = [0.0] * n_qubits
    for qubit, pauli in term.pauli_tuple:
        if pauli in ("X", "Y"):
            x_bits[qubit] = 1.0
        if pauli in ("Z", "Y"):
            z_bits[qubit] = 1.0
    return tuple(x_bits + z_bits)


def prepare_fast_cov_dict(binary_hamiltonian, qubit_operator, approx_wfn, max_workers):
    n_qubits = int(count_qubits(qubit_operator))
    terms = make_terms(qubit_operator, n_qubits)
    covariances_by_index, _ = build_covariance_dictionary(
        terms,
        approx_wfn,
        n_qubits,
        max_workers,
        DEFAULT_COVARIANCE_CHUNKSIZE,
    )

    keys_by_index = {
        term.index: binary_tuple_for_term(term)
        for term in terms
    }
    binary_term_order = {
        term.binary_tuple(): position
        for position, term in enumerate(binary_hamiltonian.binary_terms)
    }

    cov_dict = {}
    for (left_index, right_index), covariance in covariances_by_index.items():
        left_key = keys_by_index[left_index]
        right_key = keys_by_index[right_index]
        if binary_term_order[left_key] <= binary_term_order[right_key]:
            pair = (left_key, right_key)
        else:
            pair = (right_key, left_key)
        cov_dict[pair] = covariance
    return cov_dict


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare Tequila ICS against ICS initialized from GFlowNet-compatible groups."
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
        help="Wavefunction used for covariance and measurement reporting (default: FCI).",
    )
    parser.add_argument(
        "--cov-workers",
        type=int,
        default=default_cov_workers(),
        help="Worker processes used to construct Pauli action rows (default: up to 8).",
    )
    parser.add_argument(
        "--gflow-graphs",
        type=str,
        default=None,
        help="Pickle file containing sampled GFlowNet graphs. Defaults to <func_name>_sampled_graphs.p.",
    )
    args = parser.parse_args(argv)
    if args.cov_workers < 1:
        parser.error("--cov-workers must be at least 1.")
    args.func = getattr(hamlib, args.func_name, None)
    if args.func is None:
        raise ValueError("Unknown molecule '{}'".format(args.func_name))
    return args


def _to_real_if_close(value, tiny=1e-12):
    if hasattr(value, "imag") and abs(value.imag) < tiny:
        return float(value.real)
    return value


def optimal_allocation_metric(commuting_parts, suggested_sample_size, wfn, tiny=1e-12):
    measurement_metric = 0
    wf = as_tequila_wavefunction(wfn)

    for idx, part in enumerate(commuting_parts):
        op = part.to_qubit_hamiltonian()
        var_part = wf.inner((op * op)(wf)) - wf.inner(op(wf)) ** 2
        if hasattr(var_part, "imag") and abs(var_part.imag) < tiny:
            var_part = var_part.real
        measurement_metric += var_part / suggested_sample_size[idx]

    return _to_real_if_close(measurement_metric, tiny=tiny)


def _format_group(group):
    terms = group.binary_terms if isinstance(group, BinaryHamiltonian) else list(group)
    return "[" + ", ".join(str(term.to_pauli_strings()) for term in terms) + "]"


def _print_report(label, groups, suggested_sample_size, wfn):
    measurement_metric = optimal_allocation_metric(groups, suggested_sample_size, wfn)
    print("{}:".format(label))
    print("  Required number of measurements={}".format(measurement_metric))
    print("  Number of groups={}".format(len(groups)))
    print("  Suggested sample ratios={}".format([float(x) for x in suggested_sample_size]))
    # print("  Groups:")
    # for idx, group in enumerate(groups):
    #     print("    {}: {}".format(idx, _format_group(group)))


def _color_signature(graph):
    return tuple(sorted(nx.get_node_attributes(graph, "color").items()))


def _default_gflow_compatible_graph(binary_hamiltonian):
    terms = get_terms(binary_hamiltonian)
    comp_matrix = FC_CompMatrix(terms)
    graph = obj_to_comp_graph(terms, comp_matrix)
    color_map = nx.coloring.greedy_color(graph, strategy="largest_first")
    nx.set_node_attributes(graph, color_map, "color")
    return graph


def load_best_gflow_grouping(path, wfn, n_qubits, binary_hamiltonian):
    try:
        with open(path, "rb") as handle:
            sampled_graphs = pickle.load(handle)
        source_label = "GFlowNet sampled graphs from '{}'".format(path)
    except FileNotFoundError:
        fallback_graph = _default_gflow_compatible_graph(binary_hamiltonian)
        return fallback_graph, "greedy largest-first coloring fallback"

    if not isinstance(sampled_graphs, list):
        sampled_graphs = list(sampled_graphs)

    unique_graphs = []
    seen = set()
    for graph in sampled_graphs:
        signature = _color_signature(graph)
        if signature in seen:
            continue
        seen.add(signature)
        unique_graphs.append(graph)

    valid_graphs = [graph for graph in unique_graphs if color_reward(graph) > 0]
    if not valid_graphs:
        raise ValueError("No valid GFlowNet colorings were found in '{}'.".format(path))

    best_graph = min(valid_graphs, key=lambda graph: get_groups_measurement(graph, wfn, n_qubits))
    return best_graph, source_label


def main(argv=None):
    ##############Block for normal Hamiltonians###########################
    args = parse_args(argv)
    gflow_graphs_path = args.gflow_graphs or "{}_sampled_graphs.p".format(args.func_name)

    mol, H, _, n_paulis, Hq = args.func()
    print("Number of Pauli products to measure: {}".format(n_paulis))
    ######################################################################
    ##############Block for loaded Hamiltonians###########################
    #This driver takes Hamiltonians from npj Quantum Inf 9, 14 (2023). https://doi.org/10.1038/s41534-023-00683-y
    # MOLECULES = ["h2", "lih", "beh2", "h2o", "nh3", "n2"]
    # mol="lih"
    # Hq, H = load_qubit_hamiltonian(mol)
    # print("Number of Pauli products to measure: {}".format(len(Hq.terms) - 1))
    ######################################################################
    sparse_hamiltonian = get_sparse_operator(Hq)
    energy, variance_wfn = get_variance_wavefunction(mol, Hq, method=args.wfn, sparse_hamiltonian=sparse_hamiltonian)
    print("{} Energy={}".format(args.wfn, energy))

    n_qubits = count_qubits(Hq)
    binary_hamiltonian = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
    cov_dict = prepare_fast_cov_dict(binary_hamiltonian, Hq, variance_wfn, args.cov_workers)

    si_groups, si_sample_size = binary_hamiltonian.commuting_groups(
        options={"method": "si", "condition": "fc", "cov_dict": cov_dict}
    )
    si_ics_groups, si_ics_sample_size = iterative_coefficient_splitting_from_groups(
        si_groups,
        cov_dict,
        condition="fc",
    )
    tequila_ics_groups, tequila_ics_sample_size = binary_hamiltonian.commuting_groups(
        options={"method": "ics", "condition": "fc", "cov_dict": cov_dict}
    )

    best_gflow_graph, gflow_source_label = load_best_gflow_grouping(
        gflow_graphs_path,
        variance_wfn,
        n_qubits,
        binary_hamiltonian,
    )
    gflow_initial_groups = groups_from_gflow_grouping(best_gflow_graph, extract_measurable_terms(binary_hamiltonian))
    gflow_initial_groups = [BinaryHamiltonian(group) for group in gflow_initial_groups]
    gflow_initial_sample_size = get_opt_sample_size(
        [group.binary_terms for group in gflow_initial_groups],
        cov_dict,
    )
    gflow_ics_groups, gflow_ics_sample_size = iterative_coefficient_splitting_from_gflow_grouping(
        binary_hamiltonian,
        best_gflow_graph,
        cov_dict,
        condition="fc",
    )

    print("")
    print("Using {}".format(gflow_source_label))
    print("")
    _print_report("Sorted insertion", si_groups, si_sample_size, variance_wfn)
    print("")
    _print_report("ICS initialized from sorted insertion groups", si_ics_groups, si_ics_sample_size, variance_wfn)
    print("")
    _print_report("Tequila ICS", tequila_ics_groups, tequila_ics_sample_size, variance_wfn)
    print("")
    _print_report("Selected GFlowNet-compatible initial grouping", gflow_initial_groups, gflow_initial_sample_size, variance_wfn)
    print("")
    _print_report("ICS initialized from GFlowNet-compatible grouping", gflow_ics_groups, gflow_ics_sample_size, variance_wfn)


if __name__ == "__main__":
    main()
