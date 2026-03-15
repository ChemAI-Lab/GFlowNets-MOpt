import argparse
import pickle

import networkx as nx
from openfermion.linalg import get_sparse_operator
from openfermion.utils import count_qubits
from tequila.grouping.binary_rep import BinaryHamiltonian

from gflow_vqe import hamiltonians as hamlib
from gflow_vqe.overlapping_helpers import (
    as_tequila_wavefunction,
    extract_measurable_terms,
    get_opt_sample_size,
    groups_from_gflow_grouping,
    iterative_coefficient_splitting_from_gflow_grouping,
    iterative_coefficient_splitting_from_groups,
    prepare_cov_dict,
)
from gflow_vqe.utils import (
    FC_CompMatrix,
    color_reward,
    get_groups_measurement,
    get_terms,
    get_variance_wavefunction,
    obj_to_comp_graph,
)


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
        "--gflow-graphs",
        type=str,
        default=None,
        help="Pickle file containing sampled GFlowNet graphs. Defaults to <func_name>_sampled_graphs.p.",
    )
    args = parser.parse_args(argv)
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
    cov_dict = prepare_cov_dict(binary_hamiltonian, variance_wfn)

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
