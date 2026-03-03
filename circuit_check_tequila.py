import networkx as nx
from tequila.grouping.binary_rep import BinaryHamiltonian

from gflow_vqe.circuit_helpers import (
    CircuitStats,
    grouping_circuit_stats_tequila,
    sorted_insertion_circuit_stats_tequila,
)
from gflow_vqe.utils import *
from gflow_vqe.hamiltonians import parse_driver_args


def _print_circuit_stats(label: str, stats: CircuitStats) -> None:
    print("{}:".format(label))
    print("  Number of qubits={}".format(stats.n_qubits))
    print("  Number of groups={}".format(stats.num_groups))
    print("  Total gates across all groups={}".format(stats.total_gates))
    print("  Total two-qubit gates across all groups={}".format(stats.total_two_qubit_gates))
    print("  Total depth across all measurement circuits={}".format(stats.total_depth))
    print("  Maximum single-group depth={}".format(stats.max_group_depth))
    print("  Circuit-cost estimation time (s)={:.6f}".format(stats.estimation_time_seconds))
    print("  Per-group costs:")
    for group in stats.group_stats:
        print(
            "    Group {}: terms={}, generators={}, active_qubits={}, total_gates={}, two_qubit_gates={}, depth={}".format(
                group.color,
                group.n_terms,
                group.n_generators,
                group.active_qubits,
                group.total_gates,
                group.two_qubit_gates,
                group.depth,
            )
        )


def main() -> None:
    args = parse_driver_args()
    molecule = args.func
    if molecule is None:
        raise ValueError("Unknown molecule '{}'".format(args.func_name))

    print("Molecule={}".format(args.func_name))
    mol, H, Hferm, n_paulis, Hq = molecule()
    print("Number of Pauli products to measure: {}".format(n_paulis))

    binary_hamiltonian = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
    terms = get_terms(binary_hamiltonian)
    comp_matrix = QWC_CompMatrix(terms)
    graph = obj_to_comp_graph(terms, comp_matrix)

    color_map = nx.coloring.greedy_color(graph, strategy="largest_first")
    nx.set_node_attributes(graph, color_map, "color")

    print("Greedy coloring max color={}".format(max_color(graph)))
    color_stats = grouping_circuit_stats_tequila(graph)
    si_stats = sorted_insertion_circuit_stats_tequila(binary_hamiltonian, condition="qwc")

    _print_circuit_stats("Greedy coloring grouping (Tequila compiled)", color_stats)
    _print_circuit_stats("Sorted insertion grouping (fc, Tequila compiled)", si_stats)
    print("Circuit check completed.")


if __name__ == "__main__":
    main()
