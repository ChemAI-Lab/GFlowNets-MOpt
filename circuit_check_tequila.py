from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import tequila as tq
from tequila.grouping.binary_rep import BinaryHamiltonian
from tequila.grouping.compile_groups import compile_commuting_parts
from tequila.hamiltonian import PauliString, QubitHamiltonian

from circuit_check import _groups_from_grouping, _pauli_dict_from_term


@dataclass
class GroupCircuitStats:
    color: Any
    n_terms: int
    n_generators: int
    active_qubits: Tuple[int, ...]
    total_gates: int
    two_qubit_gates: int
    depth: int


@dataclass
class CircuitStats:
    num_groups: int
    n_qubits: int
    total_gates: int
    total_two_qubit_gates: int
    total_depth: int
    max_group_depth: int
    estimation_time_seconds: float
    group_stats: List[GroupCircuitStats]


def _term_coefficient(term: Any) -> complex:
    coeff = getattr(term, "coeff", None)
    if coeff is not None:
        return coeff

    terms_attr = getattr(term, "terms", None)
    if terms_attr is not None:
        non_identity_terms = [pauli_word for pauli_word in terms_attr.keys() if pauli_word]
        if len(non_identity_terms) == 1:
            return terms_attr[non_identity_terms[0]]

    return 1.0


def _term_to_tequila_paulistring(term: Any) -> PauliString:
    if isinstance(term, PauliString):
        return term

    pauli_ops = _pauli_dict_from_term(term)
    return PauliString(data=pauli_ops, coeff=_term_coefficient(term))


def _group_terms_to_qubit_hamiltonian(group_terms: Sequence[Any]) -> QubitHamiltonian:
    pauli_strings = [_term_to_tequila_paulistring(term) for term in group_terms]
    return QubitHamiltonian.from_paulistrings(pauli_strings)


def _count_two_qubit_gates(circuit: tq.QCircuit) -> int:
    return sum(1 for gate in circuit.gates if len(set(gate.qubits)) == 2)


def _compile_group_with_tequila(
    color: Any,
    group_terms: Sequence[Any],
    include_measurements: bool,
    unitary_circuit: str,
) -> GroupCircuitStats:
    group_hamiltonian = _group_terms_to_qubit_hamiltonian(group_terms)
    binary_group = BinaryHamiltonian.init_from_qubit_hamiltonian(group_hamiltonian)
    if not binary_group.is_commuting():
        raise ValueError(
            "Group {} is not fully commuting, so it cannot be compiled as one Tequila FC group.".format(color)
        )

    compiled_parts, _ = compile_commuting_parts(
        group_hamiltonian,
        unitary_circuit=unitary_circuit,
        options={"method": "si", "condition": "fc"},
    )

    if len(compiled_parts) != 1:
        raise RuntimeError(
            "Tequila split group {} into {} parts. The provided group is not treated as one FC group.".format(
                color,
                len(compiled_parts),
            )
        )

    qwc_hamiltonian, rotation_circuit = compiled_parts[0]
    total_gates = len(rotation_circuit.gates)
    two_qubit_gates = _count_two_qubit_gates(rotation_circuit)
    depth = int(rotation_circuit.depth) if total_gates > 0 else 0
    if include_measurements and len(qwc_hamiltonian.paulistrings) > 0:
        depth += 1

    return GroupCircuitStats(
        color=color,
        n_terms=len(group_terms),
        n_generators=len(qwc_hamiltonian.paulistrings),
        active_qubits=tuple(sorted(int(qubit) for qubit in qwc_hamiltonian.qubits)),
        total_gates=total_gates,
        two_qubit_gates=two_qubit_gates,
        depth=depth,
    )


def grouping_circuit_stats_tequila(
    grouping: Any,
    base_graph: Optional[nx.Graph] = None,
    include_measurements: bool = False,
    unitary_circuit: str = "improved",
) -> CircuitStats:
    """
    Compile each supplied fully-commuting group with Tequila's own grouping compiler.

    Supported `grouping` inputs match `circuit_check.py`:
    - a colored `networkx` compatibility graph
    - a mapping `color -> list[term]`
    - a color tensor/list together with `base_graph`

    The Tequila compilation path is:
    fully-commuting group -> `compile_commuting_parts(..., condition="fc")`
    -> `(qwc_hamiltonian, rotation_circuit)`
    where `rotation_circuit` is the exact basis-change circuit Tequila uses.
    """
    groups = _groups_from_grouping(grouping, base_graph=base_graph)

    start_time = time.perf_counter()
    group_stats = [
        _compile_group_with_tequila(
            color=color,
            group_terms=group_terms,
            include_measurements=include_measurements,
            unitary_circuit=unitary_circuit,
        )
        for color, group_terms in groups.items()
    ]
    total_time = time.perf_counter() - start_time

    all_qubits = sorted({qubit for group in group_stats for qubit in group.active_qubits})
    n_qubits = (max(all_qubits) + 1) if all_qubits else 0

    return CircuitStats(
        num_groups=len(group_stats),
        n_qubits=n_qubits,
        total_gates=sum(group.total_gates for group in group_stats),
        total_two_qubit_gates=sum(group.two_qubit_gates for group in group_stats),
        total_depth=sum(group.depth for group in group_stats),
        max_group_depth=max((group.depth for group in group_stats), default=0),
        estimation_time_seconds=total_time,
        group_stats=group_stats,
    )


def sorted_insertion_circuit_stats_tequila(
    binary_hamiltonian: BinaryHamiltonian,
    condition: str = "fc",
    include_measurements: bool = False,
    unitary_circuit: str = "improved",
) -> CircuitStats:
    """
    Use Tequila's own grouping utility first, then compile each returned group with Tequila.
    """
    commuting_parts, _ = binary_hamiltonian.commuting_groups(
        options={"method": "si", "condition": str(condition).lower()}
    )

    start_time = time.perf_counter()
    group_stats = []
    for color, group in enumerate(commuting_parts):
        qwc_hamiltonian, rotation_circuit = compile_commuting_parts(
            group.to_qubit_hamiltonian(),
            unitary_circuit=unitary_circuit,
            options={"method": "si", "condition": str(condition).lower()},
        )[0][0]

        total_gates = len(rotation_circuit.gates)
        two_qubit_gates = _count_two_qubit_gates(rotation_circuit)
        depth = int(rotation_circuit.depth) if total_gates > 0 else 0
        if include_measurements and len(qwc_hamiltonian.paulistrings) > 0:
            depth += 1

        group_stats.append(
            GroupCircuitStats(
                color=color,
                n_terms=group.n_term,
                n_generators=len(qwc_hamiltonian.paulistrings),
                active_qubits=tuple(sorted(int(qubit) for qubit in qwc_hamiltonian.qubits)),
                total_gates=total_gates,
                two_qubit_gates=two_qubit_gates,
                depth=depth,
            )
        )

    total_time = time.perf_counter() - start_time
    all_qubits = sorted({qubit for group in group_stats for qubit in group.active_qubits})
    n_qubits = (max(all_qubits) + 1) if all_qubits else 0

    return CircuitStats(
        num_groups=len(group_stats),
        n_qubits=n_qubits,
        total_gates=sum(group.total_gates for group in group_stats),
        total_two_qubit_gates=sum(group.two_qubit_gates for group in group_stats),
        total_depth=sum(group.depth for group in group_stats),
        max_group_depth=max((group.depth for group in group_stats), default=0),
        estimation_time_seconds=total_time,
        group_stats=group_stats,
    )


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
    from gflow_vqe.utils import FC_CompMatrix, QWC_CompMatrix, get_terms, max_color, obj_to_comp_graph
    from gflow_vqe.hamiltonians import parse_driver_args

    tequila_unitary_circuit = "improved"  # Change to "original" to test Tequila's older compiler path.

    args = parse_driver_args()
    molecule = args.func
    if molecule is None:
        raise ValueError("Unknown molecule '{}'".format(args.func_name))

    print("Molecule={}".format(args.func_name))
    mol, H, Hferm, n_paulis, Hq = molecule()
    print("Number of Pauli products to measure: {}".format(n_paulis))
    print("Tequila unitary circuit mode={}".format(tequila_unitary_circuit))

    binary_hamiltonian = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
    terms = get_terms(binary_hamiltonian)
    comp_matrix = FC_CompMatrix(terms)
    #comp_matrix = QWC_CompMatrix(terms)
    graph = obj_to_comp_graph(terms, comp_matrix)

    color_map = nx.coloring.greedy_color(graph, strategy="largest_first")
    nx.set_node_attributes(graph, color_map, "color")

    print("Greedy coloring max color={}".format(max_color(graph)))
    color_stats = grouping_circuit_stats_tequila(graph, unitary_circuit=tequila_unitary_circuit)
    si_stats = sorted_insertion_circuit_stats_tequila(
        binary_hamiltonian,
        condition="fc",
        unitary_circuit=tequila_unitary_circuit,
    )

    _print_circuit_stats("Greedy coloring grouping (Tequila compiled)", color_stats)
    _print_circuit_stats("Sorted insertion grouping (fc, Tequila compiled)", si_stats)
    print("Circuit check completed.")


if __name__ == "__main__":
    main()
