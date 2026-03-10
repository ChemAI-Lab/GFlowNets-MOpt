from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import tequila as tq
from tequila.grouping.binary_rep import BinaryHamiltonian
from tequila.grouping.compile_groups import compile_commuting_parts
from tequila.hamiltonian import PauliString, QubitHamiltonian


_PAULI_LABELS = {"I", "X", "Y", "Z"}
_TEQUILA_UNITARY_CIRCUIT = "improved"


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


def _coerce_color_list(colors: Any) -> List[int]:
    if hasattr(colors, "detach"):
        colors = colors.detach()
    if hasattr(colors, "cpu"):
        colors = colors.cpu()
    if hasattr(colors, "numpy"):
        colors = colors.numpy()
    if hasattr(colors, "tolist"):
        colors = colors.tolist()
    return [int(value) for value in colors]


def _extract_groups_from_graph(graph: nx.Graph) -> Dict[Any, List[Any]]:
    groups: Dict[Any, List[Any]] = defaultdict(list)
    for node, data in graph.nodes(data=True):
        if "color" not in data:
            raise ValueError("Graph node {} is missing the 'color' attribute.".format(node))
        if "v" not in data:
            raise ValueError("Graph node {} is missing the 'v' Pauli-term attribute.".format(node))
        groups[data["color"]].append(data["v"])
    return dict(sorted(groups.items(), key=lambda item: item[0]))


def _groups_from_grouping(grouping: Any, base_graph: Optional[nx.Graph] = None) -> Dict[Any, List[Any]]:
    if isinstance(grouping, nx.Graph):
        return _extract_groups_from_graph(grouping)

    if isinstance(grouping, Mapping):
        return {color: list(terms) for color, terms in grouping.items()}

    if base_graph is None:
        raise ValueError(
            "Non-graph groupings require base_graph. Pass the training-time color tensor/list with base_graph."
        )

    colors = _coerce_color_list(grouping)
    if len(colors) != base_graph.number_of_nodes():
        raise ValueError(
            "Coloring length {} does not match the number of graph nodes {}.".format(
                len(colors),
                base_graph.number_of_nodes(),
            )
        )

    graph = base_graph.copy()
    nx.set_node_attributes(graph, {node: colors[idx] for idx, node in enumerate(graph.nodes)}, "color")
    return _extract_groups_from_graph(graph)


def _normalize_pauli_mapping(mapping: Mapping[Any, Any]) -> Dict[int, str]:
    pauli_ops: Dict[int, str] = {}
    for key, value in mapping.items():
        label = str(value).upper()
        if label == "I":
            continue
        if label not in _PAULI_LABELS:
            raise ValueError("Unsupported Pauli label '{}' in mapping.".format(value))

        if isinstance(key, tuple) and len(key) == 2:
            if isinstance(key[0], int) and isinstance(key[1], str):
                qubit = int(key[0])
                label = key[1].upper()
            elif isinstance(key[1], int) and isinstance(key[0], str):
                qubit = int(key[1])
                label = key[0].upper()
            else:
                raise ValueError("Could not parse mapping key '{}'.".format(key))
        else:
            qubit = int(key)

        if label == "I":
            continue
        if label not in _PAULI_LABELS:
            raise ValueError("Unsupported Pauli label '{}' in mapping.".format(label))
        pauli_ops[qubit] = label
    return pauli_ops


def _parse_pauli_string(text: str) -> Dict[int, str]:
    pauli_ops: Dict[int, str] = {}
    for pattern in (
        r"([XYZ])\((\d+)\)",
        r"([XYZ])\[(\d+)\]",
        r"([XYZ])(\d+)",
        r"(\d+)\s*([XYZ])",
    ):
        matches = re.findall(pattern, text.upper())
        if not matches:
            continue
        for left, right in matches:
            if left in _PAULI_LABELS:
                label, qubit = left, int(right)
            else:
                qubit, label = int(left), right
            pauli_ops[qubit] = label
        if pauli_ops:
            return pauli_ops

    if text.strip() in {"", "I", "[]"}:
        return {}

    raise ValueError("Could not parse Pauli term from '{}'.".format(text))


def _pauli_dict_from_term(term: Any) -> Dict[int, str]:
    if term is None:
        return {}

    if isinstance(term, str):
        return _parse_pauli_string(term)

    if isinstance(term, Mapping):
        return _normalize_pauli_mapping(term)

    if isinstance(term, (tuple, list)):
        if not term:
            return {}
        if all(isinstance(item, (tuple, list)) and len(item) == 2 for item in term):
            return _normalize_pauli_mapping({item[0]: item[1] for item in term})

    terms_attr = getattr(term, "terms", None)
    if isinstance(terms_attr, Mapping):
        non_identity_terms = [pauli_word for pauli_word in terms_attr.keys() if pauli_word]
        if len(non_identity_terms) > 1:
            raise ValueError("Expected a single Pauli word, got {}.".format(len(non_identity_terms)))
        if len(non_identity_terms) == 1:
            return {int(qubit): str(label).upper() for qubit, label in non_identity_terms[0]}
        return {}

    data_attr = getattr(term, "_data", None)
    if isinstance(data_attr, Mapping):
        try:
            return _normalize_pauli_mapping(data_attr)
        except Exception:
            pass

    items_method = getattr(term, "items", None)
    if callable(items_method):
        try:
            return _normalize_pauli_mapping(dict(items_method()))
        except Exception:
            pass

    ops_attr = getattr(term, "ops", None)
    if isinstance(ops_attr, Mapping):
        return _normalize_pauli_mapping(ops_attr)

    paulis_attr = getattr(term, "paulis", None)
    if isinstance(paulis_attr, Mapping):
        return _normalize_pauli_mapping(paulis_attr)

    return _parse_pauli_string(str(term))


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

    to_pauli_strings = getattr(term, "to_pauli_strings", None)
    if callable(to_pauli_strings):
        converted = to_pauli_strings()
        if isinstance(converted, PauliString):
            return converted
        term = converted

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
) -> GroupCircuitStats:
    group_hamiltonian = _group_terms_to_qubit_hamiltonian(group_terms)
    binary_group = BinaryHamiltonian.init_from_qubit_hamiltonian(group_hamiltonian)
    if not binary_group.is_commuting():
        raise ValueError(
            "Group {} is not fully commuting, so it cannot be compiled as one Tequila FC group.".format(color)
        )

    compiled_parts, _ = compile_commuting_parts(
        group_hamiltonian,
        unitary_circuit=_TEQUILA_UNITARY_CIRCUIT,
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
) -> CircuitStats:
    groups = _groups_from_grouping(grouping, base_graph=base_graph)

    start_time = time.perf_counter()
    group_stats = [
        _compile_group_with_tequila(
            color=color,
            group_terms=group_terms,
            include_measurements=include_measurements,
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
) -> CircuitStats:
    commuting_parts, _ = binary_hamiltonian.commuting_groups(
        options={"method": "si", "condition": str(condition).lower()}
    )

    start_time = time.perf_counter()
    group_stats = []
    for color, group in enumerate(commuting_parts):
        qwc_hamiltonian, rotation_circuit = compile_commuting_parts(
            group.to_qubit_hamiltonian(),
            unitary_circuit=_TEQUILA_UNITARY_CIRCUIT,
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


def get_groups_2qgates(graph: nx.Graph) -> int:
    """Return the total number of two-qubit gates Tequila uses for the graph's current grouping."""
    return grouping_circuit_stats_tequila(graph).total_two_qubit_gates
