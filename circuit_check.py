from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
import time
from typing import Any, Dict, Iterable, List, Mapping, MutableSequence, Optional, Sequence, Tuple

import networkx as nx


_PAULI_LABELS = {"I", "X", "Y", "Z"}


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


class _DepthTracker:
    """Track all-to-all gate depth with the usual no-shared-qubit-per-layer rule."""

    def __init__(self, n_qubits: int):
        self._qubit_depths = [0] * n_qubits

    def add_single(self, qubit: int) -> None:
        self._qubit_depths[qubit] += 1

    def add_two_qubit(self, qubit_a: int, qubit_b: int) -> None:
        layer = max(self._qubit_depths[qubit_a], self._qubit_depths[qubit_b]) + 1
        self._qubit_depths[qubit_a] = layer
        self._qubit_depths[qubit_b] = layer

    @property
    def depth(self) -> int:
        return max(self._qubit_depths, default=0)


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


def _single_pauli_terms_from_group_operator(group_operator: Any) -> List[Tuple[Tuple[int, str], ...]]:
    terms_attr = getattr(group_operator, "terms", None)
    if not isinstance(terms_attr, Mapping):
        raise ValueError("Could not extract Pauli terms from grouped operator '{}'.".format(type(group_operator)))

    single_terms = []
    for pauli_word in terms_attr.keys():
        if not pauli_word:
            continue
        single_terms.append(tuple((int(qubit), str(label).upper()) for qubit, label in pauli_word))
    return single_terms


def sorted_insertion_grouping(binary_hamiltonian: Any, condition: str = "fc") -> Dict[int, List[Tuple[Tuple[int, str], ...]]]:
    """
    Build a color->terms grouping using Tequila's sorted-insertion routine.

    `condition` matches the options used in `SI_results.py`, typically `fc` or `qwc`.
    """
    options = {"method": "si", "condition": str(condition).lower()}
    commuting_parts = binary_hamiltonian.commuting_groups(options=options)[0]

    groups = {}
    for color, group in enumerate(commuting_parts):
        qubit_hamiltonian = group.to_qubit_hamiltonian()
        openfermion_group = qubit_hamiltonian.to_openfermion()
        groups[color] = _single_pauli_terms_from_group_operator(openfermion_group)

    return groups


def _infer_n_qubits(groups: Mapping[Any, Sequence[Any]], n_qubits: Optional[int]) -> int:
    if n_qubits is not None:
        return int(n_qubits)

    max_qubit = -1
    for terms in groups.values():
        for term in terms:
            pauli_ops = _pauli_dict_from_term(term)
            if pauli_ops:
                max_qubit = max(max_qubit, max(pauli_ops))
    return max_qubit + 1 if max_qubit >= 0 else 0


def _row_from_pauli_ops(pauli_ops: Mapping[int, str], n_qubits: int) -> List[int]:
    row = [0] * (2 * n_qubits)
    for qubit, label in pauli_ops.items():
        if label in {"X", "Y"}:
            row[qubit] = 1
        if label in {"Z", "Y"}:
            row[n_qubits + qubit] = 1
    return row


def _symplectic_product(row_a: Sequence[int], row_b: Sequence[int], n_qubits: int) -> int:
    value = 0
    for qubit in range(n_qubits):
        value ^= (row_a[qubit] & row_b[n_qubits + qubit]) ^ (row_a[n_qubits + qubit] & row_b[qubit])
    return value


def _binary_row_basis(rows: Iterable[Sequence[int]]) -> List[List[int]]:
    pivots: Dict[int, List[int]] = {}
    basis: List[List[int]] = []

    for input_row in rows:
        row = list(input_row)
        lead = next((idx for idx, value in enumerate(row) if value), None)
        while lead is not None and lead in pivots:
            pivot_row = pivots[lead]
            row = [lhs ^ rhs for lhs, rhs in zip(row, pivot_row)]
            lead = next((idx for idx, value in enumerate(row) if value), None)
        if lead is None:
            continue
        pivots[lead] = row
        basis.append(row)

    return basis


def _row_support(row: Sequence[int], n_qubits: int) -> List[int]:
    return [qubit for qubit in range(n_qubits) if row[qubit] or row[n_qubits + qubit]]


def _local_pauli_label(row: Sequence[int], qubit: int, n_qubits: int) -> str:
    x_part = row[qubit]
    z_part = row[n_qubits + qubit]
    if x_part and z_part:
        return "Y"
    if x_part:
        return "X"
    if z_part:
        return "Z"
    return "I"


def _xor_row_in_place(target: MutableSequence[int], source: Sequence[int]) -> None:
    for idx, value in enumerate(source):
        target[idx] ^= value


def _apply_h(rows: Sequence[MutableSequence[int]], qubit: int, n_qubits: int) -> None:
    z_idx = n_qubits + qubit
    for row in rows:
        row[qubit], row[z_idx] = row[z_idx], row[qubit]


def _apply_s(rows: Sequence[MutableSequence[int]], qubit: int, n_qubits: int) -> None:
    z_idx = n_qubits + qubit
    for row in rows:
        row[z_idx] ^= row[qubit]


def _apply_cnot(
    rows: Sequence[MutableSequence[int]],
    control: int,
    target: int,
    n_qubits: int,
) -> None:
    control_z = n_qubits + control
    target_z = n_qubits + target
    for row in rows:
        row[target] ^= row[control]
        row[control_z] ^= row[target_z]


def _synthesize_commuting_group(
    color: Any,
    group_terms: Sequence[Any],
    n_qubits: int,
    include_measurements: bool,
    verify_commutation: bool,
) -> GroupCircuitStats:
    pauli_rows = []
    active_qubits = set()
    for term in group_terms:
        pauli_ops = _pauli_dict_from_term(term)
        active_qubits.update(pauli_ops.keys())
        pauli_rows.append(_row_from_pauli_ops(pauli_ops, n_qubits))

    basis_rows = _binary_row_basis(pauli_rows)
    if verify_commutation:
        for idx in range(len(pauli_rows)):
            for jdx in range(idx + 1, len(pauli_rows)):
                if _symplectic_product(pauli_rows[idx], pauli_rows[jdx], n_qubits) != 0:
                    raise ValueError(
                        "Group {} contains non-commuting Pauli terms. The input grouping is not reward-valid.".format(
                            color
                        )
                    )

    if not basis_rows:
        return GroupCircuitStats(
            color=color,
            n_terms=len(group_terms),
            n_generators=0,
            active_qubits=tuple(sorted(active_qubits)),
            total_gates=0,
            two_qubit_gates=0,
            depth=0,
        )

    rows = [row[:] for row in basis_rows]
    used_pivots = set()
    tracker = _DepthTracker(n_qubits)
    single_qubit_gates = 0
    two_qubit_gates = 0
    fixed_rows = 0

    while True:
        candidates = []
        for idx in range(fixed_rows, len(rows)):
            support = _row_support(rows[idx], n_qubits)
            if not support:
                continue
            available = [qubit for qubit in support if qubit not in used_pivots]
            if available:
                candidates.append((len(support), len(available), min(available), idx))

        if not candidates:
            break

        _, _, _, best_idx = min(candidates)
        if best_idx != fixed_rows:
            rows[fixed_rows], rows[best_idx] = rows[best_idx], rows[fixed_rows]

        pivot_row = rows[fixed_rows]
        support = _row_support(pivot_row, n_qubits)
        available_support = [qubit for qubit in support if qubit not in used_pivots]
        if not available_support:
            raise RuntimeError("No available pivot qubit found during group synthesis.")
        pivot = min(available_support)

        for qubit in support:
            local_label = _local_pauli_label(pivot_row, qubit, n_qubits)
            if local_label == "X":
                _apply_h(rows, qubit, n_qubits)
                tracker.add_single(qubit)
                single_qubit_gates += 1
            elif local_label == "Y":
                _apply_s(rows, qubit, n_qubits)
                tracker.add_single(qubit)
                single_qubit_gates += 1
                _apply_h(rows, qubit, n_qubits)
                tracker.add_single(qubit)
                single_qubit_gates += 1

        support = _row_support(pivot_row, n_qubits)
        for qubit in support:
            if qubit == pivot:
                continue
            _apply_cnot(rows, qubit, pivot, n_qubits)
            tracker.add_two_qubit(qubit, pivot)
            two_qubit_gates += 1

        for idx in range(len(rows)):
            if idx == fixed_rows:
                continue
            if rows[idx][n_qubits + pivot]:
                _xor_row_in_place(rows[idx], rows[fixed_rows])

        if verify_commutation:
            if _row_support(rows[fixed_rows], n_qubits) != [pivot]:
                raise RuntimeError("Failed to reduce group {} generator to a single Z pivot.".format(color))
            if _local_pauli_label(rows[fixed_rows], pivot, n_qubits) != "Z":
                raise RuntimeError("Failed to reduce group {} generator to Z on the pivot.".format(color))
            for idx in range(len(rows)):
                if idx == fixed_rows:
                    continue
                if rows[idx][pivot] or rows[idx][n_qubits + pivot]:
                    raise RuntimeError(
                        "Residual pivot support remained on qubit {} while processing group {}.".format(
                            pivot,
                            color,
                        )
                    )

        used_pivots.add(pivot)
        fixed_rows += 1

    depth = tracker.depth + (1 if include_measurements and fixed_rows > 0 else 0)
    return GroupCircuitStats(
        color=color,
        n_terms=len(group_terms),
        n_generators=fixed_rows,
        active_qubits=tuple(sorted(active_qubits)),
        total_gates=single_qubit_gates + two_qubit_gates,
        two_qubit_gates=two_qubit_gates,
        depth=depth,
    )


def grouping_circuit_stats(
    grouping: Any,
    base_graph: Optional[nx.Graph] = None,
    n_qubits: Optional[int] = None,
    include_measurements: bool = False,
    verify_commutation: bool = True,
) -> CircuitStats:
    """
    Estimate measurement-circuit cost for a grouping in the same format used by training.

    Supported inputs:
    - a colored compatibility graph (`networkx.Graph`) with node attributes `color` and `v`
    - a mapping `color -> list[Pauli term]`
    - a color tensor/list from the state-vector training path, together with `base_graph`

    The synthesis model uses a Clifford basis-change circuit built from H/S/CNOT gates
    under an all-to-all connectivity assumption. The reported depth is the sum of the
    per-group basis-change depths, since the grouped measurements are executed as separate
    circuits. `total_two_qubit_gates` is the total CNOT count across all groups.
    """
    groups = _groups_from_grouping(grouping, base_graph=base_graph)
    inferred_n_qubits = _infer_n_qubits(groups, n_qubits=n_qubits)

    start_time = time.perf_counter()
    group_stats = [
        _synthesize_commuting_group(
            color=color,
            group_terms=group_terms,
            n_qubits=inferred_n_qubits,
            include_measurements=include_measurements,
            verify_commutation=verify_commutation,
        )
        for color, group_terms in groups.items()
    ]

    total_estimation_time = time.perf_counter() - start_time

    return CircuitStats(
        num_groups=len(group_stats),
        n_qubits=inferred_n_qubits,
        total_gates=sum(group.total_gates for group in group_stats),
        total_two_qubit_gates=sum(group.two_qubit_gates for group in group_stats),
        total_depth=sum(group.depth for group in group_stats),
        max_group_depth=max((group.depth for group in group_stats), default=0),
        estimation_time_seconds=total_estimation_time,
        group_stats=group_stats,
    )


def circuit_check(
    grouping: Any,
    base_graph: Optional[nx.Graph] = None,
    n_qubits: Optional[int] = None,
    include_measurements: bool = False,
    verify_commutation: bool = True,
) -> CircuitStats:
    """Short alias for grouping_circuit_stats."""
    return grouping_circuit_stats(
        grouping=grouping,
        base_graph=base_graph,
        n_qubits=n_qubits,
        include_measurements=include_measurements,
        verify_commutation=verify_commutation,
    )


def sorted_insertion_circuit_stats(
    binary_hamiltonian: Any,
    condition: str = "fc",
    n_qubits: Optional[int] = None,
    include_measurements: bool = False,
    verify_commutation: bool = True,
) -> CircuitStats:
    """Compute circuit stats for the sorted-insertion grouping used in SI_results.py."""
    grouping = sorted_insertion_grouping(binary_hamiltonian, condition=condition)
    return grouping_circuit_stats(
        grouping=grouping,
        n_qubits=n_qubits,
        include_measurements=include_measurements,
        verify_commutation=verify_commutation,
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
    from tequila.grouping.binary_rep import BinaryHamiltonian

    args = parse_driver_args()
    molecule = args.func
    if molecule is None:
        raise ValueError("Unknown molecule '{}'".format(args.func_name))

    print("Molecule={}".format(args.func_name))
    mol, H, Hferm, n_paulis, Hq = molecule()
    print("Number of Pauli products to measure: {}".format(n_paulis))

    binary_hamiltonian = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
    terms = get_terms(binary_hamiltonian)
    comp_matrix = FC_CompMatrix(terms)
    graph = obj_to_comp_graph(terms, comp_matrix)

    color_map = nx.coloring.greedy_color(graph, strategy="largest_first")
    nx.set_node_attributes(graph, color_map, "color")

    print("Greedy coloring max color={}".format(max_color(graph)))
    color_stats = grouping_circuit_stats(graph)
    si_stats = sorted_insertion_circuit_stats(binary_hamiltonian, condition="fc")

    _print_circuit_stats("Greedy coloring grouping", color_stats)
    _print_circuit_stats("Sorted insertion grouping (fc)", si_stats)

    print("Circuit check completed.")


if __name__ == "__main__":
    main()
