"""Scalable ordered-graph SI/ICS analysis for cached Pareto metrics.

The v2 path keeps the standard Tequila sorted-insertion partition, resolves
ordered graph nodes through their embedded Pauli metadata, plans exactly the
covariance pairs that SI/ICS can query, and evaluates only those moments.  It
also deduplicates equivalent graph partitions and can process independent ICS
optimizations in parallel.
"""

import argparse
import hashlib
import math
import os
import pickle
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations_with_replacement

import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import tequila as tq
from openfermion import QubitOperator, get_sparse_operator
from openfermion.utils import count_qubits
from tequila.grouping.binary_rep import BinaryHamiltonian
from tequila.hamiltonian import QubitHamiltonian
from threadpoolctl import threadpool_limits

import gflow_vqe.hamiltonians as hamlib
from gflow_vqe.circuit_helpers import grouping_circuit_stats_tequila
from gflow_vqe.overlapping_helpers import (
    as_tequila_wavefunction,
    extract_measurable_terms,
    get_opt_sample_size,
    OverlappingAuxiliary,
    OverlappingGroups,
    iterative_coefficient_splitting_from_groups,
)
from gflow_vqe.utils import get_variance_wavefunction


DEFAULT_COVARIANCE_CHUNKSIZE = 128
DEFAULT_CACHE_DIRECTORY = ".pareto_fast_cache"
V2_CACHE_VERSION = 2


@dataclass(frozen=True)
class PauliTerm:
    index: int
    pauli_tuple: tuple[tuple[int, str], ...]
    ops: tuple[str, ...]
    coefficient: complex
    word: str
    source_order: int


@dataclass
class ICSPlan:
    signature: str
    overlapping: object


class IndexedCovariance:
    """Sparse-in-use covariance matrix addressed by BinaryPauliString keys."""

    def __init__(self, binary_terms, required_pairs, values):
        self.binary_terms = list(binary_terms)
        self.position_by_key = {
            term.binary_tuple(): position
            for position, term in enumerate(self.binary_terms)
        }
        n_terms = len(self.binary_terms)
        self.values = np.zeros((n_terms, n_terms), dtype=complex)
        self.present = np.zeros((n_terms, n_terms), dtype=bool)
        for (left, right), covariance in zip(required_pairs, values):
            self.values[left, right] = covariance
            self.values[right, left] = covariance
            self.present[left, right] = True
            self.present[right, left] = True
        self.n_entries = len(required_pairs)

    def _positions(self, pair):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise KeyError(pair)
        return self.position_by_key[pair[0]], self.position_by_key[pair[1]]

    def __contains__(self, pair):
        try:
            left, right = self._positions(pair)
        except (KeyError, TypeError):
            return False
        return bool(self.present[left, right])

    def __getitem__(self, pair):
        left, right = self._positions(pair)
        if not self.present[left, right]:
            raise KeyError(pair)
        return self.values[left, right]

    def __len__(self):
        return self.n_entries


class SparseTequilaReporter:
    """Evaluate group variances with Tequila's sparse-state semantics.

    The legacy reporting path first obtains the sample allocation from the
    covariance dictionary, then evaluates ``H_g`` and ``H_g * H_g`` on a
    Tequila wavefunction.  ``QubitWaveFunction.apply_qubitoperator`` removes
    output amplitudes that are close to zero at an absolute tolerance of
    ``1e-8``.  Consequently, reconstructing a group variance only from the
    covariance matrix can differ from the value printed by the legacy script.

    Only output basis states present in the reference wavefunction can
    contribute to its inner product.  This class applies each Pauli word on
    that sparse support, reproduces Tequila's component-wise output cutoff,
    and avoids constructing dense action rows or rescanning the dense
    statevector for every graph.
    """

    OUTPUT_THRESHOLD = 1.0e-8
    DEFAULT_ACTION_CACHE_BYTES = 64 * 1024**2

    def __init__(self, wfn, action_cache_bytes=DEFAULT_ACTION_CACHE_BYTES):
        self.wavefunction = as_tequila_wavefunction(wfn)
        dense_state = np.asarray(self.wavefunction.to_array(), dtype=complex).reshape(-1)
        dimension = int(dense_state.size)
        n_qubits = dimension.bit_length() - 1
        if 2**n_qubits != dimension:
            raise ValueError(
                "Expected a power-of-two reporting state dimension, got {}.".format(
                    dimension
                )
            )
        if n_qubits > 63:
            raise ValueError(
                "Sparse Tequila reporting supports at most 63 qubits, got {}.".format(
                    n_qubits
                )
            )

        self.n_qubits = n_qubits
        self.dimension = dimension
        self.support = np.flatnonzero(dense_state).astype(np.uint64)
        self.values = dense_state[self.support]
        self.position_by_basis = np.full(dimension, -1, dtype=np.int32)
        self.position_by_basis[self.support] = np.arange(
            len(self.support),
            dtype=np.int32,
        )
        self.byte_parity = np.asarray(
            [value.bit_count() % 2 for value in range(256)],
            dtype=np.uint8,
        )
        self.action_cache_bytes = max(0, int(action_cache_bytes))
        self.action_cache_bytes_used = 0
        self.action_cache = OrderedDict()
        self.mask_cache = {}

    def _masks_for_openfermion_term(self, pauli_tuple):
        pauli_tuple = tuple(pauli_tuple)
        cached = self.mask_cache.get(pauli_tuple)
        if cached is not None:
            return cached

        x_mask = 0
        z_mask = 0
        for qubit, pauli in pauli_tuple:
            array_bit = 1 << (self.n_qubits - 1 - int(qubit))
            pauli = str(pauli).upper()
            if pauli in ("X", "Y"):
                x_mask |= array_bit
            if pauli in ("Z", "Y"):
                z_mask |= array_bit
        masks = (x_mask, z_mask)
        self.mask_cache[pauli_tuple] = masks
        return masks

    def _restricted_pauli_action(self, x_mask, z_mask):
        key = (int(x_mask), int(z_mask))
        cached = self.action_cache.get(key)
        if cached is not None:
            self.action_cache.move_to_end(key)
            return cached

        destinations = np.bitwise_xor(self.support, np.uint64(x_mask))
        destination_positions = self.position_by_basis[destinations]
        source_positions = np.flatnonzero(destination_positions >= 0)
        destination_positions = destination_positions[source_positions]

        if source_positions.size:
            masked = np.bitwise_and(
                self.support[source_positions],
                np.uint64(z_mask),
            )
            parity = np.zeros(masked.shape, dtype=np.uint8)
            for shift in range(0, self.n_qubits, 8):
                parity ^= self.byte_parity[
                    (masked >> np.uint64(shift)) & np.uint64(0xFF)
                ]
            signs = 1.0 - 2.0 * parity.astype(float)
            phase = (1.0, 1.0j, -1.0, -1.0j)[
                (x_mask & z_mask).bit_count() % 4
            ]
            action_values = phase * signs * self.values[source_positions]
        else:
            action_values = np.empty(0, dtype=complex)

        result = (
            np.asarray(destination_positions, dtype=np.int32),
            np.asarray(action_values, dtype=complex),
        )
        entry_bytes = result[0].nbytes + result[1].nbytes
        if self.action_cache_bytes and entry_bytes <= self.action_cache_bytes:
            while (
                self.action_cache
                and self.action_cache_bytes_used + entry_bytes
                > self.action_cache_bytes
            ):
                _, evicted = self.action_cache.popitem(last=False)
                self.action_cache_bytes_used -= (
                    evicted[0].nbytes + evicted[1].nbytes
                )
            self.action_cache[key] = result
            self.action_cache_bytes_used += entry_bytes
        return result

    def clear_action_cache(self):
        self.action_cache.clear()
        self.action_cache_bytes_used = 0

    def operator_inner(self, operator):
        """Return ``<phi|operator|phi>`` after Tequila's output cutoff."""

        output = np.zeros(len(self.support), dtype=complex)
        for pauli_tuple, coefficient in operator.qubit_operator.terms.items():
            x_mask, z_mask = self._masks_for_openfermion_term(pauli_tuple)
            destination_positions, action_values = self._restricted_pauli_action(
                x_mask,
                z_mask,
            )
            if destination_positions.size:
                # A Pauli word permutes basis states, so destination_positions
                # contains no duplicate within this update.  Iterating terms in
                # OpenFermion insertion order preserves Tequila's accumulation
                # order across Pauli words.
                output[destination_positions] += complex(coefficient) * action_values

        retained = np.flatnonzero(
            ~np.isclose(output, 0.0, atol=self.OUTPUT_THRESHOLD)
        )
        result = 0.0
        for position in retained:
            result += self.values[position].conjugate() * output[position]
        return result

    def group_variance(self, group):
        operator = group.to_qubit_hamiltonian()
        mean = self.operator_inner(operator)
        second_moment = self.operator_inner(operator * operator)
        return second_moment - mean**2

    def optimal_allocation_metric(self, groups, sample_size, tiny=1.0e-12):
        measurement = 0.0
        for index, group in enumerate(groups):
            variance = self.group_variance(group)
            if hasattr(variance, "imag") and abs(variance.imag) < tiny:
                variance = variance.real
            measurement += variance / sample_size[index]
        return float(_to_real_if_close(measurement, tiny=tiny))


_ACTION_STATE = None
_ACTION_N_QUBITS = None
_ACTION_TERMS = None
_MOMENT_LEFT_STATE = None
_MOMENT_RIGHT_STATE = None
_MOMENT_SUPPORT = None
_MOMENT_RIGHT_VALUES = None
_MOMENT_BYTE_PARITY = None
_MOMENT_N_QUBITS = None
_MOMENT_THREAD_LIMITER = None
_ICS_PLANS = None
_ICS_COVARIANCE = None
_ICS_REPORTER = None
_ICS_THREAD_LIMITER = None


def default_cov_workers():
    return max(1, min(8, os.cpu_count() or 1))


def default_ics_workers():
    return max(1, min(32, os.cpu_count() or 1))


def clean_complex(value, tiny=1.0e-12):
    value = complex(value)
    # Small real components can contribute to the Greedy objective when many
    # covariances are accumulated.  Preserve them; only discard imaginary
    # roundoff from quantities that should be real.
    real = value.real
    imag = 0.0 if abs(value.imag) < tiny else value.imag
    return complex(real, imag)


def canonical_pair(left, right):
    left = int(left)
    right = int(right)
    return (left, right) if left <= right else (right, left)


def ordered_measurable_terms(binary_hamiltonian):
    """Match the coefficient ordering used by driver_sv_ordered.py."""

    return sorted(
        extract_measurable_terms(binary_hamiltonian),
        key=lambda term: abs(term.get_coeff()),
        reverse=True,
    )


def binary_key_to_masks(binary_key, n_qubits):
    x_mask = 0
    z_mask = 0
    for qubit in range(n_qubits):
        array_bit = 1 << (n_qubits - 1 - qubit)
        if binary_key[qubit]:
            x_mask |= array_bit
        if binary_key[n_qubits + qubit]:
            z_mask |= array_bit
    return x_mask, z_mask


def pauli_product(left_masks, right_masks):
    """Return masks and phase for P(left) P(right)."""

    left_x, left_z = left_masks
    right_x, right_z = right_masks
    product_x = left_x ^ right_x
    product_z = left_z ^ right_z
    left_y = (left_x & left_z).bit_count()
    right_y = (right_x & right_z).bit_count()
    product_y = (product_x & product_z).bit_count()
    exponent = (left_y + right_y - product_y) % 4
    phase = (1.0, 1.0j, -1.0, -1.0j)[exponent]
    if (left_z & right_x).bit_count() % 2:
        phase = -phase
    return (product_x, product_z), phase


def build_compatibility_matrix(binary_terms, condition):
    binary = np.asarray([term.get_binary() for term in binary_terms], dtype=np.uint8)
    n_qubits = binary.shape[1] // 2
    x_bits = binary[:, :n_qubits]
    z_bits = binary[:, n_qubits:]

    if condition == "fc":
        symplectic = (
            x_bits.astype(np.int16).dot(z_bits.astype(np.int16).T)
            + z_bits.astype(np.int16).dot(x_bits.astype(np.int16).T)
        )
        return (symplectic % 2) == 0
    if condition == "qwc":
        operations = x_bits + 2 * z_bits
        compatible = np.ones((len(binary_terms), len(binary_terms)), dtype=bool)
        for qubit in range(n_qubits):
            left = operations[:, qubit][:, None]
            right = operations[:, qubit][None, :]
            compatible &= (left == 0) | (right == 0) | (left == right)
        return compatible
    raise ValueError("Unknown grouping condition '{}'.".format(condition))


def _normalized_term_groups(groups):
    normalized = []
    for group in groups:
        terms = list(group.binary_terms) if isinstance(group, BinaryHamiltonian) else list(group)
        terms = [term for term in terms if np.any(term.get_binary())]
        if terms:
            normalized.append(terms)
    if not normalized:
        raise ValueError("No measurable terms were found in the provided groups.")
    return normalized


def pairs_within_groups(groups, position_by_key):
    required = set()
    for group in _normalized_term_groups(groups):
        positions = sorted({position_by_key[term.binary_tuple()] for term in group})
        required.update(combinations_with_replacement(positions, 2))
    return required


def build_fast_ics_plan(
    initial_groups,
    condition,
    position_by_key,
    binary_terms,
    compatibility,
    signature,
):
    """Construct the repository's ICS membership model using cached compatibility."""

    normalized = _normalized_term_groups(initial_groups)
    initial_positions = []
    seen_positions = set()
    unique_terms = []
    for group in normalized:
        positions = []
        for term in group:
            position = position_by_key[term.binary_tuple()]
            positions.append(position)
            if position not in seen_positions:
                seen_positions.add(position)
                unique_terms.append(term)
        positions = list(dict.fromkeys(positions))
        if not np.all(compatibility[np.ix_(positions, positions)]):
            raise ValueError("Initial grouping contains incompatible Pauli terms.")
        initial_positions.append(positions)

    if seen_positions != set(range(len(binary_terms))):
        missing = sorted(set(range(len(binary_terms))) - seen_positions)
        extra = sorted(seen_positions - set(range(len(binary_terms))))
        raise ValueError(
            "Initial groups do not match the measurable Hamiltonian terms. "
            "Missing positions={}, Extra positions={}.".format(missing, extra)
        )

    sorted_terms = sorted(unique_terms, key=lambda term: abs(term.get_coeff()), reverse=True)
    newly_added = [[] for _ in normalized]
    overlapping_terms = []
    term_exists_in = []
    for term in sorted_terms:
        position = position_by_key[term.binary_tuple()]
        group_indices = []
        for group_index, base_positions in enumerate(initial_positions):
            added_positions = newly_added[group_index]
            if not np.all(compatibility[position, base_positions]):
                continue
            if added_positions and not np.all(compatibility[position, added_positions]):
                continue
            group_indices.append(group_index)
            added_positions.append(position)
        if len(group_indices) > 1:
            overlapping_terms.append(term.term_w_coeff(0.0))
            term_exists_in.append(group_indices)

    overlapping = OverlappingGroups(
        normalized,
        overlapping_terms,
        term_exists_in,
    )
    required_pairs = set()
    for base_group, overlap_group in zip(overlapping.no_groups, overlapping.o_groups):
        positions = sorted(
            {
                position_by_key[term.binary_tuple()]
                for term in list(base_group) + list(overlap_group)
            }
        )
        required_pairs.update(combinations_with_replacement(positions, 2))
    return ICSPlan(signature=signature, overlapping=overlapping), required_pairs


def ics_input_signature(groups, position_by_key):
    """Hash every ordering detail that can affect stable coefficient ties.

    Numeric color labels are deliberately absent, but the group order induced
    by those labels and the term order inside each group are retained.  The
    repository's ICS construction uses stable sorting for equal-magnitude
    coefficients, so a fully label-invariant partition hash would be unsafe.
    """

    ordered_groups = [
        tuple(position_by_key[term.binary_tuple()] for term in group)
        for group in _normalized_term_groups(groups)
    ]
    digest = hashlib.sha256()
    for group in ordered_groups:
        digest.update(np.asarray(group, dtype=np.int32).tobytes())
        digest.update(b"|")
    return digest.hexdigest()


def resolve_ordered_graph_groups(
    graph,
    terms_by_support,
    expected_supports,
    condition,
    position_by_key=None,
    compatibility=None,
):
    records = []
    for node, data in graph.nodes(data=True):
        if "color" not in data:
            raise ValueError("Graph node {} is missing the 'color' attribute.".format(node))
        if "v" not in data:
            raise ValueError(
                "Graph node {} is missing the embedded Pauli-term attribute 'v'. "
                "Regenerate it with driver_sv_ordered.py.".format(node)
            )
        records.append((data["color"], _pauli_support_key(data["v"])))

    supports = [support for _, support in records]
    if len(supports) != len(set(supports)):
        raise ValueError("The ordered graph contains at least one duplicate Pauli term.")
    support_set = set(supports)
    if support_set != expected_supports:
        missing = sorted(expected_supports - support_set)
        extra = sorted(support_set - expected_supports)
        raise ValueError(
            "Ordered graph terms do not match the current Hamiltonian. "
            "Missing={}, Extra={}.".format(missing, extra)
        )

    color_to_terms = {}
    for color, support in records:
        color_to_terms.setdefault(color, []).append(terms_by_support[support])
    groups = [color_to_terms[color] for color in sorted(color_to_terms)]

    if (position_by_key is None) != (compatibility is None):
        raise ValueError(
            "position_by_key and compatibility must either both be provided or both omitted."
        )

    for color, group in zip(sorted(color_to_terms), groups):
        if compatibility is not None:
            positions = [
                position_by_key[term.binary_tuple()]
                for term in group
            ]
            group_is_compatible = bool(
                np.all(compatibility[np.ix_(positions, positions)])
            )
        else:
            group_is_compatible = all(
                (
                    left.qubit_wise_commute(right)
                    if condition == "qwc"
                    else left.commute(right)
                )
                for left_index, left in enumerate(group)
                for right in group[left_index + 1:]
            )
        if not group_is_compatible:
            raise ValueError(
                "Ordered graph color {} contains terms that are not {} compatible.".format(
                    color,
                    condition.upper(),
                )
            )
    return groups


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


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value, got '{}'.".format(value))


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


def hamiltonian_state_fingerprint(binary_terms, state_vector, n_qubits, wfn_method):
    digest = hashlib.sha256()
    digest.update("pareto-fast-v{}".format(V2_CACHE_VERSION).encode("ascii"))
    digest.update(str(int(n_qubits)).encode("ascii"))
    digest.update(str(wfn_method).upper().encode("ascii"))
    for term in binary_terms:
        digest.update(np.asarray(term.get_binary(), dtype=np.uint8).tobytes())
        digest.update(np.asarray(complex(term.get_coeff()), dtype=np.complex128).tobytes())
    state = np.ascontiguousarray(np.asarray(state_vector, dtype=np.complex128).reshape(-1))
    digest.update(state.view(np.uint8).tobytes())
    return digest.hexdigest()


def _cache_path(cache_dir, func_name, wfn_method, suffix):
    safe_name = "{}_{}_{}".format(func_name, str(wfn_method).lower(), suffix)
    return os.path.join(cache_dir, safe_name)


def load_versioned_cache(path, fingerprint, rebuild=False):
    if rebuild or not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
    except Exception as error:
        print("Ignoring unreadable cache '{}': {}".format(path, error), flush=True)
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != V2_CACHE_VERSION:
        return None
    if payload.get("fingerprint") != fingerprint:
        print("Ignoring stale cache '{}' (fingerprint mismatch).".format(path), flush=True)
        return None
    return payload


def save_versioned_cache(path, fingerprint, **values):
    if not path:
        return
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    payload = {
        "version": V2_CACHE_VERSION,
        "fingerprint": fingerprint,
    }
    payload.update(values)
    temporary_path = "{}.tmp.{}".format(path, os.getpid())
    with open(temporary_path, "wb") as handle:
        pickle.dump(payload, handle, pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_path, path)


def _init_sparse_moment_worker(state_vector, truncated_state, support, n_qubits, blas_threads):
    global _MOMENT_LEFT_STATE
    global _MOMENT_RIGHT_STATE
    global _MOMENT_SUPPORT
    global _MOMENT_RIGHT_VALUES
    global _MOMENT_BYTE_PARITY
    global _MOMENT_N_QUBITS
    global _MOMENT_THREAD_LIMITER

    _MOMENT_LEFT_STATE = np.asarray(state_vector, dtype=complex).reshape(-1)
    _MOMENT_RIGHT_STATE = np.asarray(truncated_state, dtype=complex).reshape(-1)
    _MOMENT_SUPPORT = np.asarray(support, dtype=np.uint64)
    _MOMENT_RIGHT_VALUES = _MOMENT_RIGHT_STATE[_MOMENT_SUPPORT]
    _MOMENT_BYTE_PARITY = np.asarray(
        [value.bit_count() % 2 for value in range(256)],
        dtype=np.uint8,
    )
    _MOMENT_N_QUBITS = int(n_qubits)
    _MOMENT_THREAD_LIMITER = threadpool_limits(limits=int(blas_threads))


def _support_parity(mask):
    masked = np.bitwise_and(_MOMENT_SUPPORT, np.uint64(mask))
    parity = np.zeros(masked.shape, dtype=np.uint8)
    for shift in range(0, _MOMENT_N_QUBITS, 8):
        parity ^= _MOMENT_BYTE_PARITY[(masked >> np.uint64(shift)) & np.uint64(0xFF)]
    return parity


def _sparse_pauli_moment(kind, x_mask, z_mask):
    destination = np.bitwise_xor(_MOMENT_SUPPORT, np.uint64(x_mask))
    parity = _support_parity(z_mask)
    signs = 1.0 - 2.0 * parity.astype(float)
    left_state = _MOMENT_LEFT_STATE if kind == "single" else _MOMENT_RIGHT_STATE
    value = np.dot(
        np.conjugate(left_state[destination]),
        signs * _MOMENT_RIGHT_VALUES,
    )
    phase = (1.0, 1.0j, -1.0, -1.0j)[(x_mask & z_mask).bit_count() % 4]
    return clean_complex(phase * value)


def _sparse_moment_chunk(entries):
    return [
        (kind, x_mask, z_mask, _sparse_pauli_moment(kind, x_mask, z_mask))
        for kind, x_mask, z_mask in entries
    ]


def evaluate_sparse_moments(
    state_vector,
    n_qubits,
    missing_single_masks,
    missing_product_masks,
    requested_workers,
    blas_threads,
):
    state_vector = np.asarray(state_vector, dtype=complex).reshape(-1)
    dimension = 2**n_qubits
    if state_vector.size != dimension:
        raise ValueError(
            "Expected statevector size {}, got {}.".format(dimension, state_vector.size)
        )

    thresholded_wfn = tequila_wavefunction_from_array(state_vector)
    truncated_state = wavefunction_array(thresholded_wfn, dimension)
    support = np.flatnonzero(truncated_state).astype(np.uint64)
    discarded_norm_sq = float(
        np.vdot(state_vector - truncated_state, state_vector - truncated_state).real
    )
    print(
        "Pauli-moment state support={}/{}; discarded_norm_sq={:.3e}.".format(
            len(support),
            dimension,
            discarded_norm_sq,
        ),
        flush=True,
    )

    entries = [
        ("single", int(x_mask), int(z_mask))
        for x_mask, z_mask in sorted(missing_single_masks)
    ]
    entries.extend(
        ("product", int(x_mask), int(z_mask))
        for x_mask, z_mask in sorted(missing_product_masks)
    )
    if not entries:
        return {}, {}

    work = len(entries) * max(1, len(support))
    useful_workers = max(1, math.ceil(work / 30_000_000))
    effective_workers = min(
        int(requested_workers),
        32,
        useful_workers,
        len(entries),
    )
    chunk_size = max(1, math.ceil(len(entries) / (8 * effective_workers)))
    chunks = [
        entries[start:start + chunk_size]
        for start in range(0, len(entries), chunk_size)
    ]
    print(
        "Evaluating {} missing Pauli moments with {} worker(s) "
        "(requested {}, chunk_size={}).".format(
            len(entries),
            effective_workers,
            requested_workers,
            chunk_size,
        ),
        flush=True,
    )

    initializer_args = (
        state_vector,
        truncated_state,
        support,
        n_qubits,
        blas_threads,
    )
    rows = []
    if effective_workers == 1:
        _init_sparse_moment_worker(*initializer_args)
        for chunk in chunks:
            rows.extend(_sparse_moment_chunk(chunk))
    else:
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            initializer=_init_sparse_moment_worker,
            initargs=initializer_args,
        ) as executor:
            for chunk_rows in executor.map(_sparse_moment_chunk, chunks):
                rows.extend(chunk_rows)

    single_moments = {}
    product_moments = {}
    for kind, x_mask, z_mask, value in rows:
        target = single_moments if kind == "single" else product_moments
        target[(x_mask, z_mask)] = value
    return single_moments, product_moments


def build_covariance_requests(binary_terms, required_pairs, n_qubits):
    term_masks = [
        binary_key_to_masks(term.binary_tuple(), n_qubits)
        for term in binary_terms
    ]
    single_masks = set()
    product_masks = set()
    pair_products = []
    for left, right in required_pairs:
        single_masks.add(term_masks[left])
        single_masks.add(term_masks[right])
        product_masks.add(pauli_product(term_masks[left], term_masks[right])[0])
        pair_products.append(pauli_product(term_masks[left], term_masks[right]))
    return term_masks, pair_products, single_masks, product_masks


def build_selected_covariance(
    binary_terms,
    required_pairs,
    n_qubits,
    state_vector,
    requested_workers,
    blas_threads,
    moment_cache_path=None,
    fingerprint=None,
    rebuild_cache=False,
):
    required_pairs = sorted({canonical_pair(*pair) for pair in required_pairs})
    term_masks, pair_products, single_masks, product_masks = build_covariance_requests(
        binary_terms,
        required_pairs,
        n_qubits,
    )

    cached = load_versioned_cache(
        moment_cache_path,
        fingerprint,
        rebuild=rebuild_cache,
    )
    single_moments = dict(cached.get("single_moments", {})) if cached else {}
    product_moments = dict(cached.get("product_moments", {})) if cached else {}
    missing_single = single_masks - set(single_moments)
    missing_product = product_masks - set(product_moments)
    print(
        "Planned covariance pairs={}; unique singles={}; unique products={}; "
        "cached moments={}.".format(
            len(required_pairs),
            len(single_masks),
            len(product_masks),
            len(single_masks) + len(product_masks) - len(missing_single) - len(missing_product),
        ),
        flush=True,
    )

    if missing_single or missing_product:
        new_single, new_product = evaluate_sparse_moments(
            state_vector,
            n_qubits,
            missing_single,
            missing_product,
            requested_workers,
            blas_threads,
        )
    else:
        # A complete cache hit must not rebuild Tequila's sparse wavefunction
        # or scan the full 2**n statevector merely to discover there is no work.
        new_single, new_product = {}, {}
    single_moments.update(new_single)
    product_moments.update(new_product)
    if moment_cache_path and (cached is None or new_single or new_product):
        save_versioned_cache(
            moment_cache_path,
            fingerprint,
            single_moments=single_moments,
            product_moments=product_moments,
        )

    covariance_values = []
    for (left, right), (product_masks_value, product_phase) in zip(
        required_pairs,
        pair_products,
    ):
        covariance = clean_complex(
            product_phase * product_moments[product_masks_value]
            - single_moments[term_masks[left]] * single_moments[term_masks[right]]
        )
        covariance_values.append(covariance)
    return IndexedCovariance(binary_terms, required_pairs, covariance_values)


def indexed_covariance_from_legacy(binary_terms, required_pairs, legacy_covariance):
    values = []
    required_pairs = sorted({canonical_pair(*pair) for pair in required_pairs})
    for left, right in required_pairs:
        left_key = binary_terms[left].binary_tuple()
        right_key = binary_terms[right].binary_tuple()
        if (left_key, right_key) in legacy_covariance:
            value = legacy_covariance[(left_key, right_key)]
        elif (right_key, left_key) in legacy_covariance:
            value = legacy_covariance[(right_key, left_key)]
        else:
            raise KeyError(
                "Legacy covariance is missing planned term positions {} and {}.".format(
                    left,
                    right,
                )
            )
        values.append(value)
    return IndexedCovariance(binary_terms, required_pairs, values)


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
        "--cov-workers",
        type=int,
        default=default_cov_workers(),
        help=(
            "Requested worker processes for Pauli moments (default: up to 8). "
            "Moment evaluation adaptively uses no more than 32."
        ),
    )
    parser.add_argument(
        "--ics-workers",
        type=int,
        default=None,
        help=(
            "Worker processes used for independent graph ICS calculations. "
            "Defaults to up to 32 independently of --cov-workers; an explicit "
            "value is honored."
        ),
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=1,
        help="BLAS threads per ICS worker (default: 1, avoiding nested oversubscription).",
    )
    parser.add_argument(
        "--cov-backend",
        choices=("moments", "legacy"),
        default="moments",
        help=(
            "Covariance backend. 'moments' evaluates only planned Pauli products; "
            "'legacy' forms the full action/Gram matrix for comparison."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIRECTORY,
        help=(
            "Directory for reusable Pauli-moment and graph-ICS caches "
            "(default: .pareto_fast_cache)."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reading and writing v2 moment/ICS caches.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore compatible v2 cache entries and recompute them.",
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
        "--qwc",
        action="store_true",
        help="Use qubit-wise commuting groupings for SI/ICS instead of fully commuting groupings.",
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
    if args.cov_workers < 1:
        parser.error("--cov-workers must be at least 1.")
    if args.ics_workers is None:
        args.ics_workers = default_ics_workers()
    if args.ics_workers < 1:
        parser.error("--ics-workers must be at least 1.")
    if args.blas_threads < 1:
        parser.error("--blas-threads must be at least 1.")
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


def fast_color_reward(graph):
    """Equivalent color reward with work proportional to within-color pairs."""

    color_to_nodes = {}
    for node, data in graph.nodes(data=True):
        if "color" not in data:
            raise ValueError("Graph node {} is missing the 'color' attribute.".format(node))
        color_to_nodes.setdefault(data["color"], []).append(node)
    if not color_to_nodes:
        return 0
    for nodes in color_to_nodes.values():
        for node in nodes:
            if graph.has_edge(node, node):
                return 0
        for left_index, left in enumerate(nodes):
            for right in nodes[left_index + 1:]:
                if graph.has_edge(left, right):
                    return 0
    return graph.number_of_nodes() - len(color_to_nodes)


def load_sampled_graphs(sampled_graphs_path):
    if not os.path.exists(sampled_graphs_path):
        raise FileNotFoundError("Could not find sampled graphs file '{}'.".format(sampled_graphs_path))

    with open(sampled_graphs_path, "rb") as handle:
        sampled_graphs = pickle.load(handle)

    if not isinstance(sampled_graphs, list):
        sampled_graphs = list(sampled_graphs)

    valid_graphs = [graph for graph in sampled_graphs if fast_color_reward(graph) > 0]
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


def compute_group_metrics_indexed(groups, covariances, reporter):
    normalized_groups = normalize_binary_groups(groups)
    # Preserve the legacy allocation accumulation order.  The selected indexed
    # covariance object implements the same mapping protocol as the tuple-keyed
    # dictionary, so this remains cheap and queries only planned pairs.
    sample_size = get_opt_sample_size(
        [group.binary_terms for group in normalized_groups],
        covariances,
    )
    # Reporting is deliberately not reconstructed as (sum sqrt(V_g))**2.
    # Tequila applies a component-wise 1e-8 cutoff to H_g|phi> and H_g**2|phi>,
    # which can change the printed metric even when covariance values agree to
    # machine precision.  SparseTequilaReporter reproduces that route.
    measurement = reporter.optimal_allocation_metric(
        normalized_groups,
        sample_size,
    )
    group_mapping = {
        index: list(group.binary_terms)
        for index, group in enumerate(normalized_groups)
    }
    two_qubit_gates = int(
        grouping_circuit_stats_tequila(group_mapping).total_two_qubit_gates
    )
    return {
        "measurement": float(measurement),
        "num_groups": len(normalized_groups),
        "two_qubit_gates": two_qubit_gates,
        "sample_size": sample_size,
        "groups": normalized_groups,
    }


def compact_metric(metric_dict):
    return {
        "measurement": float(metric_dict["measurement"]),
        "num_groups": int(metric_dict["num_groups"]),
        "two_qubit_gates": int(metric_dict["two_qubit_gates"]),
    }


def _init_ics_worker(plans, covariances, wfn, blas_threads):
    global _ICS_PLANS
    global _ICS_COVARIANCE
    global _ICS_REPORTER
    global _ICS_THREAD_LIMITER
    _ICS_PLANS = plans
    _ICS_COVARIANCE = covariances
    # Build the sparse reporting state and its basis-position lookup once per
    # process.  Reconstructing either for every graph would rescan 2**n states.
    _ICS_REPORTER = SparseTequilaReporter(wfn)
    _ICS_THREAD_LIMITER = threadpool_limits(limits=int(blas_threads))


def _run_ics_plan(signature):
    plan = _ICS_PLANS[signature]
    groups = plan.overlapping.optimal_overlapping_groups(
        OverlappingAuxiliary(_ICS_COVARIANCE)
    )
    binary_groups = [BinaryHamiltonian(group) for group in groups]
    metrics = compute_group_metrics_indexed(
        binary_groups,
        _ICS_COVARIANCE,
        reporter=_ICS_REPORTER,
    )
    return signature, compact_metric(metrics)


def run_ics_plans(plans, covariances, wfn, max_workers, blas_threads):
    if not plans:
        return {}
    signatures = list(plans)
    effective_workers = min(int(max_workers), len(signatures))
    print(
        "Running {} unique graph ICS calculation(s) with {} worker(s).".format(
            len(signatures),
            effective_workers,
        ),
        flush=True,
    )
    initializer_args = (plans, covariances, wfn, blas_threads)
    results = {}
    if effective_workers == 1:
        _init_ics_worker(*initializer_args)
        for signature in signatures:
            key, metric = _run_ics_plan(signature)
            results[key] = metric
        return results

    with ProcessPoolExecutor(
        max_workers=effective_workers,
        initializer=_init_ics_worker,
        initargs=initializer_args,
    ) as executor:
        future_by_signature = {
            executor.submit(_run_ics_plan, signature): signature
            for signature in signatures
        }
        completed = 0
        for future in as_completed(future_by_signature):
            signature, metric = future.result()
            results[signature] = metric
            completed += 1
            if completed == len(signatures) or completed % max(1, len(signatures) // 20) == 0:
                print(
                    "  Completed {}/{} unique graph ICS calculations.".format(
                        completed,
                        len(signatures),
                    ),
                    flush=True,
                )
    return results


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


def _pauli_support_key(pauli_string):
    """Return a coefficient-independent key for a Tequila PauliString."""

    if not hasattr(pauli_string, "items"):
        raise TypeError(
            "Expected graph node attribute 'v' to be a Tequila PauliString, got {}.".format(
                type(pauli_string).__name__
            )
        )
    return tuple(
        sorted(
            (int(qubit), str(pauli).upper())
            for qubit, pauli in pauli_string.items()
        )
    )


def groups_from_ordered_gflow_graph(binary_hamiltonian, graph, condition="fc"):
    """Build canonical BinaryPauliString groups from graph-embedded Pauli terms."""

    if condition not in {"fc", "qwc"}:
        raise ValueError("Unknown grouping condition '{}'; expected 'fc' or 'qwc'.".format(condition))

    measurable_terms = extract_measurable_terms(binary_hamiltonian)
    terms_by_key = {}
    for term in measurable_terms:
        key = _pauli_support_key(term.to_pauli_strings())
        if key in terms_by_key:
            raise ValueError("Hamiltonian contains duplicate measurable Pauli term {}.".format(key))
        terms_by_key[key] = term

    graph_records = []
    for node, data in graph.nodes(data=True):
        if "color" not in data:
            raise ValueError("Graph node {} is missing the 'color' attribute.".format(node))
        if "v" not in data:
            raise ValueError(
                "Graph node {} is missing the embedded Pauli-term attribute 'v'. "
                "Regenerate it with driver_sv_ordered.py.".format(node)
            )
        graph_records.append(
            (node, data["color"], _pauli_support_key(data["v"]))
        )

    graph_keys = [key for _, _, key in graph_records]
    if len(graph_keys) != len(set(graph_keys)):
        raise ValueError("The ordered graph contains at least one duplicate Pauli term.")

    graph_key_set = set(graph_keys)
    hamiltonian_key_set = set(terms_by_key)
    if graph_key_set != hamiltonian_key_set:
        missing = sorted(hamiltonian_key_set - graph_key_set)
        extra = sorted(graph_key_set - hamiltonian_key_set)
        raise ValueError(
            "Ordered graph terms do not match the current Hamiltonian. "
            "Missing={}, Extra={}.".format(missing, extra)
        )

    color_to_terms = {}
    for _, color, key in graph_records:
        color_to_terms.setdefault(color, []).append(terms_by_key[key])

    groups = [color_to_terms[color] for color in sorted(color_to_terms)]
    for color, group in zip(sorted(color_to_terms), groups):
        for left_index, left in enumerate(group):
            for right in group[left_index + 1:]:
                if condition == "qwc":
                    compatible = left.qubit_wise_commute(right)
                else:
                    compatible = left.commute(right)
                if not compatible:
                    raise ValueError(
                        "Ordered graph color {} contains terms that are not {} compatible: "
                        "{} and {}.".format(
                            color,
                            condition.upper(),
                            _pauli_support_key(left.to_pauli_strings()),
                            _pauli_support_key(right.to_pauli_strings()),
                        )
                    )
    return groups


def iterative_coefficient_splitting_from_ordered_gflow_graph(
    binary_hamiltonian,
    graph,
    cov_dict,
    condition="fc",
):
    """Run ICS using the ordered graph's embedded Pauli-term mapping."""

    initial_groups = groups_from_ordered_gflow_graph(
        binary_hamiltonian,
        graph,
        condition=condition,
    )
    return iterative_coefficient_splitting_from_groups(
        initial_groups,
        cov_dict,
        condition=condition,
    )


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


def rank_graphs_by_custom_reward(sampled_graphs, metrics, l0, l1, l2):
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
    return scored_graphs


def write_top_graphs(
    fig_name,
    sampled_graphs,
    scored_graphs,
    n_save,
    l0,
    l1,
    l2,
):
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
    return top_indices


def report_top_graphs(
    metrics,
    scored_graphs,
    n_save,
    ics_results_by_index=None,
):
    n_to_report = min(n_save, len(scored_graphs))
    for rank, (reward, idx) in enumerate(scored_graphs[:n_to_report], start=1):
        metric_dict = {
            "measurement": float(metrics["measurements"][idx]),
            "num_groups": int(metrics["num_groups"][idx]),
            "two_qubit_gates": int(metrics["two_qubit_gates"][idx]),
        }
        print("  [{}] reward={:.10g}, {}".format(rank, reward, format_metric_triplet(metric_dict)))

        if ics_results_by_index is not None:
            print(
                "      After  ICS: {}".format(
                    format_metric_triplet(ics_results_by_index[idx])
                )
            )
    return n_to_report


def make_output_path(fig_name, y_axis):
    if y_axis == "groups":
        return fig_name + "_ics_pareto_joint_all.svg"
    return fig_name + "_ics_pareto_joint_all_2qubit.svg"


def main(argv=None):
    total_start = time.perf_counter()
    args = parse_args(argv)
    fig_name = args.func_name
    sampled_graphs_path = fig_name + "_sampled_graphs.p"
    metrics_path = fig_name + "_sampled_graphs_metrics.p"

    load_start = time.perf_counter()
    sampled_graphs = load_sampled_graphs(sampled_graphs_path)
    metrics = load_cached_metrics(metrics_path, len(sampled_graphs))
    output_path = make_output_path(fig_name, args.y_axis)

    print(
        "Loaded {} valid sampled graphs from {} in {:.3f} s.".format(
            len(sampled_graphs),
            sampled_graphs_path,
            time.perf_counter() - load_start,
        ),
        flush=True,
    )
    print("Loaded metrics from {}".format(metrics_path), flush=True)

    points = np.column_stack(
        (
            metrics["measurements"].astype(float),
            metrics["num_groups"].astype(float),
        )
    )
    if args.y_axis == "two-qubit":
        points[:, 1] = metrics["two_qubit_gates"].astype(float)
    x = points[:, 0]
    mask = pareto_front_min(points)
    pareto_indices = np.flatnonzero(mask)
    pareto_indices = pareto_indices[
        np.argsort(points[pareto_indices, 0], kind="stable")
    ]
    pareto_sorted = points[pareto_indices]

    scored_graphs = None
    top_indices = []
    if args.save:
        scored_graphs = rank_graphs_by_custom_reward(
            sampled_graphs,
            metrics,
            args.l0,
            args.l1,
            args.l2,
        )
        # Persist the requested ranked graphs before chemistry, covariance, or
        # ICS work so a later expensive-stage failure does not lose them.
        top_indices = write_top_graphs(
            fig_name,
            sampled_graphs,
            scored_graphs,
            args.n_save,
            args.l0,
            args.l1,
            args.l2,
        )

    chemistry_start = time.perf_counter()
    mol, H, _, n_paulis, Hq = args.func()
    n_qubits = int(count_qubits(Hq))
    binary_hamiltonian = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
    binary_terms = ordered_measurable_terms(binary_hamiltonian)
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
    sparse_hamiltonian = get_sparse_operator(Hq)
    _, variance_wfn = get_variance_wavefunction(
        mol,
        Hq,
        method=args.wfn,
        sparse_hamiltonian=sparse_hamiltonian,
    )
    variance_wfn = np.asarray(variance_wfn, dtype=complex).reshape(-1)
    print(
        "Sparse Hamiltonian and {} state built in {:.3f} s.".format(
            args.wfn,
            time.perf_counter() - state_start,
        ),
        flush=True,
    )

    grouping_condition = "qwc" if args.qwc else "fc"
    print(
        "Using {} groupings for SI/ICS.".format(
            "QWC" if args.qwc else "fully commuting"
        ),
        flush=True,
    )
    position_by_key = {
        term.binary_tuple(): position
        for position, term in enumerate(binary_terms)
    }
    compatibility_start = time.perf_counter()
    compatibility = build_compatibility_matrix(binary_terms, grouping_condition)
    print(
        "Compatibility matrix built in {:.3f} s.".format(
            time.perf_counter() - compatibility_start
        ),
        flush=True,
    )

    si_groups, _ = binary_hamiltonian.commuting_groups(
        options={"method": "si", "condition": grouping_condition}
    )
    required_pairs = set()
    si_plan = None
    if args.ics:
        si_plan, si_pairs = build_fast_ics_plan(
            si_groups,
            grouping_condition,
            position_by_key,
            binary_terms,
            compatibility,
            signature="__sorted_insertion__",
        )
        required_pairs.update(si_pairs)
    else:
        required_pairs.update(pairs_within_groups(si_groups, position_by_key))

    fingerprint = hamiltonian_state_fingerprint(
        binary_terms,
        variance_wfn,
        n_qubits,
        args.wfn,
    )
    cache_dir = None if args.no_cache else args.cache_dir
    moment_cache_path = None
    ics_cache_path = None
    if cache_dir:
        moment_cache_path = _cache_path(
            cache_dir,
            fig_name,
            args.wfn,
            "moments_v2.p",
        )
        ics_cache_path = _cache_path(
            cache_dir,
            fig_name,
            args.wfn,
            "{}_{}_ics_metrics_v2.p".format(
                grouping_condition,
                args.cov_backend,
            ),
        )
    ics_fingerprint = hashlib.sha256(
        "{}|{}|{}|ics|sparse-tequila-report-v1".format(
            fingerprint,
            grouping_condition,
            args.cov_backend,
        ).encode("ascii")
    ).hexdigest()
    cached_ics_payload = load_versioned_cache(
        ics_cache_path,
        ics_fingerprint,
        rebuild=args.rebuild_cache,
    )
    graph_ics_by_signature = (
        dict(cached_ics_payload.get("results", {}))
        if cached_ics_payload
        else {}
    )

    graph_index_to_signature = {}
    graph_plans = {}
    selected_graph_indices = []
    if args.ics:
        selected_graph_indices = list(
            dict.fromkeys(list(map(int, pareto_indices)) + list(map(int, top_indices)))
        )
        terms_by_support = {}
        for term in binary_terms:
            support = _pauli_support_key(term.to_pauli_strings())
            if support in terms_by_support:
                raise ValueError("Hamiltonian contains duplicate measurable Pauli term {}.".format(support))
            terms_by_support[support] = term
        expected_supports = set(terms_by_support)

        planning_start = time.perf_counter()
        for graph_index in selected_graph_indices:
            initial_groups = resolve_ordered_graph_groups(
                sampled_graphs[graph_index],
                terms_by_support,
                expected_supports,
                grouping_condition,
                position_by_key=position_by_key,
                compatibility=compatibility,
            )
            signature = ics_input_signature(initial_groups, position_by_key)
            graph_index_to_signature[graph_index] = signature
            if signature in graph_ics_by_signature or signature in graph_plans:
                continue
            plan, plan_pairs = build_fast_ics_plan(
                initial_groups,
                grouping_condition,
                position_by_key,
                binary_terms,
                compatibility,
                signature=signature,
            )
            graph_plans[signature] = plan
            required_pairs.update(plan_pairs)
        print(
            "Planned {} selected graph indices as {} unique ICS inputs "
            "({} cached, {} to compute) in {:.3f} s.".format(
                len(selected_graph_indices),
                len(set(graph_index_to_signature.values())),
                len(set(graph_index_to_signature.values()) & set(graph_ics_by_signature)),
                len(graph_plans),
                time.perf_counter() - planning_start,
            ),
            flush=True,
        )

    covariance_start = time.perf_counter()
    if args.cov_backend == "legacy":
        legacy_covariance = prepare_fast_cov_dict(
            binary_hamiltonian,
            Hq,
            variance_wfn,
            args.cov_workers,
        )
        covariances = indexed_covariance_from_legacy(
            binary_terms,
            required_pairs,
            legacy_covariance,
        )
    else:
        covariances = build_selected_covariance(
            binary_terms,
            required_pairs,
            n_qubits,
            variance_wfn,
            args.cov_workers,
            args.blas_threads,
            moment_cache_path=moment_cache_path,
            fingerprint=fingerprint,
            rebuild_cache=args.rebuild_cache,
        )
    print(
        "Covariance backend='{}': entries={}, runtime_s={:.3f}.".format(
            args.cov_backend,
            len(covariances),
            time.perf_counter() - covariance_start,
        ),
        flush=True,
    )

    reporting_start = time.perf_counter()
    reporter = SparseTequilaReporter(variance_wfn)
    print(
        "Sparse Tequila reporter support={}/{}; action_cache_budget={:.1f} MiB; "
        "initialized in {:.3f} s.".format(
            len(reporter.support),
            reporter.dimension,
            reporter.action_cache_bytes / 2**20,
            time.perf_counter() - reporting_start,
        ),
        flush=True,
    )
    si_metrics = compute_group_metrics_indexed(
        si_groups,
        covariances,
        reporter=reporter,
    )

    print("")
    print("Sorted insertion:")
    print("  Before ICS: {}".format(format_metric_triplet(si_metrics)))

    si_ics_metrics = None
    pareto_ics_results = []
    ics_results_by_index = None
    if args.ics:
        with threadpool_limits(limits=args.blas_threads):
            si_ics_groups = si_plan.overlapping.optimal_overlapping_groups(
                OverlappingAuxiliary(covariances)
            )
        si_ics_metrics = compute_group_metrics_indexed(
            [BinaryHamiltonian(group) for group in si_ics_groups],
            covariances,
            reporter=reporter,
        )

        print("  After  ICS: {}".format(format_metric_triplet(si_ics_metrics)))
        reporting_wavefunction = reporter.wavefunction
        # Graph workers construct independent reporters.  Drop the parent's
        # bounded action cache before ProcessPool fork so children do not
        # inherit cache pages that they will never use.
        reporter.clear_action_cache()
        new_graph_results = run_ics_plans(
            graph_plans,
            covariances,
            reporting_wavefunction,
            args.ics_workers,
            args.blas_threads,
        )
        graph_ics_by_signature.update(new_graph_results)
        if ics_cache_path:
            save_versioned_cache(
                ics_cache_path,
                ics_fingerprint,
                results=graph_ics_by_signature,
            )
        ics_results_by_index = {
            graph_index: graph_ics_by_signature[signature]
            for graph_index, signature in graph_index_to_signature.items()
        }

    if args.save:
        report_top_graphs(
            metrics,
            scored_graphs,
            args.n_save,
            ics_results_by_index=ics_results_by_index,
        )

    if args.ics:
        print("")
        print("Pareto-front GFlowNet graphs:")
        for display_idx, graph_idx in enumerate(pareto_indices):
            before_metrics = {
                "measurement": float(metrics["measurements"][graph_idx]),
                "num_groups": int(metrics["num_groups"][graph_idx]),
                "two_qubit_gates": int(metrics["two_qubit_gates"][graph_idx]),
            }
            after_metrics = ics_results_by_index[int(graph_idx)]
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
    print("Analysis before plotting runtime_s={:.3f}.".format(time.perf_counter() - total_start))

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
            linewidth=2.0,
            label=None,
            zorder=1,
        )
        g.ax_joint.plot(
            si_ics_x,
            si_ics_y,
            marker="*",
            color="#cc5c00",
            markersize=12,
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
                color="#7f53d8",
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
    png_output_path = output_path[:-4] + ".png"
    g.figure.savefig(png_output_path, format="png", dpi=300, bbox_inches="tight")
    print("Saved Pareto plot to {}".format(output_path))
    print("Saved Pareto plot to {}".format(png_output_path))


if __name__ == "__main__":
    main()
