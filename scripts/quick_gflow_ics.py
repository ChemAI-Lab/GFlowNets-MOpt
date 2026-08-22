"""Quick SI/ICS analysis for one coefficient-ordered GFlowNet graph.

The preferred input is a saved top-graphs pickle.  When no such file is
available, the graph with the smallest cached ``eps^2 M`` value is selected
from the valid entries in ``*_sampled_graphs.p`` and
``*_sampled_graphs_metrics.p``.  Only that graph, standard sorted insertion
(SI), and SI initialized ICS are evaluated.

The selected-moment covariance and ordered-graph helpers are included directly
in this file.  It can therefore be copied into a results directory without any
other file from this repository's ``scripts`` directory.
"""

import argparse
import glob
import hashlib
import math
import os
import pickle
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np
import tequila as tq
from openfermion import get_sparse_operator
from openfermion.utils import count_qubits
from tequila.grouping.binary_rep import BinaryHamiltonian
from threadpoolctl import threadpool_limits

import gflow_vqe.hamiltonians as hamlib
from gflow_vqe.overlapping_helpers import (
    OverlappingAuxiliary,
    OverlappingGroups,
    as_tequila_wavefunction,
    extract_measurable_terms,
    get_opt_sample_size,
)
from gflow_vqe.utils import get_variance_wavefunction


DEFAULT_CACHE_DIRECTORY = ".pareto_fast_cache"
CACHE_VERSION = 2


@dataclass
class ICSPlan:
    signature: str
    overlapping: object


class IndexedCovariance:
    """Sparse-in-use covariance matrix addressed by binary Pauli keys."""

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
    """Evaluate group variances with Tequila's sparse-state semantics."""

    OUTPUT_THRESHOLD = 1.0e-8
    DEFAULT_ACTION_CACHE_BYTES = 64 * 1024**2

    def __init__(self, wfn, action_cache_bytes=DEFAULT_ACTION_CACHE_BYTES):
        self.wavefunction = as_tequila_wavefunction(wfn)
        dense_state = np.asarray(
            self.wavefunction.to_array(),
            dtype=complex,
        ).reshape(-1)
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
                output[destination_positions] += (
                    complex(coefficient) * action_values
                )

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


_MOMENT_LEFT_STATE = None
_MOMENT_RIGHT_STATE = None
_MOMENT_SUPPORT = None
_MOMENT_RIGHT_VALUES = None
_MOMENT_BYTE_PARITY = None
_MOMENT_N_QUBITS = None
_MOMENT_THREAD_LIMITER = None


def default_cov_workers():
    return max(1, min(8, os.cpu_count() or 1))


def _to_real_if_close(value, tiny=1.0e-12):
    if hasattr(value, "imag") and abs(value.imag) < tiny:
        return float(value.real)
    return value


def clean_complex(value, tiny=1.0e-12):
    value = complex(value)
    imag = 0.0 if abs(value.imag) < tiny else value.imag
    return complex(value.real, imag)


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
    """Return the masks and phase for the Pauli product ``P(left) P(right)``."""

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
    binary = np.asarray(
        [term.get_binary() for term in binary_terms],
        dtype=np.uint8,
    )
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
        terms = (
            list(group.binary_terms)
            if isinstance(group, BinaryHamiltonian)
            else list(group)
        )
        terms = [term for term in terms if np.any(term.get_binary())]
        if terms:
            normalized.append(terms)
    if not normalized:
        raise ValueError("No measurable terms were found in the provided groups.")
    return normalized


def build_fast_ics_plan(
    initial_groups,
    condition,
    position_by_key,
    binary_terms,
    compatibility,
    signature,
):
    """Construct the ICS membership model and its required covariance pairs."""

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

    expected_positions = set(range(len(binary_terms)))
    if seen_positions != expected_positions:
        missing = sorted(expected_positions - seen_positions)
        extra = sorted(seen_positions - expected_positions)
        raise ValueError(
            "Initial groups do not match the measurable Hamiltonian terms. "
            "Missing positions={}, Extra positions={}.".format(missing, extra)
        )

    sorted_terms = sorted(
        unique_terms,
        key=lambda term: abs(term.get_coeff()),
        reverse=True,
    )
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
            if added_positions and not np.all(
                compatibility[position, added_positions]
            ):
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
    for base_group, overlap_group in zip(
        overlapping.no_groups,
        overlapping.o_groups,
    ):
        positions = sorted(
            {
                position_by_key[term.binary_tuple()]
                for term in list(base_group) + list(overlap_group)
            }
        )
        required_pairs.update(combinations_with_replacement(positions, 2))
    return ICSPlan(signature=signature, overlapping=overlapping), required_pairs


def ics_input_signature(groups, position_by_key):
    """Hash ordering details that can affect stable coefficient ties."""

    ordered_groups = [
        tuple(position_by_key[term.binary_tuple()] for term in group)
        for group in _normalized_term_groups(groups)
    ]
    digest = hashlib.sha256()
    for group in ordered_groups:
        digest.update(np.asarray(group, dtype=np.int32).tobytes())
        digest.update(b"|")
    return digest.hexdigest()


def _pauli_support_key(pauli_string):
    """Return a coefficient-independent key for a Tequila PauliString."""

    if not hasattr(pauli_string, "items"):
        raise TypeError(
            "Expected graph node attribute 'v' to be a Tequila PauliString, "
            "got {}.".format(type(pauli_string).__name__)
        )
    return tuple(
        sorted(
            (int(qubit), str(pauli).upper())
            for qubit, pauli in pauli_string.items()
        )
    )


def resolve_ordered_graph_groups(
    graph,
    terms_by_support,
    expected_supports,
    condition,
    position_by_key,
    compatibility,
):
    records = []
    for node, data in graph.nodes(data=True):
        if "color" not in data:
            raise ValueError(
                "Graph node {} is missing the 'color' attribute.".format(node)
            )
        if "v" not in data:
            raise ValueError(
                "Graph node {} is missing the embedded Pauli-term attribute 'v'. "
                "Regenerate it with driver_sv_ordered.py.".format(node)
            )
        records.append((data["color"], _pauli_support_key(data["v"])))

    supports = [support for _, support in records]
    if len(supports) != len(set(supports)):
        raise ValueError("The ordered graph contains a duplicate Pauli term.")
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
    ordered_colors = sorted(color_to_terms)
    groups = [color_to_terms[color] for color in ordered_colors]

    for color, group in zip(ordered_colors, groups):
        positions = [
            position_by_key[term.binary_tuple()]
            for term in group
        ]
        if not np.all(compatibility[np.ix_(positions, positions)]):
            raise ValueError(
                "Ordered graph color {} contains terms that are not {} "
                "compatible.".format(color, condition.upper())
            )
    return groups


def fast_color_reward(graph):
    """Equivalent color reward with work proportional to within-color pairs."""

    color_to_nodes = {}
    for node, data in graph.nodes(data=True):
        if "color" not in data:
            raise ValueError(
                "Graph node {} is missing the 'color' attribute.".format(node)
            )
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


def load_sampled_graphs(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Could not find sampled graphs file '{}'.".format(path)
        )
    with open(path, "rb") as handle:
        sampled_graphs = pickle.load(handle)
    if not isinstance(sampled_graphs, list):
        sampled_graphs = list(sampled_graphs)

    valid_graphs = [
        graph for graph in sampled_graphs if fast_color_reward(graph) > 0
    ]
    if not valid_graphs:
        raise RuntimeError(
            "No valid sampled graphs were found in '{}'.".format(path)
        )
    return valid_graphs


def load_cached_metrics(path, expected_length):
    if not os.path.exists(path):
        raise FileNotFoundError("Could not find metrics file '{}'.".format(path))
    with open(path, "rb") as handle:
        cached = pickle.load(handle)

    required_keys = ("measurements", "num_groups", "two_qubit_gates")
    if not isinstance(cached, dict) or any(
        key not in cached for key in required_keys
    ):
        raise ValueError(
            "Metrics file '{}' does not match the expected cache format.".format(
                path
            )
        )
    metrics = {key: np.asarray(cached[key]) for key in required_keys}
    if metrics["measurements"].shape[0] != expected_length:
        raise ValueError(
            "Metrics file '{}' contains {} entries, but {} valid sampled graphs "
            "were loaded.".format(
                path,
                metrics["measurements"].shape[0],
                expected_length,
            )
        )
    return metrics


def normalize_binary_groups(groups):
    normalized_groups = []
    for group in groups:
        terms = (
            list(group.binary_terms)
            if isinstance(group, BinaryHamiltonian)
            else list(group)
        )
        terms = [term for term in terms if np.any(term.get_binary())]
        if terms:
            normalized_groups.append(BinaryHamiltonian(terms))
    if not normalized_groups:
        raise ValueError("No measurable terms were found in the provided groups.")
    return normalized_groups


def hamiltonian_state_fingerprint(
    binary_terms,
    state_vector,
    n_qubits,
    wfn_method,
):
    digest = hashlib.sha256()
    digest.update("pareto-fast-v{}".format(CACHE_VERSION).encode("ascii"))
    digest.update(str(int(n_qubits)).encode("ascii"))
    digest.update(str(wfn_method).upper().encode("ascii"))
    for term in binary_terms:
        digest.update(np.asarray(term.get_binary(), dtype=np.uint8).tobytes())
        digest.update(
            np.asarray(
                complex(term.get_coeff()),
                dtype=np.complex128,
            ).tobytes()
        )
    state = np.ascontiguousarray(
        np.asarray(state_vector, dtype=np.complex128).reshape(-1)
    )
    digest.update(state.view(np.uint8).tobytes())
    return digest.hexdigest()


def load_versioned_cache(path, fingerprint, rebuild=False):
    if rebuild or not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
    except Exception as error:
        print(
            "Ignoring unreadable cache '{}': {}".format(path, error),
            flush=True,
        )
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != CACHE_VERSION:
        return None
    if payload.get("fingerprint") != fingerprint:
        print(
            "Ignoring stale cache '{}' (fingerprint mismatch).".format(path),
            flush=True,
        )
        return None
    return payload


def save_versioned_cache(path, fingerprint, **values):
    if not path:
        return
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    payload = {
        "version": CACHE_VERSION,
        "fingerprint": fingerprint,
    }
    payload.update(values)
    temporary_path = "{}.tmp.{}".format(path, os.getpid())
    with open(temporary_path, "wb") as handle:
        pickle.dump(payload, handle, pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_path, path)


def tequila_wavefunction_from_array(state_vector):
    return tq.QubitWaveFunction.from_array(
        np.asarray(state_vector, dtype=complex)
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


def _init_sparse_moment_worker(
    state_vector,
    truncated_state,
    support,
    n_qubits,
    blas_threads,
):
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
        parity ^= _MOMENT_BYTE_PARITY[
            (masked >> np.uint64(shift)) & np.uint64(0xFF)
        ]
    return parity


def _sparse_pauli_moment(kind, x_mask, z_mask):
    destination = np.bitwise_xor(_MOMENT_SUPPORT, np.uint64(x_mask))
    parity = _support_parity(z_mask)
    signs = 1.0 - 2.0 * parity.astype(float)
    left_state = (
        _MOMENT_LEFT_STATE if kind == "single" else _MOMENT_RIGHT_STATE
    )
    value = np.dot(
        np.conjugate(left_state[destination]),
        signs * _MOMENT_RIGHT_VALUES,
    )
    phase = (1.0, 1.0j, -1.0, -1.0j)[
        (x_mask & z_mask).bit_count() % 4
    ]
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
            "Expected statevector size {}, got {}.".format(
                dimension,
                state_vector.size,
            )
        )

    thresholded_wfn = tequila_wavefunction_from_array(state_vector)
    truncated_state = wavefunction_array(thresholded_wfn, dimension)
    support = np.flatnonzero(truncated_state).astype(np.uint64)
    discarded_norm_sq = float(
        np.vdot(
            state_vector - truncated_state,
            state_vector - truncated_state,
        ).real
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
        product, phase = pauli_product(term_masks[left], term_masks[right])
        product_masks.add(product)
        pair_products.append((product, phase))
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
    term_masks, pair_products, single_masks, product_masks = (
        build_covariance_requests(
            binary_terms,
            required_pairs,
            n_qubits,
        )
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
            len(single_masks)
            + len(product_masks)
            - len(missing_single)
            - len(missing_product),
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
            - single_moments[term_masks[left]]
            * single_moments[term_masks[right]]
        )
        covariance_values.append(covariance)
    return IndexedCovariance(binary_terms, required_pairs, covariance_values)


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
        default=default_cov_workers(),
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
        default=DEFAULT_CACHE_DIRECTORY,
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
    sampled_graphs = load_sampled_graphs(sampled_path)
    metrics = load_cached_metrics(metrics_path, len(sampled_graphs))
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

    normalized_groups = normalize_binary_groups(groups)
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
    compatibility = build_compatibility_matrix(
        binary_terms,
        grouping_condition,
    )

    terms_by_support = {}
    for term in binary_terms:
        support = _pauli_support_key(term.to_pauli_strings())
        if support in terms_by_support:
            raise ValueError(
                "Hamiltonian contains duplicate measurable Pauli term {}.".format(
                    support
                )
            )
        terms_by_support[support] = term
    gflow_initial_groups = resolve_ordered_graph_groups(
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

    si_plan, si_pairs = build_fast_ics_plan(
        si_groups,
        grouping_condition,
        position_by_key,
        binary_terms,
        compatibility,
        signature="__sorted_insertion__",
    )
    graph_signature = ics_input_signature(
        gflow_initial_groups,
        position_by_key,
    )
    graph_plan, graph_pairs = build_fast_ics_plan(
        gflow_initial_groups,
        grouping_condition,
        position_by_key,
        binary_terms,
        compatibility,
        signature=graph_signature,
    )
    required_pairs = set(si_pairs)
    required_pairs.update(graph_pairs)

    fingerprint = hamiltonian_state_fingerprint(
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
    covariances = build_selected_covariance(
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

    reporter = SparseTequilaReporter(variance_wfn)
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

    graph_cache = load_versioned_cache(
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
        save_versioned_cache(
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
