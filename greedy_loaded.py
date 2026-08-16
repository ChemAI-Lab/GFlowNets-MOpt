from __future__ import annotations

"""Fast SI, Greedy grouping, and ICS for loaded BK Hamiltonians."""

import argparse
import math
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path


def configure_runtime_environment():
    """Keep library caches writable and avoid nested BLAS parallelism."""

    cache_defaults = {
        "MPLCONFIGDIR": "/tmp/greedy_mplconfig",
        "XDG_CACHE_HOME": "/tmp/greedy_cache",
    }
    for variable, default_path in cache_defaults.items():
        if variable not in os.environ:
            os.environ[variable] = default_path
            Path(default_path).mkdir(parents=True, exist_ok=True)

    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")


configure_runtime_environment()

import numpy as np
import tequila as tq
from openfermion import (
    FermionOperator,
    QubitOperator,
    variance as operator_variance,
)
from openfermion.linalg import get_ground_state, get_sparse_operator
from openfermion.transforms import bravyi_kitaev
from openfermion.utils import count_qubits
from tequila.grouping.binary_rep import BinaryHamiltonian
from tequila.hamiltonian import QubitHamiltonian

from gflow_vqe.overlapping_helpers import iterative_coefficient_splitting_from_groups


LOADED_SYSTEM_NAMES = (
    "h2",
    "lih",
    "beh2",
    "h2o",
    "nh3",
    "n2",
)
DISPLAY_NAMES = {
    "h2": "H2",
    "lih": "LiH",
    "beh2": "BeH2",
    "h2o": "H2O",
    "nh3": "NH3",
    "n2": "N2",
}
HAM_LIBRARY_DIRECTORY = Path(__file__).resolve().parent / "ham_lib"
REFERENCE_METHOD = "FCI"
DEFAULT_COVARIANCE_CHUNKSIZE = 128
METHOD_LABELS = {
    "SI": "Sorted insertion (SI)",
    "SI-ICS": "Sorted insertion + ICS (SI-ICS)",
    "GREEDY": "Greedy grouping",
    "GREEDY-ICS": "Greedy grouping + ICS",
}
METHOD_LABEL = METHOD_LABELS["GREEDY"]
RESULT_ORDER = ("SI", "SI-ICS", "GREEDY", "GREEDY-ICS")


@dataclass(frozen=True)
class PauliTerm:
    index: int
    pauli_tuple: tuple[tuple[int, str], ...]
    ops: tuple[str, ...]
    coefficient: complex
    word: str
    source_order: int


@dataclass
class GreedyResult:
    groups: list[list[PauliTerm]]
    variances: list[float]
    eps_sq_m: float
    sample_ratios: list[float]
    runtime_s: float
    fci_eps_sq_m: float | None = None
    method: str = "GREEDY"


@dataclass
class GreedyContext:
    """Pairwise lookup tables shared by SI and Greedy grouping."""

    terms: list[PauliTerm]
    single_variances: np.ndarray
    scaled_covariances: np.ndarray
    compatible: np.ndarray


_ACTION_STATE = None
_ACTION_N_QUBITS = None
_ACTION_TERMS = None


def default_cov_workers():
    return max(1, min(8, os.cpu_count() or 1))


def load_qubit_hamiltonian(
    molecule,
    library_directory=HAM_LIBRARY_DIRECTORY,
):
    """Load a serialized FermionOperator and apply the driver's BK mapping."""

    molecule = str(molecule).lower()
    if molecule not in LOADED_SYSTEM_NAMES:
        raise ValueError(
            "Unknown loaded molecule '{}'. Available values: {}.".format(
                molecule,
                ", ".join(LOADED_SYSTEM_NAMES),
            )
        )

    input_path = Path(library_directory) / "{}_fer.bin".format(molecule)
    with input_path.open("rb") as handle:
        fermion_hamiltonian = pickle.load(handle)
    if not isinstance(fermion_hamiltonian, FermionOperator):
        raise TypeError(
            "Expected '{}' to contain a FermionOperator, found {}.".format(
                input_path,
                type(fermion_hamiltonian).__name__,
            )
        )

    qubit_hamiltonian = bravyi_kitaev(fermion_hamiltonian)
    tequila_hamiltonian = QubitHamiltonian(qubit_hamiltonian)
    return qubit_hamiltonian, tequila_hamiltonian


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Fast sorted insertion, Greedy grouping, and their ICS variants "
            "for the serialized Bravyi-Kitaev Hamiltonians used by "
            "driver_loaded_hams.py."
        )
    )
    parser.add_argument(
        "molecule",
        type=lambda value: str(value).lower(),
        choices=LOADED_SYSTEM_NAMES,
        metavar="MOLECULE",
        help=(
            "Loaded molecule name: {}."
            .format(", ".join(LOADED_SYSTEM_NAMES))
        ),
    )
    parser.add_argument(
        "--condition",
        choices=("fc", "qwc"),
        default="fc",
        help="Fully commuting or qubit-wise commuting groups (default: fc).",
    )
    parser.add_argument(
        "--cov-workers",
        type=int,
        default=default_cov_workers(),
        help="Worker processes used to construct Pauli action rows (default: up to 8).",
    )
    parser.add_argument(
        "--cov-chunksize",
        type=int,
        default=DEFAULT_COVARIANCE_CHUNKSIZE,
        help="Maximum Pauli terms in one covariance worker task (default: 128).",
    )
    parser.add_argument(
        "--serial-cov-dict",
        action="store_true",
        help="Compatibility alias for --cov-workers 1.",
    )
    parser.add_argument(
        "--print-groups",
        action="store_true",
        help="Print the Pauli words in every result group.",
    )
    parser.add_argument(
        "--no-ics",
        action="store_true",
        help="Run only SI and Greedy grouping, without coefficient splitting.",
    )
    args = parser.parse_args(argv)
    if args.cov_workers < 1:
        parser.error("--cov-workers must be at least 1.")
    if args.cov_chunksize < 1:
        parser.error("--cov-chunksize must be at least 1.")
    if args.serial_cov_dict:
        args.cov_workers = 1
    return args


def clean_complex(value, tiny=1.0e-12):
    value = complex(value)
    # Small real components can contribute to the Greedy objective when many
    # covariances are accumulated.  Preserve them; only discard imaginary
    # roundoff from quantities that should be real.
    real = value.real
    imag = 0.0 if abs(value.imag) < tiny else value.imag
    return complex(real, imag)


def clean_real(value, tiny=1.0e-9):
    value = complex(value)
    if abs(value.imag) > tiny:
        raise ValueError("Expected a real value, got {}.".format(value))
    return float(value.real)


def clean_variance(value, tiny=1.0e-10):
    value = complex(value)
    if abs(value.imag) > tiny:
        raise ValueError("Expected a real variance, got {}.".format(value))
    real_value = float(value.real)
    if real_value < 0.0 and abs(real_value) < tiny:
        return 0.0
    if real_value < 0.0:
        raise ValueError("Computed a negative variance: {}.".format(real_value))
    return real_value


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


def terms_qubit_wise_commute(term1, term2):
    return all(
        op1 == "I" or op2 == "I" or op1 == op2
        for op1, op2 in zip(term1.ops, term2.ops)
    )


def terms_compatible(term1, term2, condition):
    if condition == "fc":
        return terms_fully_commute(term1, term2)
    if condition == "qwc":
        return terms_qubit_wise_commute(term1, term2)
    raise ValueError("Unsupported compatibility condition '{}'.".format(condition))


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


def get_covariance(term1, term2, covariances):
    if term1.index <= term2.index:
        key = (term1.index, term2.index)
    else:
        key = (term2.index, term1.index)
    return covariances[key]


def eps_sq_m_from_variances(variances):
    sqrt_sum = sum(math.sqrt(max(float(variance), 0.0)) for variance in variances)
    return sqrt_sum * sqrt_sum


def sample_ratios_from_variances(variances):
    weights = [math.sqrt(max(float(variance), 0.0)) for variance in variances]
    total = sum(weights)
    if total == 0.0:
        return [1.0 / len(weights) for _ in weights]
    return [weight / total for weight in weights]


def build_greedy_context(measurable_terms, covariances, condition):
    n_terms = len(measurable_terms)
    scaled_covariances = np.zeros((n_terms, n_terms), dtype=float)
    compatible = np.zeros((n_terms, n_terms), dtype=bool)

    for left_position, left in enumerate(measurable_terms):
        for right_position in range(left_position, n_terms):
            right = measurable_terms[right_position]
            pair_is_compatible = terms_compatible(left, right, condition)
            compatible[left_position, right_position] = pair_is_compatible
            compatible[right_position, left_position] = pair_is_compatible
            if not pair_is_compatible:
                continue

            scaled_covariance = clean_real(
                left.coefficient
                * right.coefficient
                * get_covariance(left, right, covariances)
            )
            scaled_covariances[left_position, right_position] = scaled_covariance
            scaled_covariances[right_position, left_position] = scaled_covariance

    single_variances = np.asarray(
        [clean_variance(value) for value in scaled_covariances.diagonal()],
        dtype=float,
    )
    np.fill_diagonal(scaled_covariances, single_variances)
    return GreedyContext(
        terms=list(measurable_terms),
        single_variances=single_variances,
        scaled_covariances=scaled_covariances,
        compatible=compatible,
    )


def group_variances_from_positions(context, position_groups):
    """Evaluate group variances through the shared scaled-covariance matrix."""

    variances = []
    for group in position_groups:
        group_array = np.asarray(group, dtype=int)
        variance = context.scaled_covariances[
            np.ix_(group_array, group_array)
        ].sum()
        variances.append(clean_variance(variance))
    return variances


def sorted_insertion_grouping(context):
    """Group terms by descending coefficient magnitude (standard SI)."""

    ordered_positions = sorted(
        range(len(context.terms)),
        key=lambda position: (
            -abs(context.terms[position].coefficient),
            context.terms[position].source_order,
        ),
    )
    groups = []
    compatibility_masks = []
    for position in ordered_positions:
        for group_index, mask in enumerate(compatibility_masks):
            if bool(mask[position]):
                groups[group_index].append(position)
                compatibility_masks[group_index] = np.logical_and(
                    mask,
                    context.compatible[position],
                )
                break
        else:
            groups.append([position])
            compatibility_masks.append(context.compatible[position].copy())

    return groups, group_variances_from_positions(context, groups)


def validate_position_groups(context, position_groups):
    seen = []
    for group_index, group in enumerate(position_groups):
        for local_index, position in enumerate(group):
            for other_position in group[local_index + 1 :]:
                if not bool(context.compatible[position, other_position]):
                    raise ValueError(
                        "Group {} contains incompatible terms.".format(group_index)
                    )
            seen.append(position)

    expected = set(range(len(context.terms)))
    if len(seen) != len(set(seen)) or set(seen) != expected:
        raise ValueError(
            "Grouping is not a non-overlapping cover of all measurable terms."
        )


def add_position_to_group(
    context,
    groups,
    variances,
    covariance_sums,
    masks,
    group_index,
    position,
    new_variance,
):
    groups[group_index].append(position)
    variances[group_index] = clean_variance(new_variance)
    covariance_sums[group_index] = (
        covariance_sums[group_index] + context.scaled_covariances[position]
    )
    masks[group_index] = np.logical_and(
        masks[group_index],
        context.compatible[position],
    )


def open_position_group(
    context,
    groups,
    variances,
    covariance_sums,
    masks,
    position,
):
    groups.append([position])
    variances.append(float(context.single_variances[position]))
    covariance_sums.append(context.scaled_covariances[position].copy())
    masks.append(context.compatible[position].copy())


def greedy_grouping(context):
    """Build groups by globally minimizing the next insertion metric.

    If group ``g`` contains ``H_g = sum_i c_i P_i``, its variance is

        V_g = sum_ij c_i c_j Cov(P_i, P_j).

    The first group is seeded with the largest single-term variance.  At every
    later step, all remaining terms and all compatible destination groups are
    considered, and the insertion minimizing ``(sum_g sqrt(V_g))**2`` is
    accepted.  A singleton is considered only when a term has no compatible
    existing group, matching the original Greedy/VarSI-G rule.
    """

    remaining_positions = sorted(
        range(len(context.terms)),
        key=lambda position: (
            -context.single_variances[position],
            context.terms[position].source_order,
        ),
    )
    groups = []
    variances = []
    covariance_sums = []
    masks = []

    while remaining_positions:
        if not groups:
            position = remaining_positions.pop(0)
            open_position_group(
                context,
                groups,
                variances,
                covariance_sums,
                masks,
                position,
            )
            continue

        current_sqrt_sum = sum(math.sqrt(variance) for variance in variances)
        best_candidate = None

        for remaining_index, position in enumerate(remaining_positions):
            term_variance = float(context.single_variances[position])
            compatible_candidates = []

            for group_index, mask in enumerate(masks):
                if not bool(mask[position]):
                    continue

                new_variance = clean_variance(
                    variances[group_index]
                    + term_variance
                    + 2.0 * covariance_sums[group_index][position]
                )
                new_sqrt_sum = (
                    current_sqrt_sum
                    - math.sqrt(variances[group_index])
                    + math.sqrt(new_variance)
                )
                compatible_candidates.append(
                    (new_sqrt_sum * new_sqrt_sum, new_variance, group_index)
                )

            if compatible_candidates:
                metric, new_variance, group_index = min(compatible_candidates)
                candidate = (
                    metric,
                    new_variance,
                    group_index,
                    remaining_index,
                )
            else:
                new_variance = term_variance
                metric = (current_sqrt_sum + math.sqrt(new_variance)) ** 2
                candidate = (
                    metric,
                    new_variance,
                    len(groups),
                    remaining_index,
                )

            if best_candidate is None or candidate < best_candidate:
                best_candidate = candidate

        _, new_variance, group_index, remaining_index = best_candidate
        position = remaining_positions.pop(remaining_index)
        if group_index == len(groups):
            open_position_group(
                context,
                groups,
                variances,
                covariance_sums,
                masks,
                position,
            )
        else:
            add_position_to_group(
                context,
                groups,
                variances,
                covariance_sums,
                masks,
                group_index,
                position,
                new_variance,
            )

    return groups, variances


def make_base_result(method, context, position_groups, variances, runtime_s):
    validate_position_groups(context, position_groups)
    groups = [
        [context.terms[position] for position in group]
        for group in position_groups
    ]
    variances = [float(variance) for variance in variances]
    return GreedyResult(
        groups=groups,
        variances=variances,
        eps_sq_m=eps_sq_m_from_variances(variances),
        sample_ratios=sample_ratios_from_variances(variances),
        runtime_s=runtime_s,
        method=method,
    )


def make_greedy_result(context, position_groups, variances, runtime_s):
    """Backward-compatible constructor for the base Greedy result."""

    return make_base_result(
        "GREEDY",
        context,
        position_groups,
        variances,
        runtime_s,
    )


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


def build_ics_bridge(tequila_hamiltonian, measurable_terms, covariances):
    """Match the fast Pauli representation to Tequila's ICS representation."""

    binary_hamiltonian = BinaryHamiltonian.init_from_qubit_hamiltonian(
        tequila_hamiltonian
    )
    binary_terms_by_key = {
        term.binary_tuple(): term
        for term in binary_hamiltonian.binary_terms
        if np.any(term.get_binary())
    }
    pauli_terms_by_key = {
        binary_tuple_for_term(term): term for term in measurable_terms
    }
    if set(binary_terms_by_key) != set(pauli_terms_by_key):
        missing = sorted(set(pauli_terms_by_key) - set(binary_terms_by_key))
        extra = sorted(set(binary_terms_by_key) - set(pauli_terms_by_key))
        raise ValueError(
            "Fast and ICS Pauli representations do not match. "
            "Missing={}, Extra={}.".format(missing, extra)
        )

    for key, pauli_term in pauli_terms_by_key.items():
        binary_coefficient = binary_terms_by_key[key].get_coeff()
        if not np.isclose(binary_coefficient, pauli_term.coefficient):
            raise ValueError(
                "Fast and ICS coefficients differ for {}: {} vs {}.".format(
                    pauli_term.word,
                    pauli_term.coefficient,
                    binary_coefficient,
                )
            )

    keys_by_index = {
        term.index: binary_tuple_for_term(term) for term in measurable_terms
    }
    ics_covariances = {
        (keys_by_index[left_index], keys_by_index[right_index]): covariance
        for (left_index, right_index), covariance in covariances.items()
    }
    return binary_terms_by_key, pauli_terms_by_key, ics_covariances


def pauli_groups_to_binary_groups(groups, binary_terms_by_key):
    return [
        [binary_terms_by_key[binary_tuple_for_term(term)] for term in group]
        for group in groups
    ]


def binary_groups_to_pauli_groups(binary_groups, pauli_terms_by_key):
    pauli_groups = []
    for group in binary_groups:
        if isinstance(group, BinaryHamiltonian):
            binary_terms = group.binary_terms
        else:
            binary_terms = list(group)

        pauli_group = []
        for binary_term in binary_terms:
            base_term = pauli_terms_by_key[binary_term.binary_tuple()]
            pauli_group.append(
                PauliTerm(
                    index=base_term.index,
                    pauli_tuple=base_term.pauli_tuple,
                    ops=base_term.ops,
                    coefficient=clean_complex(binary_term.get_coeff()),
                    word=base_term.word,
                    source_order=base_term.source_order,
                )
            )
        pauli_groups.append(pauli_group)
    return pauli_groups


def group_variances_from_covariances(groups, covariances):
    variances = []
    for group in groups:
        variance = 0.0 + 0.0j
        for left in group:
            for right in group:
                variance += (
                    left.coefficient
                    * right.coefficient
                    * get_covariance(left, right, covariances)
                )
        variances.append(clean_variance(variance))
    return variances


def validate_ics_groups(groups, pauli_terms_by_key, condition):
    """Validate compatibility and conservation of split coefficients."""

    coefficient_totals = {
        key: 0.0 + 0.0j for key in pauli_terms_by_key
    }
    for group_index, group in enumerate(groups):
        group_keys = set()
        for term_index, term in enumerate(group):
            key = binary_tuple_for_term(term)
            if key not in coefficient_totals:
                raise ValueError(
                    "ICS returned an unknown Pauli term {}.".format(term.word)
                )
            if key in group_keys:
                raise ValueError(
                    "ICS group {} contains duplicate term {}.".format(
                        group_index,
                        term.word,
                    )
                )
            group_keys.add(key)
            coefficient_totals[key] += term.coefficient
            for other in group[term_index + 1 :]:
                if not terms_compatible(term, other, condition):
                    raise ValueError(
                        "ICS group {} contains incompatible terms {} and {}.".format(
                            group_index,
                            term.word,
                            other.word,
                        )
                    )

    for key, base_term in pauli_terms_by_key.items():
        if not np.isclose(coefficient_totals[key], base_term.coefficient):
            raise ValueError(
                "ICS split coefficients for {} sum to {}, expected {}.".format(
                    base_term.word,
                    coefficient_totals[key],
                    base_term.coefficient,
                )
            )


def make_ics_result(method, groups, covariances, sample_ratios, runtime_s):
    variances = group_variances_from_covariances(groups, covariances)
    sample_ratios = [
        float(value) for value in np.asarray(sample_ratios).reshape(-1)
    ]
    if len(sample_ratios) != len(groups):
        raise ValueError(
            "ICS returned {} sample ratios for {} groups.".format(
                len(sample_ratios),
                len(groups),
            )
        )
    if (
        not np.all(np.isfinite(sample_ratios))
        or any(value < -1.0e-12 for value in sample_ratios)
        or not np.isclose(sum(sample_ratios), 1.0)
    ):
        raise ValueError(
            "ICS returned invalid sample ratios: {}.".format(sample_ratios)
        )

    return GreedyResult(
        groups=groups,
        variances=variances,
        eps_sq_m=eps_sq_m_from_variances(variances),
        sample_ratios=sample_ratios,
        runtime_s=runtime_s,
        method=method,
    )


def run_ics_method(
    base_result,
    binary_terms_by_key,
    pauli_terms_by_key,
    covariances,
    ics_covariances,
    condition,
):
    method = "{}-ICS".format(base_result.method)
    start = time.perf_counter()
    try:
        initial_groups = pauli_groups_to_binary_groups(
            base_result.groups,
            binary_terms_by_key,
        )
        binary_groups, sample_ratios = iterative_coefficient_splitting_from_groups(
            initial_groups,
            ics_covariances,
            condition=condition,
        )
        groups = binary_groups_to_pauli_groups(
            binary_groups,
            pauli_terms_by_key,
        )
        if len(groups) != len(base_result.groups):
            raise ValueError(
                "ICS changed the number of groups from {} to {}.".format(
                    len(base_result.groups),
                    len(groups),
                )
            )
        validate_ics_groups(groups, pauli_terms_by_key, condition)
        result = make_ics_result(
            method,
            groups,
            covariances,
            sample_ratios,
            time.perf_counter() - start,
        )
    except Exception as exc:
        return None, {
            "method": method,
            "runtime_s": time.perf_counter() - start,
            "error": "{}: {}".format(type(exc).__name__, exc),
        }
    return result, None


def run_grouping_methods(
    context,
    tequila_hamiltonian,
    measurable_terms,
    covariances,
    condition,
    include_ics=True,
):
    """Run SI and Greedy, optionally followed by ICS for each grouping."""

    base_results = {}
    for method, grouping_function in (
        ("SI", sorted_insertion_grouping),
        ("GREEDY", greedy_grouping),
    ):
        start = time.perf_counter()
        position_groups, variances = grouping_function(context)
        runtime_s = time.perf_counter() - start
        base_results[method] = make_base_result(
            method,
            context,
            position_groups,
            variances,
            runtime_s,
        )

    results_by_method = dict(base_results)
    failures = []
    if include_ics:
        (
            binary_terms_by_key,
            pauli_terms_by_key,
            ics_covariances,
        ) = build_ics_bridge(
            tequila_hamiltonian,
            measurable_terms,
            covariances,
        )
        for method in ("SI", "GREEDY"):
            ics_result, failure = run_ics_method(
                base_results[method],
                binary_terms_by_key,
                pauli_terms_by_key,
                covariances,
                ics_covariances,
                condition,
            )
            if ics_result is not None:
                results_by_method[ics_result.method] = ics_result
            if failure is not None:
                failures.append(failure)

    results = [
        results_by_method[method]
        for method in RESULT_ORDER
        if method in results_by_method
    ]
    return results, failures


def direct_group_variances(groups, state_vector, n_qubits):
    state_vector = np.asarray(state_vector, dtype=complex).reshape(-1)
    variances = []
    for group in groups:
        operator = QubitOperator()
        for term in group:
            operator += QubitOperator(term.pauli_tuple, term.coefficient)
        variance = operator_variance(
            get_sparse_operator(operator, n_qubits=n_qubits),
            state_vector,
        )
        variances.append(clean_variance(variance, tiny=1.0e-9))
    return variances


def hamiltonian_expectation(terms, single_expectations):
    value = 0.0 + 0.0j
    for term in terms:
        if term.pauli_tuple:
            value += term.coefficient * single_expectations[term.index]
        else:
            value += term.coefficient
    return clean_real(value, tiny=1.0e-7)


def print_result(result, wfn_label):
    print("")
    print("{}:".format(METHOD_LABELS[result.method]))
    print("  eps^2 M(wfn={})={:.12g}".format(wfn_label, result.eps_sq_m))
    print("  eps^2 M(FCI)={:.12g}".format(result.fci_eps_sq_m))
    print("  Number of groups={}".format(len(result.groups)))
    print("  Compatible groups=True")
    print("  Runtime (s)={:.6f}".format(result.runtime_s))


def print_group_contents(result):
    print("")
    print("{} groups:".format(METHOD_LABELS[result.method]))
    for group_index, group in enumerate(result.groups):
        if result.method.endswith("-ICS"):
            terms = [
                "{}*{}".format(term.coefficient, term.word)
                for term in group
            ]
        else:
            terms = [term.word for term in group]
        print("  Group {}: {}".format(group_index, ", ".join(terms)))


def main(argv=None):
    args = parse_args(argv)
    display_name = DISPLAY_NAMES[args.molecule]
    input_path = HAM_LIBRARY_DIRECTORY / "{}_fer.bin".format(args.molecule)

    print(
        "Loading {} fermionic Hamiltonian from {} and applying "
        "Bravyi-Kitaev...".format(
            display_name,
            input_path,
        ),
        flush=True,
    )
    qubit_operator, tequila_hamiltonian = load_qubit_hamiltonian(
        args.molecule
    )
    n_qubits = int(count_qubits(qubit_operator))
    terms = make_terms(qubit_operator, n_qubits)
    measurable_terms = [term for term in terms if term.pauli_tuple]
    if not measurable_terms:
        raise ValueError("The Hamiltonian has no measurable Pauli terms.")

    print("Molecule={}".format(display_name), flush=True)
    print("Mapping=Bravyi-Kitaev", flush=True)
    print("Covariance wavefunction={}".format(REFERENCE_METHOD), flush=True)
    print("Compatibility condition={}".format(args.condition), flush=True)
    print("Number of qubits={}".format(n_qubits), flush=True)
    print(
        "Number of Pauli products to measure={}".format(len(measurable_terms)),
        flush=True,
    )

    sparse_hamiltonian = get_sparse_operator(
        qubit_operator,
        n_qubits=n_qubits,
    )
    energy, variance_state = get_ground_state(sparse_hamiltonian)
    variance_state = np.asarray(variance_state, dtype=complex).reshape(-1)
    reference_energy = clean_real(energy, tiny=1.0e-7)
    print(
        "{} Energy={:.16g}".format(REFERENCE_METHOD, reference_energy),
        flush=True,
    )
    action_matrix_gib = (
        len(measurable_terms) * (2**n_qubits) * 16 / (1024**3)
    )
    print(
        "Building covariance matrix with {} worker(s); "
        "Pauli-action matrix={:.3f} GiB...".format(
            args.cov_workers,
            action_matrix_gib,
        ),
        flush=True,
    )
    covariance_start = time.perf_counter()
    covariances, single_expectations = build_covariance_dictionary(
        measurable_terms,
        variance_state,
        n_qubits,
        args.cov_workers,
        args.cov_chunksize,
    )
    covariance_runtime = time.perf_counter() - covariance_start

    action_energy = hamiltonian_expectation(terms, single_expectations)
    if abs(action_energy - reference_energy) > 1.0e-7:
        raise ValueError(
            "Pauli-action energy {} does not match reference energy {}.".format(
                action_energy,
                reference_energy,
            )
        )
    print(
        "Covariance entries={} runtime_s={:.6f}".format(
            len(covariances),
            covariance_runtime,
        ),
        flush=True,
    )
    print(
        "Pauli-action energy check={:.16g}".format(action_energy),
        flush=True,
    )

    context_start = time.perf_counter()
    context = build_greedy_context(
        measurable_terms,
        covariances,
        args.condition,
    )
    context_runtime = time.perf_counter() - context_start
    context_mib = (
        context.scaled_covariances.nbytes
        + context.compatible.nbytes
        + context.single_variances.nbytes
    ) / (1024**2)
    print(
        "Grouping lookup context={:.3f} MiB runtime_s={:.6f}".format(
            context_mib,
            context_runtime,
        ),
        flush=True,
    )

    results, failed_methods = run_grouping_methods(
        context,
        tequila_hamiltonian,
        measurable_terms,
        covariances,
        args.condition,
        include_ics=not args.no_ics,
    )
    for result in results:
        result.fci_eps_sq_m = result.eps_sq_m
        print_result(result, REFERENCE_METHOD)

    if failed_methods:
        print("")
        print("Skipped methods:")
        for failure in failed_methods:
            print(
                "  {} failed after {:.6f} s: {}".format(
                    METHOD_LABELS[failure["method"]],
                    failure["runtime_s"],
                    failure["error"],
                )
            )

    print("")
    print("Ranking by eps^2 M(FCI) (lowest to highest):")
    for rank, result in enumerate(
        sorted(results, key=lambda item: item.fci_eps_sq_m),
        start=1,
    ):
        print(
            "  {}. {}: {:.12g} Groups: {}".format(
                rank,
                METHOD_LABELS[result.method],
                result.fci_eps_sq_m,
                len(result.groups),
            )
        )

    if args.print_groups:
        for result in results:
            print_group_contents(result)

    return next(result for result in results if result.method == "GREEDY")


if __name__ == "__main__":
    main()
