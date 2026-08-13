"""Compare distributions for serialized, Bravyi-Kitaev Hamiltonians.

For each requested Hamiltonian from ``ham_lib/*_fer.bin``, this script follows
the loading path used by ``driver_loaded_hams.py``: unpickle the
FermionOperator, apply the Bravyi-Kitaev transform, and construct the exact
ground-state wavefunction.  It then builds the coefficient-free Pauli
covariance dictionary

    C_ij = <P_i P_j> - <P_i><P_j>

for the upper triangle of fully commuting, non-identity Pauli pairs. It saves
each plot as both PNG and SVG. The six standard probability-density plots show
unnormalized and normalized covariance distributions, the coefficient-weighted
covariance distribution c_i c_j C_ij, its maximum-absolute-normalized
distribution (c_i c_j C_ij) / max_kl |c_k c_l C_kl|, and unnormalized and
normalized Hamiltonian-coefficient distributions. A multi-molecule run also
saves a unit-peak relative-density companion for each standard plot.
Normalized covariances are correlation coefficients
C_ij / sqrt(C_ii C_jj); normalized coefficients are scaled by max_i |c_i|.
The coefficient-weighted covariances use the original, unnormalized c_i.
For each molecule, their maximum-absolute-normalized values use the same
selected pair population (including the ``--off-diagonal-only`` choice) and
are divided by the largest absolute weighted-covariance magnitude, preserving
the signs of the samples and bounding the normalized values to [-1, 1].
The constant identity term is excluded so both kinds of plots describe the
same measurable Pauli terms. All one-dimensional raw and normalized
distribution values retain their sign.

The standard normalized coefficient-weighted covariance plot always spans
[-1.05, 1.05]. Use ``--tight`` to add a second view of that distribution over
[-0.25, 0.25], without replacing the standard plot. In multi-molecule runs,
the flag also adds the corresponding unit-peak relative-density view. These
plots use a finer KDE grid and narrower bandwidth than the other plots so
structure in the restricted interval remains visible. Use ``--bars`` to
overlay probability-density histogram bars on the base density plots and save
a separate histogram-only companion for every base plot. The bars use common
adaptive bin edges in multi-molecule figures. Combined KDE-and-bar filenames
end in ``_bars``; histogram-only filenames end in ``_bars_only``.
Relative-density (unit-peak) plots never contain bars or receive histogram-only
companions. Standalone histogram bars use the same fill intensity as KDE
areas; combined-plot bars remain lighter. In comparisons, a molecule keeps the
same Seaborn colorblind-palette color across every base, histogram-only, and
unit-peak plot. Use ``--no-grid`` for a clean ticks-style version of every
figure: interior gridlines and the zero-reference line are omitted, top/right
spines and ticks are removed, and bottom/left tick marks point outward.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path


def configure_plotting_environment():
    """Select a non-interactive backend and writable cache directories."""

    os.environ["MPLBACKEND"] = "Agg"
    defaults = {
        "MPLCONFIGDIR": "/tmp/distribution_analysis_loaded_mplconfig",
        "XDG_CACHE_HOME": "/tmp/distribution_analysis_loaded_cache",
    }
    for variable, value in defaults.items():
        os.environ.setdefault(variable, value)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")


configure_plotting_environment()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tequila as tq
from matplotlib.patches import Patch
from openfermion import FermionOperator, QubitOperator
from openfermion.linalg import get_ground_state, get_sparse_operator
from openfermion.transforms import bravyi_kitaev
from openfermion.utils import count_qubits
from tequila.hamiltonian import QubitHamiltonian


# ---------------------------------------------------------------------------
# Loaded-Hamiltonian configuration
#
# Edit DEFAULT_MOLECULES below to choose the systems analyzed when no
# positional molecule names are supplied. Command-line names override it:
#
#     python distribution_analysis_loaded.py h2 h2o nh3
#
# The available serialized Hamiltonians and the BK mapping match
# driver_loaded_hams.py exactly.
# ---------------------------------------------------------------------------
AVAILABLE_MOLECULES = ("h2", "lih", "beh2", "h2o", "nh3", "n2")
DEFAULT_MOLECULES = ["lih"]
# These two values document the fixed mapping and reference-state calculation
# used by driver_loaded_hams.py; they are not alternative-method switches.
TRANSFORMATION = "bk"
REFERENCE_METHOD = "FCI"
HAM_LIBRARY_DIRECTORY = Path(__file__).resolve().parent / "ham_lib"

DISPLAY_NAMES = {
    "h2": "H2",
    "lih": "LiH",
    "beh2": "BeH2",
    "h2o": "H2O",
    "nh3": "NH3",
    "n2": "N2",
}

DEFAULT_OUTPUT_DIRECTORY = "distribution_analysis_loaded_plots"
DEFAULT_BANDWIDTH_ADJUSTMENT = 0.8
DEFAULT_KDE_GRIDSIZE = 200
DEFAULT_COVARIANCE_CHUNKSIZE = 128
DEFAULT_MAX_MEMORY_GIB = 8.0
DEFAULT_NORMALIZED_X_LIMITS = (-1.05, 1.05)
TIGHT_WEIGHTED_NORMALIZED_X_LIMITS = (-0.25, 0.25)
DETAILED_KDE_ATTRIBUTE = "covariances_coefficient_weighted_normalized"
DETAILED_KDE_BANDWIDTH_FACTOR = 0.5
DETAILED_KDE_GRIDSIZE = 1024
HISTOGRAM_MAX_BINS = 100
HISTOGRAM_CONSTANT_RANGE_PADDING = 0.025
HISTOGRAM_SINGLE_ALPHA = 0.28
HISTOGRAM_MULTI_ALPHA = 0.18
DENSITY_PALETTE = "colorblind"
KDE_FILL_ALPHA = 0.38
KDE_FILL_ZORDER = 1.5
KDE_LINE_ZORDER = 2.5
REAL_TOLERANCE = 1.0e-9
ZERO_VARIANCE_TOLERANCE = 1.0e-12


def load_qubit_hamiltonian(
    molecule,
    transformation=TRANSFORMATION,
    library_directory=HAM_LIBRARY_DIRECTORY,
):
    """Load and BK-map a serialized FermionOperator like the loaded driver."""

    molecule = str(molecule).lower()
    if molecule not in AVAILABLE_MOLECULES:
        raise ValueError(
            "Unknown loaded molecule '{}'. Available values: {}.".format(
                molecule,
                ", ".join(AVAILABLE_MOLECULES),
            )
        )
    if str(transformation).lower() != "bk":
        raise ValueError(
            "Transformation '{}' not supported; loaded Hamiltonians use BK."
            .format(transformation)
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


@dataclass
class MoleculeDistributions:
    name: str
    n_qubits: int
    n_terms: int
    energy: float
    covariance_entries: int
    covariance_runtime_s: float
    coefficients_raw: np.ndarray
    coefficients_normalized: np.ndarray
    covariances_raw: np.ndarray
    covariances_normalized: np.ndarray
    skipped_normalized_covariances: int
    covariances_coefficient_weighted: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    covariances_coefficient_weighted_normalized: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build covariance dictionaries for serialized BK Hamiltonians "
            "and compare coefficient/covariance distributions, saving every "
            "plot as PNG and SVG."
        )
    )
    parser.add_argument(
        "molecules",
        nargs="*",
        type=lambda value: str(value).lower(),
        metavar="MOLECULE",
        help=(
            "Optional loaded molecule names. These override DEFAULT_MOLECULES; "
            "multiple molecules are overlaid in each output figure. Available: "
            "{}.".format(", ".join(AVAILABLE_MOLECULES))
        ),
    )
    parser.add_argument(
        "--cov-workers",
        type=int,
        default=default_cov_workers(),
        help=(
            "Worker processes used to build Pauli action rows "
            "(default: up to 8)."
        ),
    )
    parser.add_argument(
        "--cov-chunksize",
        type=int,
        default=DEFAULT_COVARIANCE_CHUNKSIZE,
        help=(
            "Maximum Pauli terms in one covariance worker task "
            "(default: 128)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIRECTORY),
        help=(
            "Directory for generated PNG and SVG figures "
            "(default: distribution_analysis_loaded_plots)."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution for PNG output (default: 300).",
    )
    parser.add_argument(
        "--bw-adjust",
        type=float,
        default=DEFAULT_BANDWIDTH_ADJUSTMENT,
        help=(
            "Seaborn KDE bandwidth multiplier (default: 0.8). The finer "
            "normalized coefficient-weighted covariance KDE uses half this "
            "multiplier, so this option continues to control its smoothing."
        ),
    )
    parser.add_argument(
        "--off-diagonal-only",
        action="store_true",
        help="Exclude diagonal variances C_ii from all covariance-based plots.",
    )
    parser.add_argument(
        "--tight",
        action="store_true",
        help=(
            "Add a [-0.25, 0.25] view of the normalized coefficient-weighted "
            "covariance distribution, plus its unit-peak companion for a "
            "multi-molecule run. The standard [-1.05, 1.05] plots are "
            "always generated."
        ),
    )
    parser.add_argument(
        "--bars",
        action="store_true",
        help=(
            "Overlay probability-density histogram bars on base density "
            "plots and save a separate histogram-only '_bars_only' companion "
            "for each. Multi-molecule plots use common adaptive bin edges. "
            "Relative-density/unit-peak plots never show bars or receive "
            "histogram-only companions. Standalone bars use the same fill "
            "intensity as KDE areas; combined-plot bars remain lighter."
        ),
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help=(
            "Use clean ticks-style axes for every plot: hide interior "
            "gridlines and the zero-reference line, remove top/right spines "
            "and ticks, and point bottom/left ticks outward."
        ),
    )
    parser.add_argument(
        "--max-memory-gib",
        type=float,
        default=DEFAULT_MAX_MEMORY_GIB,
        help=(
            "Abort before covariance construction when the estimated dense "
            "state/action/Gram allocation exceeds this many GiB. Use 0 to "
            "disable the guard (default: 8)."
        ),
    )
    args = parser.parse_args(argv)

    if not args.molecules:
        args.molecules = [str(name).lower() for name in DEFAULT_MOLECULES]
    if not args.molecules:
        parser.error(
            "Select at least one molecule positionally or in DEFAULT_MOLECULES."
        )
    invalid_molecules = [
        name for name in args.molecules if name not in AVAILABLE_MOLECULES
    ]
    if invalid_molecules:
        parser.error(
            "Unknown molecule value(s): {}. Available values: {}."
            .format(
                ", ".join(invalid_molecules),
                ", ".join(AVAILABLE_MOLECULES),
            )
        )

    if len(args.molecules) != len(set(args.molecules)):
        parser.error("Each molecule may be listed only once.")
    if args.cov_workers < 1:
        parser.error("--cov-workers must be at least 1.")
    if args.cov_chunksize < 1:
        parser.error("--cov-chunksize must be at least 1.")
    if args.dpi < 1:
        parser.error("--dpi must be at least 1.")
    if args.bw_adjust <= 0.0:
        parser.error("--bw-adjust must be greater than zero.")
    if args.max_memory_gib < 0.0:
        parser.error("--max-memory-gib cannot be negative.")
    return args


def clean_complex(value, tiny=1.0e-12):
    """Discard only negligible imaginary roundoff from Pauli moments."""

    value = complex(value)
    imaginary = 0.0 if abs(value.imag) < tiny else value.imag
    return complex(value.real, imaginary)


def pauli_word(pauli_tuple):
    if not pauli_tuple:
        return "I"
    return " ".join(
        "{}{}".format(pauli, qubit) for qubit, pauli in pauli_tuple
    )


def make_terms(qubit_operator, n_qubits):
    """Convert an OpenFermion QubitOperator into canonical Pauli terms."""

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
                    pauli_by_qubit.get(qubit, "I")
                    for qubit in range(n_qubits)
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
    return tq.QubitWaveFunction.from_array(
        np.asarray(state_vector, dtype=complex)
    )


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
            action_row_for_term(
                _ACTION_TERMS[position],
                _ACTION_STATE,
                dimension,
            ),
        )
        for position in term_positions
    ]


def iter_index_chunks(n_items, chunksize):
    for start in range(0, n_items, chunksize):
        yield list(range(start, min(start + chunksize, n_items)))


def build_action_matrix(terms, state_vector, n_qubits, max_workers, chunksize):
    """Apply every unit-coefficient Pauli word to the reference state once."""

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
    """Build coefficient-free covariances from one Pauli-action matrix."""

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


def real_scalar(value, label, tolerance=REAL_TOLERANCE):
    value = complex(value)
    if abs(value.imag) > tolerance:
        raise ValueError(
            "{} was expected to be real, but got {}.".format(label, value)
        )
    return float(value.real)


def real_array(values, label, tolerance=REAL_TOLERANCE, allow_empty=False):
    array = np.asarray(values, dtype=complex).reshape(-1)
    if array.size == 0:
        if allow_empty:
            return np.asarray([], dtype=float)
        raise ValueError("{} is empty.".format(label))
    maximum_imaginary = float(np.max(np.abs(array.imag)))
    if maximum_imaginary > tolerance:
        raise ValueError(
            "{} contains an imaginary component as large as {}.".format(
                label,
                maximum_imaginary,
            )
        )
    result = np.asarray(array.real, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError("{} contains a non-finite value.".format(label))
    return result


def normalize_coefficients(coefficients):
    scale = float(np.max(np.abs(coefficients)))
    if scale <= 0.0:
        raise ValueError(
            "Cannot normalize Hamiltonian coefficients with zero scale."
        )
    return coefficients / scale


def normalize_coefficient_weighted_covariances(weighted_covariances):
    """Divide c_i c_j C_ij samples by their maximum absolute value."""

    values = real_array(
        weighted_covariances,
        "coefficient-weighted covariance samples to normalize",
        allow_empty=True,
    )
    if values.size == 0:
        return values

    maximum_absolute_value = float(np.max(np.abs(values)))
    if not math.isfinite(maximum_absolute_value):
        raise ValueError(
            "Cannot normalize coefficient-weighted covariances with a "
            "non-finite maximum absolute value."
        )
    if maximum_absolute_value == 0.0:
        raise ValueError(
            "Cannot normalize coefficient-weighted covariances because "
            "their maximum absolute value is zero."
        )
    normalized = values / maximum_absolute_value
    if not np.all(np.isfinite(normalized)):
        raise ValueError(
            "Normalized coefficient-weighted covariances contain a "
            "non-finite value."
        )
    return normalized


def clean_diagonal_variance(value, term_index):
    variance = real_scalar(value, "C_{}{}".format(term_index, term_index))
    if variance < 0.0 and abs(variance) <= ZERO_VARIANCE_TOLERANCE:
        return 0.0
    if variance < 0.0:
        raise ValueError(
            "Term {} has a negative diagonal covariance: {}.".format(
                term_index,
                variance,
            )
        )
    return variance


def covariance_samples(covariance_dictionary, off_diagonal_only=False):
    """Return signed raw covariances and defined correlation coefficients."""

    diagonal = {}
    for (left_index, right_index), value in covariance_dictionary.items():
        if left_index == right_index:
            diagonal[left_index] = clean_diagonal_variance(value, left_index)

    raw_values = []
    normalized_values = []
    skipped = 0
    for (left_index, right_index), value in covariance_dictionary.items():
        if off_diagonal_only and left_index == right_index:
            continue

        covariance = real_scalar(
            value,
            "C_{}{}".format(left_index, right_index),
        )
        raw_values.append(covariance)

        left_variance = diagonal[left_index]
        right_variance = diagonal[right_index]
        if (
            left_variance <= ZERO_VARIANCE_TOLERANCE
            or right_variance <= ZERO_VARIANCE_TOLERANCE
        ):
            skipped += 1
            continue
        denominator = math.sqrt(left_variance * right_variance)

        correlation = covariance / denominator
        if correlation < -1.0 and correlation >= -1.0 - REAL_TOLERANCE:
            correlation = -1.0
        elif correlation > 1.0 and correlation <= 1.0 + REAL_TOLERANCE:
            correlation = 1.0
        elif (
            correlation < -1.0 - REAL_TOLERANCE
            or correlation > 1.0 + REAL_TOLERANCE
        ):
            raise ValueError(
                "Normalized covariance C_{}{}/sqrt(C_{}{} C_{}{})={} lies "
                "outside [-1, 1].".format(
                    left_index,
                    right_index,
                    left_index,
                    left_index,
                    right_index,
                    right_index,
                    correlation,
                )
            )
        normalized_values.append(correlation)

    return (
        real_array(
            raw_values,
            "raw covariance samples",
            allow_empty=True,
        ),
        real_array(
            normalized_values,
            "normalized covariance samples",
            allow_empty=True,
        ),
        skipped,
    )


def coefficient_weighted_covariance_samples(
    covariance_dictionary,
    terms,
    off_diagonal_only=False,
):
    """Return signed c_i c_j C_ij samples for covariance-dictionary pairs."""

    coefficients_by_index = {}
    for term in terms:
        if term.index in coefficients_by_index:
            raise ValueError("Duplicate Pauli-term index {}.".format(term.index))
        coefficients_by_index[term.index] = real_scalar(
            term.coefficient,
            "c_{}".format(term.index),
        )

    weighted_values = []
    for pair, covariance_value in covariance_dictionary.items():
        left_index, right_index = pair
        if off_diagonal_only and left_index == right_index:
            continue
        try:
            left_coefficient = coefficients_by_index[left_index]
            right_coefficient = coefficients_by_index[right_index]
        except KeyError as error:
            raise ValueError(
                "Covariance pair ({}, {}) references Pauli-term index {} "
                "without a coefficient.".format(
                    left_index,
                    right_index,
                    error.args[0],
                )
            ) from error

        if left_index == right_index:
            covariance = clean_diagonal_variance(
                covariance_value,
                left_index,
            )
        else:
            covariance = real_scalar(
                covariance_value,
                "C_{}{}".format(left_index, right_index),
            )
        weighted_values.append(left_coefficient * right_coefficient * covariance)

    return real_array(
        weighted_values,
        "coefficient-weighted covariance samples",
        allow_empty=True,
    )


def estimate_dense_memory_gib(n_terms, n_qubits):
    """Estimate the principal dense arrays used by the fast covariance build."""

    dimension = 2**n_qubits
    complex_bytes = np.dtype(complex).itemsize
    state_bytes = dimension * complex_bytes
    action_bytes = n_terms * dimension * complex_bytes
    gram_bytes = n_terms * n_terms * complex_bytes
    return {
        "state": state_bytes / (1024**3),
        "actions": action_bytes / (1024**3),
        "action_copy": action_bytes / (1024**3),
        "gram": gram_bytes / (1024**3),
        "total": (state_bytes + 2 * action_bytes + gram_bytes) / (1024**3),
    }


def analyze_molecule(name, args):
    display_name = DISPLAY_NAMES[name]
    print("")
    print(
        "Loading {} Bravyi-Kitaev Hamiltonian...".format(display_name),
        flush=True,
    )
    qubit_operator, _ = load_qubit_hamiltonian(name)
    n_qubits = int(count_qubits(qubit_operator))
    terms = make_terms(qubit_operator, n_qubits)
    measurable_terms = [term for term in terms if term.pauli_tuple]

    memory = estimate_dense_memory_gib(len(measurable_terms), n_qubits)
    print(
        "{}: qubits={}, measurable_terms={}, estimated_dense_memory={:.3f} GiB "
        "(state={:.3f}, actions+temporary={:.3f}, gram={:.3f})".format(
            display_name,
            n_qubits,
            len(measurable_terms),
            memory["total"],
            memory["state"],
            memory["actions"] + memory["action_copy"],
            memory["gram"],
        ),
        flush=True,
    )
    if args.max_memory_gib and memory["total"] > args.max_memory_gib:
        raise MemoryError(
            "{} requires an estimated {:.3f} GiB for the principal dense "
            "covariance arrays, exceeding --max-memory-gib={:.3f}. Choose a "
            "smaller system or explicitly raise the guard if the machine has "
            "enough memory.".format(
                display_name,
                memory["total"],
                args.max_memory_gib,
            )
        )

    sparse_hamiltonian = get_sparse_operator(
        qubit_operator,
        n_qubits=n_qubits,
    )
    energy, state_vector = get_ground_state(sparse_hamiltonian)
    energy = real_scalar(energy, "{} energy".format(display_name))
    state_vector = np.asarray(state_vector, dtype=complex).reshape(-1)

    print(
        (
            "{}: {} energy={:.16g}; building covariance dictionary "
            "with {} worker(s)..."
        ).format(
            display_name,
            REFERENCE_METHOD,
            energy,
            args.cov_workers,
        ),
        flush=True,
    )
    covariance_start = time.perf_counter()
    covariance_dictionary, _ = build_covariance_dictionary(
        measurable_terms,
        state_vector,
        n_qubits,
        args.cov_workers,
        args.cov_chunksize,
    )
    covariance_runtime_s = time.perf_counter() - covariance_start

    coefficients_raw = real_array(
        [term.coefficient for term in measurable_terms],
        "{} Hamiltonian coefficients".format(display_name),
    )
    covariances_raw, covariances_normalized, skipped = covariance_samples(
        covariance_dictionary,
        off_diagonal_only=args.off_diagonal_only,
    )
    covariances_coefficient_weighted = coefficient_weighted_covariance_samples(
        covariance_dictionary,
        measurable_terms,
        off_diagonal_only=args.off_diagonal_only,
    )
    if covariances_coefficient_weighted.size != covariances_raw.size:
        raise RuntimeError(
            "Coefficient-weighted covariance sample count does not match "
            "the raw covariance sample count."
        )
    covariances_coefficient_weighted_normalized = (
        normalize_coefficient_weighted_covariances(
            covariances_coefficient_weighted
        )
    )
    if (
        covariances_coefficient_weighted_normalized.size
        != covariances_coefficient_weighted.size
    ):
        raise RuntimeError(
            "Normalized coefficient-weighted covariance sample count does "
            "not match the coefficient-weighted covariance sample count."
        )
    result = MoleculeDistributions(
        name=display_name,
        n_qubits=n_qubits,
        n_terms=len(measurable_terms),
        energy=energy,
        covariance_entries=len(covariance_dictionary),
        covariance_runtime_s=covariance_runtime_s,
        coefficients_raw=coefficients_raw,
        coefficients_normalized=normalize_coefficients(coefficients_raw),
        covariances_raw=covariances_raw,
        covariances_normalized=covariances_normalized,
        skipped_normalized_covariances=skipped,
        covariances_coefficient_weighted=covariances_coefficient_weighted,
        covariances_coefficient_weighted_normalized=(
            covariances_coefficient_weighted_normalized
        ),
    )
    print(
        "{}: covariance_entries={} runtime_s={:.6f}; plotted_raw={} "
        "plotted_normalized={} plotted_coefficient_weighted={} "
        "plotted_coefficient_weighted_normalized={} "
        "skipped_zero_variance={}".format(
            display_name,
            result.covariance_entries,
            result.covariance_runtime_s,
            result.covariances_raw.size,
            result.covariances_normalized.size,
            result.covariances_coefficient_weighted.size,
            result.covariances_coefficient_weighted_normalized.size,
            result.skipped_normalized_covariances,
        ),
        flush=True,
    )
    return result


def safe_filename_component(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def comparison_prefix(molecule_names):
    names = "_".join(safe_filename_component(name) for name in molecule_names)
    if len(molecule_names) > 1:
        names = "comparison_{}".format(names)
    return "loaded_bk_{}_{}".format(
        names,
        safe_filename_component(REFERENCE_METHOD),
    )


def has_kde_support(values):
    if values.size < 2:
        return False
    spread = float(np.ptp(values))
    scale = max(1.0, float(np.max(np.abs(values))))
    return spread > np.finfo(float).eps * scale


def visible_density_peak(x_values, density, visible_xlim=None):
    """Return the maximum sampled KDE density in the visible x interval."""

    x_values = np.asarray(x_values, dtype=float)
    density = np.asarray(density, dtype=float)
    visible_density = density
    if visible_xlim is not None:
        lower, upper = visible_xlim
        visible = (x_values >= lower) & (x_values <= upper)
        if np.any(visible):
            visible_density = density[visible]

    peak = float(np.max(visible_density))
    if not math.isfinite(peak) or peak <= 0.0:
        raise ValueError("Cannot peak-normalize a KDE with a nonpositive maximum.")
    return peak


def normalize_density_height(x_values, density, visible_xlim=None):
    """Scale a sampled KDE so its visible maximum is one."""

    density = np.asarray(density, dtype=float)
    return density / visible_density_peak(x_values, density, visible_xlim)


def common_histogram_bin_edges(distributions, attribute, fixed_xlim=None):
    """Return fixed-width bin edges shared by every molecule in one plot."""

    sample_arrays = []
    for distribution in distributions:
        values = np.asarray(getattr(distribution, attribute), dtype=float)
        if values.size:
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    "Histogram samples for {} contain a non-finite value."
                    .format(attribute)
                )
            sample_arrays.append(values)
    if not sample_arrays:
        return None

    pooled_size = sum(values.size for values in sample_arrays)
    bin_count = min(
        HISTOGRAM_MAX_BINS,
        max(1, int(math.ceil(math.sqrt(pooled_size)))),
    )

    if fixed_xlim is not None:
        lower, upper = map(float, fixed_xlim)
        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            raise ValueError("Histogram x-axis limits must be finite and ordered.")
        return np.linspace(lower, upper, bin_count + 1)

    pooled_values = (
        sample_arrays[0]
        if len(sample_arrays) == 1
        else np.concatenate(sample_arrays)
    )
    lower = float(np.min(pooled_values))
    upper = float(np.max(pooled_values))
    if lower == upper:
        padding = HISTOGRAM_CONSTANT_RANGE_PADDING * max(1.0, abs(lower))
        lower -= padding
        upper += padding
    return np.linspace(lower, upper, bin_count + 1)


def histogram_density(values, bin_edges):
    """Return full-sample density heights, preserving clipped-out mass."""

    values = np.asarray(values, dtype=float)
    bin_edges = np.asarray(bin_edges, dtype=float)
    if values.size == 0:
        return np.zeros(max(0, bin_edges.size - 1), dtype=float)

    counts, _ = np.histogram(values, bins=bin_edges)
    widths = np.diff(bin_edges)
    if np.any(widths <= 0.0):
        raise ValueError("Histogram bin edges must be strictly increasing.")
    return counts.astype(float) / (float(values.size) * widths)


def visible_histogram_peak(bin_edges, density, visible_xlim=None):
    """Return the maximum histogram height among bins intersecting the view."""

    bin_edges = np.asarray(bin_edges, dtype=float)
    density = np.asarray(density, dtype=float)
    visible_density = density
    if visible_xlim is not None:
        lower, upper = visible_xlim
        visible_bins = (bin_edges[:-1] < upper) & (bin_edges[1:] > lower)
        if np.any(visible_bins) and np.any(density[visible_bins] > 0.0):
            visible_density = density[visible_bins]

    peak = float(np.max(visible_density))
    if not math.isfinite(peak) or peak < 0.0:
        raise ValueError("Histogram density has an invalid visible maximum.")
    return peak


def kde_resolution(attribute, bw_adjust):
    """Return the requested KDE bandwidth and optional detailed grid size."""

    if attribute == DETAILED_KDE_ATTRIBUTE:
        return bw_adjust * DETAILED_KDE_BANDWIDTH_FACTOR, DETAILED_KDE_GRIDSIZE
    return bw_adjust, None


def distribution_color_map(distributions):
    """Assign stable Seaborn colorblind colors by first molecule appearance."""

    molecule_names = []
    for distribution in distributions:
        if distribution.name not in molecule_names:
            molecule_names.append(distribution.name)
    palette = sns.color_palette(DENSITY_PALETTE, n_colors=len(molecule_names))
    return dict(zip(molecule_names, palette))


def apply_no_grid_axis_style(axis):
    """Apply clean outward ticks and left/bottom spines to one axis."""

    axis.grid(False)
    axis.tick_params(
        axis="both",
        which="both",
        direction="out",
        bottom=True,
        left=True,
        top=False,
        right=False,
    )
    axis.xaxis.set_ticks_position("bottom")
    axis.yaxis.set_ticks_position("left")
    sns.despine(ax=axis, top=True, right=True, left=False, bottom=False)


def plot_distributions(
    distributions,
    attribute,
    x_label,
    output_path,
    bw_adjust,
    dpi,
    fixed_xlim=None,
    peak_normalized=False,
    show_bars=False,
    colors_by_name=None,
    no_grid=False,
):
    """Overlay filled KDEs and optional base-density bars, then save."""

    output_path = Path(output_path)
    # Relative density is an arbitrary unit-peak rescaling, so probability
    # histogram bars have no statistical meaning there. Keep this invariant
    # even for direct API callers that pass both flags.
    show_bars = bool(show_bars and not peak_normalized)

    sns.set_theme(style="ticks" if no_grid else "whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(10.5, 6.5))
    if colors_by_name is None:
        colors_by_name = distribution_color_map(distributions)
    histogram_bin_edges = (
        common_histogram_bin_edges(
            distributions,
            attribute,
            fixed_xlim=fixed_xlim,
        )
        if show_bars
        else None
    )
    effective_bw_adjust, gridsize = kde_resolution(attribute, bw_adjust)
    histogram_alpha = (
        HISTOGRAM_SINGLE_ALPHA
        if len(distributions) == 1
        else HISTOGRAM_MULTI_ALPHA
    )

    any_samples = False
    relative_y_max = 1.0
    for distribution in distributions:
        try:
            color = colors_by_name[distribution.name]
        except KeyError as error:
            raise ValueError(
                "No density color was assigned to molecule '{}'.".format(
                    distribution.name
                )
            ) from error
        values = np.asarray(getattr(distribution, attribute), dtype=float)
        label = "{} (n={:,})".format(distribution.name, values.size)
        bar_heights = (
            histogram_density(values, histogram_bin_edges)
            if values.size and histogram_bin_edges is not None
            else None
        )
        relative_bar_scale = 1.0
        if values.size == 0:
            axis.plot(
                [],
                [],
                color=color,
                linewidth=2.0,
                label=label + "; undefined",
            )
        elif has_kde_support(values):
            any_samples = True
            line_count = len(axis.lines)
            sns.kdeplot(
                x=values,
                ax=axis,
                label=label,
                color=color,
                fill=False,
                bw_adjust=effective_bw_adjust,
                common_norm=False,
                warn_singular=False,
                linewidth=2.0,
                gridsize=(
                    gridsize if gridsize is not None else DEFAULT_KDE_GRIDSIZE
                ),
                zorder=KDE_LINE_ZORDER,
            )
            if len(axis.lines) != line_count + 1:
                raise RuntimeError("Seaborn did not produce one KDE curve.")
            curve = axis.lines[-1]
            if peak_normalized:
                relative_bar_scale = visible_density_peak(
                    curve.get_xdata(),
                    curve.get_ydata(),
                    visible_xlim=fixed_xlim,
                )
                curve.set_ydata(curve.get_ydata() / relative_bar_scale)
            axis.fill_between(
                curve.get_xdata(),
                0.0,
                curve.get_ydata(),
                facecolor=color,
                edgecolor="none",
                alpha=KDE_FILL_ALPHA,
                label="_nolegend_",
                zorder=KDE_FILL_ZORDER,
            )
        else:
            any_samples = True
            constant_label = "{}; constant={:.6g}".format(label, values[0])
            if peak_normalized:
                axis.vlines(
                    float(values[0]),
                    0.0,
                    1.0,
                    color=color,
                    linewidth=2.0,
                    label=constant_label,
                )
            else:
                axis.axvline(
                    float(values[0]),
                    color=color,
                    linewidth=2.0,
                    label=constant_label,
                )
            if peak_normalized and bar_heights is not None:
                histogram_peak = visible_histogram_peak(
                    histogram_bin_edges,
                    bar_heights,
                    visible_xlim=fixed_xlim,
                )
                if histogram_peak > 0.0:
                    relative_bar_scale = histogram_peak

        if bar_heights is not None:
            if peak_normalized:
                bar_heights = bar_heights / relative_bar_scale
                relative_y_max = max(
                    relative_y_max,
                    visible_histogram_peak(
                        histogram_bin_edges,
                        bar_heights,
                        visible_xlim=fixed_xlim,
                    ),
                )
            axis.bar(
                histogram_bin_edges[:-1],
                bar_heights,
                width=np.diff(histogram_bin_edges),
                align="edge",
                color=color,
                edgecolor=color,
                linewidth=0.45,
                alpha=histogram_alpha,
                label="_nolegend_",
                zorder=1,
            )

    if not no_grid:
        axis.axvline(0.0, color="black", linewidth=0.9, alpha=0.45)
    if not any_samples:
        axis.text(
            0.5,
            0.5,
            "No defined samples",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
    if fixed_xlim is not None:
        axis.set_xlim(*fixed_xlim)
    axis.set_xlabel(x_label)
    if peak_normalized:
        axis.set_ylim(0.0, 1.05 * relative_y_max)
        axis.set_ylabel("Relative density")
    else:
        axis.set_ylabel("Probability density")
    if no_grid:
        apply_no_grid_axis_style(axis)
    axis.legend(title="Molecule", frameon=True)
    figure.tight_layout()
    output_stem = output_path.with_suffix("") if output_path.suffix else output_path
    output_paths = (
        output_stem.with_suffix(".png"),
        output_stem.with_suffix(".svg"),
    )

    for figure_path in output_paths:
        figure.savefig(
            figure_path,
            format=figure_path.suffix.lstrip("."),
            dpi=dpi,
            bbox_inches="tight",
        )
        print("Saved {}".format(figure_path), flush=True)
    plt.close(figure)
    return output_paths


def plot_histogram_only(
    distributions,
    attribute,
    x_label,
    output_path,
    dpi,
    fixed_xlim=None,
    colors_by_name=None,
    no_grid=False,
):
    """Overlay probability-density histograms and save both formats."""

    output_path = Path(output_path)
    sns.set_theme(style="ticks" if no_grid else "whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(10.5, 6.5))
    if colors_by_name is None:
        colors_by_name = distribution_color_map(distributions)
    histogram_bin_edges = common_histogram_bin_edges(
        distributions,
        attribute,
        fixed_xlim=fixed_xlim,
    )
    histogram_alpha = KDE_FILL_ALPHA

    any_samples = False
    legend_handles = []
    for distribution in distributions:
        try:
            color = colors_by_name[distribution.name]
        except KeyError as error:
            raise ValueError(
                "No density color was assigned to molecule '{}'.".format(
                    distribution.name
                )
            ) from error
        values = np.asarray(getattr(distribution, attribute), dtype=float)
        label = "{} (n={:,})".format(distribution.name, values.size)
        if values.size == 0:
            legend_handles.append(
                Patch(
                    facecolor=color,
                    edgecolor=color,
                    alpha=histogram_alpha,
                    label=label + "; undefined",
                )
            )
            continue

        any_samples = True
        legend_handles.append(
            Patch(
                facecolor=color,
                edgecolor=color,
                alpha=histogram_alpha,
                label=label,
            )
        )
        bar_heights = histogram_density(values, histogram_bin_edges)
        axis.bar(
            histogram_bin_edges[:-1],
            bar_heights,
            width=np.diff(histogram_bin_edges),
            align="edge",
            color=color,
            edgecolor=color,
            linewidth=0.45,
            alpha=histogram_alpha,
            label="_nolegend_",
            zorder=1,
        )

    if not no_grid:
        axis.axvline(0.0, color="black", linewidth=0.9, alpha=0.45)
    if not any_samples:
        axis.text(
            0.5,
            0.5,
            "No defined samples",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
    if fixed_xlim is not None:
        axis.set_xlim(*fixed_xlim)
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Probability density")
    if no_grid:
        apply_no_grid_axis_style(axis)
    axis.legend(handles=legend_handles, title="Molecule", frameon=True)
    figure.tight_layout()
    output_stem = output_path.with_suffix("") if output_path.suffix else output_path
    output_paths = (
        output_stem.with_suffix(".png"),
        output_stem.with_suffix(".svg"),
    )

    for figure_path in output_paths:
        figure.savefig(
            figure_path,
            format=figure_path.suffix.lstrip("."),
            dpi=dpi,
            bbox_inches="tight",
        )
        print("Saved {}".format(figure_path), flush=True)
    plt.close(figure)
    return output_paths


def make_all_plots(distributions, args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    colors_by_name = distribution_color_map(distributions)
    no_grid = bool(getattr(args, "no_grid", False))
    prefix = comparison_prefix([item.name for item in distributions])
    if args.off_diagonal_only:
        prefix += "_offdiag"
    plot_specs = (
        (
            "covariances_raw",
            "covariances_raw",
            "covariances_unnormalized",
            r"Covariance $C_{ij}$",
            None,
        ),
        (
            "covariances_normalized",
            "covariances_normalized",
            "covariances_normalized",
            r"Normalized covariance $C_{ij}/\sqrt{C_{ii}C_{jj}}$",
            DEFAULT_NORMALIZED_X_LIMITS,
        ),
        (
            "covariances_coefficient_weighted",
            "covariances_coefficient_weighted",
            "covariances_coefficient_weighted",
            r"Coefficient-weighted covariance $c_i c_j C_{ij}$",
            None,
        ),
        (
            "covariances_coefficient_weighted_normalized",
            "covariances_coefficient_weighted_normalized",
            "covariances_coefficient_weighted_normalized",
            r"Normalized coefficient-weighted covariance "
            r"$\frac{c_i c_j C_{ij}}{\max_{k,\ell}\left|c_k c_\ell C_{k\ell}\right|}$",
            DEFAULT_NORMALIZED_X_LIMITS,
        ),
        (
            "coefficients_raw",
            "coefficients_raw",
            "coefficients_unnormalized",
            r"Coefficient $c_i$",
            None,
        ),
        (
            "coefficients_normalized",
            "coefficients_normalized",
            "coefficients_normalized",
            r"Normalized coefficient $c_i/\max_j|c_j|$",
            DEFAULT_NORMALIZED_X_LIMITS,
        ),
    )
    if args.tight:
        plot_specs += (
            (
                "covariances_coefficient_weighted_normalized_tight",
                "covariances_coefficient_weighted_normalized",
                "covariances_coefficient_weighted_normalized_tight",
                r"Normalized coefficient-weighted covariance "
                r"$\frac{c_i c_j C_{ij}}{\max_{k,\ell}\left|c_k c_\ell C_{k\ell}\right|}$",
                TIGHT_WEIGHTED_NORMALIZED_X_LIMITS,
            ),
        )

    paths = {}
    for key, attribute, filename_component, x_label, fixed_xlim in plot_specs:
        output_stem = args.output_dir / "{}_{}".format(
            prefix,
            filename_component,
        )
        if args.bars:
            output_stem = output_stem.with_name(output_stem.name + "_bars")
        png_path, svg_path = plot_distributions(
            distributions,
            attribute,
            x_label,
            output_stem,
            args.bw_adjust,
            args.dpi,
            fixed_xlim=fixed_xlim,
            show_bars=args.bars,
            colors_by_name=colors_by_name,
            no_grid=no_grid,
        )
        paths[key] = png_path
        paths["{}_svg".format(key)] = svg_path
        if args.bars:
            bars_only_stem = args.output_dir / "{}_{}_bars_only".format(
                prefix,
                filename_component,
            )
            bars_only_png_path, bars_only_svg_path = plot_histogram_only(
                distributions,
                attribute,
                x_label,
                bars_only_stem,
                args.dpi,
                fixed_xlim=fixed_xlim,
                colors_by_name=colors_by_name,
                no_grid=no_grid,
            )
            bars_only_key = "{}_bars_only".format(key)
            paths[bars_only_key] = bars_only_png_path
            paths["{}_svg".format(bars_only_key)] = bars_only_svg_path

    if len(distributions) > 1:
        for key, attribute, filename_component, x_label, fixed_xlim in plot_specs:
            peak_key = "{}_unit_peak_density".format(key)
            output_stem = args.output_dir / "{}_{}_unit_peak_density".format(
                prefix,
                filename_component,
            )
            png_path, svg_path = plot_distributions(
                distributions,
                attribute,
                x_label,
                output_stem,
                args.bw_adjust,
                args.dpi,
                fixed_xlim=fixed_xlim,
                peak_normalized=True,
                show_bars=False,
                colors_by_name=colors_by_name,
                no_grid=no_grid,
            )
            paths[peak_key] = png_path
            paths["{}_svg".format(peak_key)] = svg_path
    return paths


def main(argv=None):
    args = parse_args(argv)
    print(
        "Molecules={}".format(
            ", ".join(DISPLAY_NAMES[name] for name in args.molecules)
        )
    )
    print("Hamiltonian mapping=Bravyi-Kitaev")
    print("Covariance wavefunction={}".format(REFERENCE_METHOD))
    print(
        "Normalized coefficient-weighted covariance standard x-axis={}".format(
            DEFAULT_NORMALIZED_X_LIMITS
        )
    )
    print(
        "Additional tight normalized coefficient-weighted covariance "
        "x-axis={}".format(
            TIGHT_WEIGHTED_NORMALIZED_X_LIMITS
            if args.tight
            else "disabled"
        )
    )
    print(
        "Normalized coefficient-weighted covariance KDE: "
        "bw_adjust={:.6g} ({} x requested --bw-adjust), gridsize={}".format(
            args.bw_adjust * DETAILED_KDE_BANDWIDTH_FACTOR,
            DETAILED_KDE_BANDWIDTH_FACTOR,
            DETAILED_KDE_GRIDSIZE,
        )
    )
    print(
        "Histogram bars={}".format(
            "enabled for probability-density plots as combined KDE-and-bar "
            "and separate histogram-only outputs (common adaptive bins, "
            "at most {}; density-scaled; standalone alpha={:.2f})".format(
                HISTOGRAM_MAX_BINS,
                KDE_FILL_ALPHA,
            )
            if args.bars
            else "disabled"
        )
    )
    print(
        "Plot axes={}".format(
            "clean outward ticks on bottom/left; no grid or zero-reference line"
            if args.no_grid
            else "whitegrid with zero-reference line"
        )
    )
    print(
        "Normalization: covariance=correlation coefficient; "
        "coefficient=max-absolute scaling; coefficient-weighted covariance="
        "raw coefficients; normalized coefficient-weighted covariance="
        "per-molecule maximum-absolute scaling over the selected "
        "covariance pairs."
    )

    distributions = []
    for molecule_name in args.molecules:
        distributions.append(analyze_molecule(molecule_name, args))

    paths = make_all_plots(distributions, args)
    print("")
    print(
        "Generated {} plots as {} PNG and {} SVG files in {}:".format(
            len(paths) // 2,
            sum(path.suffix == ".png" for path in paths.values()),
            sum(path.suffix == ".svg" for path in paths.values()),
            args.output_dir,
        )
    )
    for path in paths.values():
        print("  {}".format(path))
    return distributions, paths


if __name__ == "__main__":
    main()
