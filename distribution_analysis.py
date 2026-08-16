"""Compare Pauli-coefficient and covariance distributions across molecules.

For each requested Jordan-Wigner or Bravyi-Kitaev Hamiltonian, this script
constructs the selected reference wavefunction and the coefficient-free Pauli
covariance dictionary

    C_ij = <P_i P_j> - <P_i><P_j>

for the upper triangle of fully commuting, non-identity Pauli pairs.  It saves
eight probability-density figures: raw and normalized covariance
distributions, raw and normalized coefficient-weighted covariance
distributions, raw and normalized Hamiltonian-coefficient distributions, the
term variances C_ii = Var(P_i), and the coefficient-weighted term variances
c_i^2 C_ii.  A multi-molecule run also saves a unit-peak relative-density
companion for each figure.  The term-variance x-axis is fixed to [0, 1], while
the coefficient-weighted term variance uses a nonnegative auto-scaled range.
Normalized covariances are correlation coefficients
C_ij / sqrt(C_ii C_jj); normalized coefficients are scaled by max_i |c_i|.
The coefficient-weighted covariances use the original, unnormalized c_i.  For
each molecule, their normalized values are
(c_i c_j C_ij) / max_kl |c_k c_l C_kl|, where the maximum absolute value is
taken over the same selected covariance-pair population (including any
off-diagonal filter).
The constant identity term is excluded so all plots describe the same
measurable Pauli terms.  Term variances retain one diagonal C_ii per term even
when ``--off-diagonal-only`` removes diagonal pairs from the pair-covariance
plots.  Since P_i^2 = I, term variances are validated in [0, 1], with only
roundoff-sized endpoint excursions clamped.  The c_i^2 C_ii samples use the
original coefficients and are not normalized or restricted to [0, 1]; their
plots use a nonnegative, auto-scaled x range.  KDE evaluation for both
term-variance families is clipped at zero.  Raw and normalized distributions
retain their sign.
Every nonsingular KDE is drawn with a translucent fill, using consistent
Seaborn colorblind-palette molecule colors across all figures in one run.
Every plot is saved in both PNG and SVG format.
Use ``--tight`` to add a second normalized coefficient-weighted covariance plot
over the narrower x-axis interval [-0.25, 0.25], without replacing its standard
[-1.05, 1.05] plot.  Multi-molecule runs also receive a tight unit-peak
companion.  The unweighted term-variance plot uses one quarter of the usual KDE
bandwidth so it captures more detail.  The normalized coefficient-weighted
covariance plots retain half the usual bandwidth and a finer KDE grid.
``--bw-adjust`` remains the user-facing bandwidth multiplier.  Use ``--bars``
to overlay density-normalized
histogram bars on the probability-density KDE figures and additionally save a
combined bar-only density figure.  It also saves one count histogram per
requested molecule for every plot specification; these individual plots use
common comparison-wide bin edges.  The combined and bar-only filenames end in
``_bars`` and ``_bars_only``, while individual count histograms end in
``_<molecule>_count_histogram``.  Standalone bars use the same colorblind
molecule colors and fill intensity as the KDE areas; the combined overlays
remain lighter.  Unit-peak relative-density figures never include histogram
bars or bar-only companions.  Use ``--no-grid`` for clean tick-style axes with
outward ticks only on the bottom and left; this also hides the interior
zero-reference line.  By default, plots retain the white grid and zero-reference
line.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path


def configure_plotting_environment():
    """Select a non-interactive backend and writable cache directories."""

    os.environ["MPLBACKEND"] = "Agg"
    defaults = {
        "MPLCONFIGDIR": "/tmp/distribution_analysis_mplconfig",
        "XDG_CACHE_HOME": "/tmp/distribution_analysis_cache",
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
from matplotlib.ticker import MaxNLocator
from openfermion import QubitOperator
from openfermion.linalg import get_sparse_operator
from openfermion.utils import count_qubits
from tequila.hamiltonian import QubitHamiltonian

import gflow_vqe.hamiltonians as hamlib
from gflow_vqe.utils import get_variance_wavefunction


DEFAULT_OUTPUT_DIRECTORY = "distribution_analysis_plots"
DEFAULT_BANDWIDTH_ADJUSTMENT = 0.8
DEFAULT_KDE_GRIDSIZE = 200
DETAILED_KDE_BANDWIDTH_FACTOR = 0.5
DETAILED_KDE_GRIDSIZE = 1024
TERM_VARIANCE_KDE_BANDWIDTH_FACTOR = 0.25
HISTOGRAM_MAX_BINS = 100
HISTOGRAM_SINGLE_ALPHA = 0.28
HISTOGRAM_MULTI_ALPHA = 0.18
KDE_FILL_ALPHA = 0.38
KDE_FILL_ZORDER = 1.5
KDE_LINE_ZORDER = 2.5
DEFAULT_COVARIANCE_CHUNKSIZE = 128
DEFAULT_MAX_MEMORY_GIB = 8.0
DEFAULT_NORMALIZED_X_LIMITS = (-1.05, 1.05)
TERM_VARIANCE_X_LIMITS = (0.0, 1.0)
TIGHT_WEIGHTED_NORMALIZED_X_LIMITS = (-0.25, 0.25)
NONNEGATIVE_PLOT_ATTRIBUTES = frozenset(
    ("term_variances", "term_variances_coefficient_weighted")
)
REAL_TOLERANCE = 1.0e-9
ZERO_VARIANCE_TOLERANCE = 1.0e-12
JW_SYSTEM_NAMES = (
    "H2",
    "LiH",
    "MgO",
    "SiO",
    "N2",
    "H4",
    "H6",
    "BeH2",
    "H2O",
    "H2Os",
    "NH3",
)
BK_SYSTEM_NAMES = (
    "H2bk",
    "LiHbk",
    "N2bk",
    "H4bk",
    "H6bk",
    "BeH2bk",
    "H2Obk",
    "NH3bk",
)
SYSTEM_NAMES = JW_SYSTEM_NAMES + BK_SYSTEM_NAMES


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
    term_variances: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    term_variances_coefficient_weighted: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build Pauli covariance dictionaries and compare raw and "
            "normalized coefficient, covariance, coefficient-weighted "
            "covariance, term-variance, and coefficient-weighted "
            "term-variance distributions, saving every plot as PNG and SVG."
        )
    )
    parser.add_argument(
        "molecules",
        nargs="+",
        choices=SYSTEM_NAMES,
        help=(
            "One or more Jordan-Wigner or Bravyi-Kitaev molecule helpers. "
            "BK helper names end in 'bk'. Multiple molecules are overlaid "
            "in each output figure."
        ),
    )
    parser.add_argument(
        "--wfn",
        type=lambda value: str(value).upper(),
        default="FCI",
        choices=("FCI", "HF", "CISD"),
        help=(
            "Wavefunction used to construct each covariance dictionary "
            "(default: FCI)."
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
            "(default: distribution_analysis_plots)."
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
            "Seaborn KDE bandwidth multiplier (default: 0.8). The unweighted "
            "term-variance KDE additionally uses a 0.25 detail factor; the "
            "normalized coefficient-weighted covariance KDE uses a 0.5 "
            "detail factor."
        ),
    )
    parser.add_argument(
        "--off-diagonal-only",
        action="store_true",
        help=(
            "Exclude diagonal entries C_ii from covariance-pair plots. "
            "Dedicated term-variance plots still include every measurable "
            "Pauli term."
        ),
    )
    parser.add_argument(
        "--tight",
        action="store_true",
        help=(
            "Add a normalized coefficient-weighted covariance plot over "
            "[-0.25, 0.25], plus its unit-peak companion for multi-molecule "
            "runs. The standard [-1.05, 1.05] plots are still generated."
        ),
    )
    parser.add_argument(
        "--bars",
        action="store_true",
        help=(
            "Overlay density-normalized histogram bars on probability-density "
            "KDE figures and save combined bar-only density figures. Also "
            "save one count histogram per molecule and plot specification, "
            "using common comparison-wide bin edges. Combined and bar-only "
            "outputs end in '_bars' and '_bars_only'; individual outputs end "
            "in '_<molecule>_count_histogram'. Standalone bars match the KDE "
            "fill intensity, while combined overlays remain lighter. "
            "Unit-peak relative-density figures never show bars or receive "
            "bar-only companions."
        ),
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help=(
            "Remove interior gridlines and the zero-reference line from every "
            "plot, using outward ticks only on the bottom and left. The "
            "default keeps the white grid and zero-reference line."
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


def term_variance_samples(covariance_dictionary, terms):
    """Return C_ii and c_i^2 C_ii once per measurable Pauli term."""

    variances = []
    coefficient_weighted_variances = []
    seen_indices = set()
    for term in terms:
        term_index = term.index
        if term_index in seen_indices:
            raise ValueError(
                "Duplicate Pauli-term index {} while extracting diagonal "
                "variances.".format(term_index)
            )
        seen_indices.add(term_index)

        diagonal_key = (term_index, term_index)
        if diagonal_key not in covariance_dictionary:
            raise ValueError(
                "Missing diagonal covariance C_{}{} for measurable Pauli "
                "term {}.".format(term_index, term_index, term_index)
            )
        variance = clean_diagonal_variance(
            covariance_dictionary[diagonal_key],
            term_index,
        )
        if variance > 1.0:
            if variance - 1.0 <= ZERO_VARIANCE_TOLERANCE:
                variance = 1.0
            else:
                raise ValueError(
                    "Term {} has a diagonal covariance larger than one: {}."
                    .format(term_index, variance)
                )

        coefficient = real_scalar(
            term.coefficient,
            "c_{}".format(term_index),
        )
        variances.append(variance)
        coefficient_weighted_variances.append(
            coefficient * coefficient * variance
        )

    return (
        real_array(
            variances,
            "term variance samples",
            allow_empty=True,
        ),
        real_array(
            coefficient_weighted_variances,
            "coefficient-weighted term variance samples",
            allow_empty=True,
        ),
    )


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
    """Return signed c_i c_j C_ij samples using covariance-pair indices."""

    coefficients_by_index = {}
    for term in terms:
        if term.index in coefficients_by_index:
            raise ValueError("Duplicate Pauli-term index {}.".format(term.index))
        coefficients_by_index[term.index] = real_scalar(
            term.coefficient,
            "c_{}".format(term.index),
        )

    weighted_values = []
    for (left_index, right_index), value in covariance_dictionary.items():
        if off_diagonal_only and left_index == right_index:
            continue

        try:
            left_coefficient = coefficients_by_index[left_index]
        except KeyError as error:
            raise ValueError(
                "Missing Hamiltonian coefficient for covariance index {}."
                .format(left_index)
            ) from error
        try:
            right_coefficient = coefficients_by_index[right_index]
        except KeyError as error:
            raise ValueError(
                "Missing Hamiltonian coefficient for covariance index {}."
                .format(right_index)
            ) from error

        if left_index == right_index:
            covariance = clean_diagonal_variance(value, left_index)
        else:
            covariance = real_scalar(
                value,
                "C_{}{}".format(left_index, right_index),
            )

        weighted_values.append(
            left_coefficient * right_coefficient * covariance
        )

    return real_array(
        weighted_values,
        "coefficient-weighted covariance samples",
        allow_empty=True,
    )


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
    helper = getattr(hamlib, name, None)
    if helper is None or not callable(helper):
        raise ValueError("Unknown molecule helper '{}'.".format(name))

    print("")
    mapping = "Bravyi-Kitaev" if name in BK_SYSTEM_NAMES else "Jordan-Wigner"
    print(
        "Building {} {} Hamiltonian...".format(name, mapping),
        flush=True,
    )
    molecule, _, _, reported_n_paulis, qubit_operator = helper()
    n_qubits = int(count_qubits(qubit_operator))
    terms = make_terms(qubit_operator, n_qubits)
    measurable_terms = [term for term in terms if term.pauli_tuple]
    if reported_n_paulis != len(measurable_terms):
        raise ValueError(
            "{} reports {} measurable terms, but {} were found.".format(
                name,
                reported_n_paulis,
                len(measurable_terms),
            )
        )

    memory = estimate_dense_memory_gib(len(measurable_terms), n_qubits)
    print(
        "{}: qubits={}, measurable_terms={}, estimated_dense_memory={:.3f} GiB "
        "(state={:.3f}, actions+temporary={:.3f}, gram={:.3f})".format(
            name,
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
                name,
                memory["total"],
                args.max_memory_gib,
            )
        )

    sparse_hamiltonian = get_sparse_operator(
        qubit_operator,
        n_qubits=n_qubits,
    )
    energy, state_vector = get_variance_wavefunction(
        molecule,
        qubit_operator,
        method=args.wfn,
        sparse_hamiltonian=sparse_hamiltonian,
    )
    energy = real_scalar(energy, "{} energy".format(name))
    state_vector = np.asarray(state_vector, dtype=complex).reshape(-1)

    print(
        (
            "{}: {} energy={:.16g}; building covariance dictionary "
            "with {} worker(s)..."
        ).format(
            name,
            args.wfn,
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
        "{} Hamiltonian coefficients".format(name),
    )
    (
        term_variances,
        term_variances_coefficient_weighted,
    ) = term_variance_samples(
        covariance_dictionary,
        measurable_terms,
    )
    if term_variances.size != len(measurable_terms):
        raise RuntimeError(
            "Term-variance and measurable-term sample counts differ."
        )
    if term_variances_coefficient_weighted.size != term_variances.size:
        raise RuntimeError(
            "Coefficient-weighted and raw term-variance sample counts differ."
        )
    covariances_raw, covariances_normalized, skipped = covariance_samples(
        covariance_dictionary,
        off_diagonal_only=args.off_diagonal_only,
    )
    covariances_coefficient_weighted = (
        coefficient_weighted_covariance_samples(
            covariance_dictionary,
            measurable_terms,
            off_diagonal_only=args.off_diagonal_only,
        )
    )
    covariances_coefficient_weighted_normalized = (
        normalize_coefficient_weighted_covariances(
            covariances_coefficient_weighted
        )
    )
    if covariances_coefficient_weighted.size != covariances_raw.size:
        raise RuntimeError(
            "Coefficient-weighted and raw covariance sample counts differ."
        )
    if (
        covariances_coefficient_weighted_normalized.size
        != covariances_coefficient_weighted.size
    ):
        raise RuntimeError(
            "Normalized and raw coefficient-weighted covariance sample "
            "counts differ."
        )
    result = MoleculeDistributions(
        name=name,
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
        covariances_coefficient_weighted=(
            covariances_coefficient_weighted
        ),
        covariances_coefficient_weighted_normalized=(
            covariances_coefficient_weighted_normalized
        ),
        term_variances=term_variances,
        term_variances_coefficient_weighted=(
            term_variances_coefficient_weighted
        ),
    )
    print(
        "{}: covariance_entries={} runtime_s={:.6f}; plotted_raw={} "
        "plotted_normalized={} plotted_coefficient_weighted={} "
        "plotted_coefficient_weighted_normalized={} "
        "plotted_term_variances={} "
        "plotted_coefficient_weighted_term_variances={} "
        "skipped_zero_variance={}".format(
            name,
            result.covariance_entries,
            result.covariance_runtime_s,
            result.covariances_raw.size,
            result.covariances_normalized.size,
            result.covariances_coefficient_weighted.size,
            result.covariances_coefficient_weighted_normalized.size,
            result.term_variances.size,
            result.term_variances_coefficient_weighted.size,
            result.skipped_normalized_covariances,
        ),
        flush=True,
    )
    return result


def safe_filename_component(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def comparison_prefix(molecule_names, wfn):
    names = "_".join(safe_filename_component(name) for name in molecule_names)
    if len(molecule_names) > 1:
        names = "comparison_{}".format(names)
    return "{}_{}".format(names, safe_filename_component(wfn))


def has_kde_support(values):
    if values.size < 2:
        return False
    spread = float(np.ptp(values))
    scale = max(1.0, float(np.max(np.abs(values))))
    return spread > np.finfo(float).eps * scale


def visible_density_peak(x_values, density, visible_xlim=None):
    """Return the maximum sampled KDE height inside the visible x window."""

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
    """Return shared, uniformly spaced histogram edges for one figure."""

    sample_count = 0
    lower = math.inf
    upper = -math.inf
    for distribution in distributions:
        values = np.asarray(getattr(distribution, attribute), dtype=float)
        if values.size == 0:
            continue
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Cannot plot histogram bars for non-finite {} values."
                .format(attribute)
            )
        sample_count += values.size
        if fixed_xlim is None:
            lower = min(lower, float(np.min(values)))
            upper = max(upper, float(np.max(values)))

    if sample_count == 0:
        return None

    if fixed_xlim is not None:
        lower, upper = map(float, fixed_xlim)
        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            raise ValueError("Histogram x-axis limits must be finite and ordered.")

    bin_count = min(
        HISTOGRAM_MAX_BINS,
        max(1, math.ceil(math.sqrt(sample_count))),
    )
    if lower == upper:
        half_width = max(1.0, abs(lower)) * 0.025
        lower -= half_width
        upper += half_width
    return np.linspace(lower, upper, bin_count + 1)


def histogram_density(values, bin_edges):
    """Return probability-density heights for fixed histogram edges."""

    values = np.asarray(values, dtype=float)
    counts = histogram_counts(values, bin_edges)
    widths = np.diff(bin_edges)
    return counts.astype(float) / (values.size * widths)


def histogram_counts(values, bin_edges):
    """Return unnormalized bin counts for fixed histogram edges."""

    values = np.asarray(values, dtype=float)
    counts, _ = np.histogram(values, bins=bin_edges)
    return counts


def figure_output_paths(output_path):
    """Return the PNG and SVG paths for one plot."""

    output_path = Path(output_path)
    return (output_path.with_suffix(".png"), output_path.with_suffix(".svg"))


def distribution_color_map(distributions):
    """Assign the Seaborn colorblind cycle by first molecule appearance."""

    molecule_names = []
    for distribution in distributions:
        if distribution.name not in molecule_names:
            molecule_names.append(distribution.name)
    palette = sns.color_palette("colorblind", n_colors=len(molecule_names))
    return dict(zip(molecule_names, palette))


def configure_axes_style(axis, no_grid):
    """Apply the optional clean, outward-tick axes style."""

    if not no_grid:
        return
    axis.grid(False, which="both", axis="both")
    axis.tick_params(
        axis="both",
        which="both",
        direction="out",
        bottom=True,
        left=True,
        top=False,
        right=False,
        labeltop=False,
        labelright=False,
    )
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
    bars=False,
    kde_bandwidth_factor=1.0,
    kde_gridsize=DEFAULT_KDE_GRIDSIZE,
    colors_by_name=None,
    no_grid=False,
    nonnegative=False,
):
    """Overlay KDEs/bars, optionally clipping support to nonnegative x."""

    sns.set_theme(style="ticks" if no_grid else "whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(10.5, 6.5))
    if colors_by_name is None:
        colors_by_name = distribution_color_map(distributions)
    show_probability_density_bars = bool(bars and not peak_normalized)
    histogram_edges = (
        common_histogram_bin_edges(
            distributions,
            attribute,
            fixed_xlim=fixed_xlim,
        )
        if show_probability_density_bars
        else None
    )
    histogram_alpha = (
        HISTOGRAM_SINGLE_ALPHA
        if len(distributions) == 1
        else HISTOGRAM_MULTI_ALPHA
    )
    any_samples = False
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
            axis.plot(
                [],
                [],
                color=color,
                linewidth=2.0,
                label=label + "; undefined",
            )
            continue

        any_samples = True
        bar_density = None
        if histogram_edges is not None:
            bar_density = histogram_density(values, histogram_edges)

        if has_kde_support(values):
            line_count = len(axis.lines)
            kde_support_options = {}
            if nonnegative:
                kde_support_options["clip"] = (0.0, np.inf)
            sns.kdeplot(
                x=values,
                ax=axis,
                label=label,
                color=color,
                fill=False,
                bw_adjust=bw_adjust * kde_bandwidth_factor,
                gridsize=kde_gridsize,
                common_norm=False,
                warn_singular=False,
                linewidth=2.0,
                zorder=KDE_LINE_ZORDER,
                **kde_support_options,
            )
            if len(axis.lines) != line_count + 1:
                raise RuntimeError("Seaborn did not produce one KDE curve.")
            curve = axis.lines[-1]
            if peak_normalized:
                kde_peak = visible_density_peak(
                    curve.get_xdata(),
                    curve.get_ydata(),
                    visible_xlim=fixed_xlim,
                )
                curve.set_ydata(curve.get_ydata() / kde_peak)
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

        if bar_density is not None:
            axis.bar(
                histogram_edges[:-1],
                bar_density,
                width=np.diff(histogram_edges),
                align="edge",
                color=color,
                edgecolor=color,
                linewidth=0.5,
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
    elif nonnegative:
        axis.set_xlim(left=0.0)
    axis.set_xlabel(x_label)
    if peak_normalized:
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("Relative density")
    else:
        axis.set_ylabel("Probability density")
    configure_axes_style(axis, no_grid)
    axis.legend(title="Molecule", frameon=True)
    figure.tight_layout()
    saved_paths = figure_output_paths(output_path)
    for saved_path in saved_paths:
        figure.savefig(
            saved_path,
            format=saved_path.suffix.lstrip("."),
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(figure)
    for saved_path in saved_paths:
        print("Saved {}".format(saved_path), flush=True)
    return saved_paths


def plot_histograms(
    distributions,
    attribute,
    x_label,
    output_path,
    dpi,
    fixed_xlim=None,
    colors_by_name=None,
    no_grid=False,
    nonnegative=False,
):
    """Overlay histograms, optionally keeping the auto x-axis nonnegative."""

    sns.set_theme(style="ticks" if no_grid else "whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(10.5, 6.5))
    if colors_by_name is None:
        colors_by_name = distribution_color_map(distributions)
    histogram_edges = common_histogram_bin_edges(
        distributions,
        attribute,
        fixed_xlim=fixed_xlim,
    )

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
        legend_label = label + "; undefined" if values.size == 0 else label
        legend_handles.append(
            Patch(
                facecolor=color,
                edgecolor=color,
                linewidth=0.5,
                alpha=KDE_FILL_ALPHA,
                label=legend_label,
            )
        )
        if values.size == 0:
            continue

        any_samples = True
        bar_density = histogram_density(values, histogram_edges)
        axis.bar(
            histogram_edges[:-1],
            bar_density,
            width=np.diff(histogram_edges),
            align="edge",
            color=color,
            edgecolor=color,
            linewidth=0.5,
            alpha=KDE_FILL_ALPHA,
            label="_nolegend_",
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
    elif nonnegative:
        axis.set_xlim(left=0.0)
    axis.set_xlabel(x_label)
    axis.set_ylim(bottom=0.0)
    axis.set_ylabel("Probability density")
    configure_axes_style(axis, no_grid)
    axis.legend(handles=legend_handles, title="Molecule", frameon=True)
    figure.tight_layout()

    saved_paths = figure_output_paths(output_path)
    for saved_path in saved_paths:
        figure.savefig(
            saved_path,
            format=saved_path.suffix.lstrip("."),
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(figure)
    for saved_path in saved_paths:
        print("Saved {}".format(saved_path), flush=True)
    return saved_paths


def plot_individual_count_histogram(
    distribution,
    attribute,
    x_label,
    output_path,
    dpi,
    histogram_edges,
    color,
    fixed_xlim=None,
    no_grid=False,
    nonnegative=False,
):
    """Save one molecule's count histogram using comparison-wide bins."""

    sns.set_theme(style="ticks" if no_grid else "whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(10.5, 6.5))
    values = np.asarray(getattr(distribution, attribute), dtype=float)
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError(
            "Cannot plot histogram counts for non-finite {} values."
            .format(attribute)
        )

    color_patch = Patch(
        facecolor=color,
        edgecolor=color,
        linewidth=0.5,
        alpha=KDE_FILL_ALPHA,
        label="{} (n={:,}){}".format(
            distribution.name,
            values.size,
            "; undefined" if values.size == 0 else "",
        ),
    )
    if values.size == 0:
        axis.text(
            0.5,
            0.5,
            "No defined samples",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
    else:
        if histogram_edges is None:
            raise ValueError(
                "Shared histogram bin edges are required for nonempty values."
            )
        counts = histogram_counts(values, histogram_edges)
        axis.bar(
            histogram_edges[:-1],
            counts,
            width=np.diff(histogram_edges),
            align="edge",
            color=color,
            edgecolor=color,
            linewidth=0.5,
            alpha=KDE_FILL_ALPHA,
            label="_nolegend_",
        )

    if not no_grid:
        axis.axvline(0.0, color="black", linewidth=0.9, alpha=0.45)
    if fixed_xlim is not None:
        axis.set_xlim(*fixed_xlim)
    elif nonnegative:
        axis.set_xlim(left=0.0)
    axis.set_xlabel(x_label)
    axis.set_ylim(bottom=0.0)
    axis.set_ylabel("Count")
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    configure_axes_style(axis, no_grid)
    axis.legend(handles=[color_patch], title="Molecule", frameon=True)
    figure.tight_layout()

    saved_paths = figure_output_paths(output_path)
    for saved_path in saved_paths:
        figure.savefig(
            saved_path,
            format=saved_path.suffix.lstrip("."),
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(figure)
    for saved_path in saved_paths:
        print("Saved {}".format(saved_path), flush=True)
    return saved_paths


def make_all_plots(distributions, args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    colors_by_name = distribution_color_map(distributions)
    no_grid = bool(getattr(args, "no_grid", False))
    prefix = comparison_prefix([item.name for item in distributions], args.wfn)
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
        (
            "term_variances",
            "term_variances",
            "term_variances",
            r"Term variance $\mathrm{Var}(P_i)=C_{ii}$",
            TERM_VARIANCE_X_LIMITS,
        ),
        (
            "term_variances_coefficient_weighted",
            "term_variances_coefficient_weighted",
            "term_variances_coefficient_weighted",
            r"Coefficient-weighted term variance "
            r"$c_i^2\mathrm{Var}(P_i)=c_i^2 C_{ii}$",
            None,
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
        detailed_kde = (
            attribute == "covariances_coefficient_weighted_normalized"
        )
        term_variance_kde = attribute == "term_variances"
        kde_bandwidth_factor = 1.0
        if detailed_kde:
            kde_bandwidth_factor = DETAILED_KDE_BANDWIDTH_FACTOR
        elif term_variance_kde:
            kde_bandwidth_factor = TERM_VARIANCE_KDE_BANDWIDTH_FACTOR
        nonnegative = attribute in NONNEGATIVE_PLOT_ATTRIBUTES
        output_path = args.output_dir / "{}_{}".format(
            prefix,
            filename_component,
        )
        if args.bars:
            output_path = output_path.with_name(output_path.name + "_bars")
        saved_paths = plot_distributions(
            distributions,
            attribute,
            x_label,
            output_path,
            args.bw_adjust,
            args.dpi,
            fixed_xlim=fixed_xlim,
            bars=args.bars,
            kde_bandwidth_factor=kde_bandwidth_factor,
            kde_gridsize=(
                DETAILED_KDE_GRIDSIZE
                if detailed_kde
                else DEFAULT_KDE_GRIDSIZE
            ),
            colors_by_name=colors_by_name,
            no_grid=no_grid,
            nonnegative=nonnegative,
        )
        png_path, svg_path = saved_paths
        paths[key] = png_path
        paths["{}_svg".format(key)] = svg_path
        if args.bars:
            histogram_edges = common_histogram_bin_edges(
                distributions,
                attribute,
                fixed_xlim=fixed_xlim,
            )
            bars_only_key = "{}_bars_only".format(key)
            bars_only_output_path = args.output_dir / "{}_{}_bars_only".format(
                prefix,
                filename_component,
            )
            bars_only_paths = plot_histograms(
                distributions,
                attribute,
                x_label,
                bars_only_output_path,
                args.dpi,
                fixed_xlim=fixed_xlim,
                colors_by_name=colors_by_name,
                no_grid=no_grid,
                nonnegative=nonnegative,
            )
            bars_only_png_path, bars_only_svg_path = bars_only_paths
            paths[bars_only_key] = bars_only_png_path
            paths["{}_svg".format(bars_only_key)] = bars_only_svg_path

            for distribution in distributions:
                molecule_component = safe_filename_component(
                    distribution.name
                )
                if not molecule_component:
                    raise ValueError(
                        "Molecule '{}' has no filename-safe characters."
                        .format(distribution.name)
                    )
                count_key = "{}_{}_count_histogram".format(
                    key,
                    molecule_component,
                )
                if count_key in paths:
                    raise ValueError(
                        "Duplicate count-histogram path key '{}'.".format(
                            count_key
                        )
                    )
                try:
                    color = colors_by_name[distribution.name]
                except KeyError as error:
                    raise ValueError(
                        "No density color was assigned to molecule '{}'."
                        .format(distribution.name)
                    ) from error
                count_output_path = args.output_dir / (
                    "{}_{}_{}_count_histogram".format(
                        prefix,
                        filename_component,
                        molecule_component,
                    )
                )
                count_paths = plot_individual_count_histogram(
                    distribution,
                    attribute,
                    x_label,
                    count_output_path,
                    args.dpi,
                    histogram_edges,
                    color,
                    fixed_xlim=fixed_xlim,
                    no_grid=no_grid,
                    nonnegative=nonnegative,
                )
                count_png_path, count_svg_path = count_paths
                paths[count_key] = count_png_path
                paths["{}_svg".format(count_key)] = count_svg_path

    if len(distributions) > 1:
        for key, attribute, filename_component, x_label, fixed_xlim in plot_specs:
            peak_key = "{}_unit_peak_density".format(key)
            detailed_kde = (
                attribute == "covariances_coefficient_weighted_normalized"
            )
            term_variance_kde = attribute == "term_variances"
            kde_bandwidth_factor = 1.0
            if detailed_kde:
                kde_bandwidth_factor = DETAILED_KDE_BANDWIDTH_FACTOR
            elif term_variance_kde:
                kde_bandwidth_factor = TERM_VARIANCE_KDE_BANDWIDTH_FACTOR
            nonnegative = attribute in NONNEGATIVE_PLOT_ATTRIBUTES
            output_path = args.output_dir / "{}_{}_unit_peak_density".format(
                prefix,
                filename_component,
            )
            saved_paths = plot_distributions(
                distributions,
                attribute,
                x_label,
                output_path,
                args.bw_adjust,
                args.dpi,
                fixed_xlim=fixed_xlim,
                peak_normalized=True,
                bars=False,
                kde_bandwidth_factor=kde_bandwidth_factor,
                kde_gridsize=(
                    DETAILED_KDE_GRIDSIZE
                    if detailed_kde
                    else DEFAULT_KDE_GRIDSIZE
                ),
                colors_by_name=colors_by_name,
                no_grid=no_grid,
                nonnegative=nonnegative,
            )
            png_path, svg_path = saved_paths
            paths[peak_key] = png_path
            paths["{}_svg".format(peak_key)] = svg_path
    return paths


def main(argv=None):
    args = parse_args(argv)
    print("Molecules={}".format(", ".join(args.molecules)))
    print("Covariance wavefunction={}".format(args.wfn))
    print(
        "Normalized coefficient-weighted covariance x-axis="
        "standard {}; tight zoom={}".format(
            DEFAULT_NORMALIZED_X_LIMITS,
            (
                TIGHT_WEIGHTED_NORMALIZED_X_LIMITS
                if args.tight
                else "disabled"
            ),
        )
    )
    print(
        "Term variance x-axis={}; coefficient-weighted term variance "
        "x-axis=nonnegative auto-scaled range".format(
            TERM_VARIANCE_X_LIMITS
        )
    )
    print(
        "Unweighted term-variance KDE: bw_adjust={:.6g} "
        "({} x --bw-adjust)".format(
            args.bw_adjust * TERM_VARIANCE_KDE_BANDWIDTH_FACTOR,
            TERM_VARIANCE_KDE_BANDWIDTH_FACTOR,
        )
    )
    print(
        "Normalized coefficient-weighted covariance KDE: "
        "bw_adjust={:.6g} ({} x --bw-adjust), gridsize={}".format(
            args.bw_adjust * DETAILED_KDE_BANDWIDTH_FACTOR,
            DETAILED_KDE_BANDWIDTH_FACTOR,
            DETAILED_KDE_GRIDSIZE,
        )
    )
    print(
        "Density histogram bars={}".format(
            "enabled as KDE overlays and separate bar-only probability-density "
            "plots, plus one count histogram per molecule and plot "
            "specification (shared comparison bins/colors; standalone "
            "alpha={:.2f}; combined overlays remain lighter)".format(
                KDE_FILL_ALPHA
            )
            if args.bars
            else "disabled"
        )
    )
    print(
        "Plot grid={}".format(
            "disabled (outward bottom/left ticks; zero reference hidden)"
            if args.no_grid
            else "enabled (zero reference shown)"
        )
    )
    print(
        "Normalization: covariance=correlation coefficient; "
        "coefficient=max-absolute scaling; coefficient-weighted covariance="
        "raw coefficients; normalized coefficient-weighted covariance="
        "per-molecule maximum-absolute scaling over the selected pairs; "
        "term variance=C_ii; coefficient-weighted term variance="
        "raw c_i^2 C_ii."
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
