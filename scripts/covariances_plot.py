"""Generate individual and route-level off-diagonal covariance KDE plots.

The fixed batch covers the requested Jordan-Wigner and Bravyi-Kitaev molecule
helpers.  Fully commuting covariances are built once per molecule; QWC data are
then selected as a subset of those same covariance moments.  Every individual
plot is a probability-density KDE without histogram bars.  Each route also
receives a multi-molecule companion whose KDEs are independently scaled to a
visible maximum of one.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path


def configure_plotting_environment():
    """Select a non-interactive backend and writable cache directories."""

    os.environ["MPLBACKEND"] = "Agg"
    defaults = {
        "MPLCONFIGDIR": "/tmp/covariances_plot_mplconfig",
        "XDG_CACHE_HOME": "/tmp/covariances_plot_cache",
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
from openfermion import QubitOperator
from openfermion.linalg import get_sparse_operator
from openfermion.utils import count_qubits
from tequila.hamiltonian import QubitHamiltonian

import gflow_vqe.hamiltonians as hamlib
from gflow_vqe.utils import get_variance_wavefunction


DEFAULT_BANDWIDTH_ADJUSTMENT = 0.8
DEFAULT_KDE_GRIDSIZE = 200
HISTOGRAM_MAX_BINS = 100
HISTOGRAM_SINGLE_ALPHA = 0.28
HISTOGRAM_MULTI_ALPHA = 0.18
KDE_FILL_ALPHA = 0.38
KDE_FILL_ZORDER = 1.5
KDE_LINE_ZORDER = 2.5
DEFAULT_COVARIANCE_CHUNKSIZE = 128
DEFAULT_MAX_MEMORY_GIB = 8.0
DEFAULT_NORMALIZED_X_LIMITS = (-1.05, 1.05)
REAL_TOLERANCE = 1.0e-8
ZERO_VARIANCE_TOLERANCE = 1.0e-12


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

DEFAULT_OUTPUT_DIRECTORY = "covariances_plots"
JW_SYSTEM_NAMES = ("H4", "LiH", "BeH2", "H2O", "N2", "SiO", "MgO")
BK_SYSTEM_NAMES = ("H4bk", "LiHbk", "BeH2bk", "H2Obk", "N2bk")
QWC_EXCLUDED_SYSTEM_NAMES = frozenset(("SiO", "MgO"))
ROUTE_NAMES = ("JW_FC", "JW_QWC", "BK_FC", "BK_QWC")

# The six requested colors follow the established molecule color sequence.
# MgO was not specified, so it receives the next color in that sequence.
MOLECULE_COLORS = {
    "H4": "#ff7f0e",
    "LiH": "#2ca02c",
    "BeH2": "#d62728",
    "H2O": "#9467bd",
    "N2": "#8c564b",
    "SiO": "#e377c2",
    "MgO": "#7f7f7f",
}

PLOT_SPECS = (
    (
        "covariances_raw",
        "covariances_unnormalized",
        r"Covariance $C_{ij}$",
        None,
    ),
)


def default_cov_workers():
    return max(1, min(8, os.cpu_count() or 1))


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


@dataclass
class CovarianceDistribution:
    name: str
    condition: str
    covariance_entries: int
    covariances_raw: np.ndarray


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate the requested individual FC/QWC off-diagonal covariance "
            "KDE plots and unit-peak route comparisons."
        )
    )
    parser.add_argument(
        "--wfn",
        type=lambda value: str(value).upper(),
        default="FCI",
        choices=("FCI", "HF", "CISD"),
        help="Wavefunction used to construct covariance moments (default: FCI).",
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
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIRECTORY),
        help=(
            "Root directory containing JW_FC, JW_QWC, BK_FC, and BK_QWC "
            "(default: covariances_plots)."
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
        help="Seaborn KDE bandwidth multiplier (default: 0.8).",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Use the existing clean outward-tick style without a grid or zero line.",
    )
    parser.add_argument(
        "--max-memory-gib",
        type=float,
        default=DEFAULT_MAX_MEMORY_GIB,
        help=(
            "Abort when principal dense covariance arrays exceed this estimate; "
            "use 0 to disable the guard (default: 8)."
        ),
    )
    args = parser.parse_args(argv)

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


def base_molecule_name(name):
    return name[:-2] if name.endswith("bk") else name


def molecule_color(name):
    base_name = base_molecule_name(name)
    try:
        return MOLECULE_COLORS[base_name]
    except KeyError as error:
        raise ValueError("No plot color is configured for '{}'.".format(name)) from error


def terms_qubit_wise_commute(left, right):
    return all(
        left_axis == "I" or right_axis == "I" or left_axis == right_axis
        for left_axis, right_axis in zip(left.ops, right.ops)
    )


def qwc_covariance_dictionary(fc_covariances, terms):
    terms_by_index = {term.index: term for term in terms}
    if len(terms_by_index) != len(terms):
        raise ValueError("Duplicate Pauli-term indices were found.")

    selected = {}
    for pair, covariance in fc_covariances.items():
        left_index, right_index = pair
        try:
            left = terms_by_index[left_index]
            right = terms_by_index[right_index]
        except KeyError as error:
            raise ValueError(
                "Covariance pair {} references an unknown Pauli term.".format(pair)
            ) from error
        if terms_qubit_wise_commute(left, right):
            selected[pair] = covariance
    return selected


def build_distribution(name, condition, covariance_dictionary):
    raw = real_array(
        [
            real_scalar(value, "C_{}{}".format(left_index, right_index))
            for (left_index, right_index), value in covariance_dictionary.items()
            if left_index != right_index
        ],
        "off-diagonal covariance samples",
        allow_empty=True,
    )

    return CovarianceDistribution(
        name=name,
        condition=condition,
        covariance_entries=len(covariance_dictionary),
        covariances_raw=raw,
    )


def analyze_molecule(name, args):
    helper = getattr(hamlib, name, None)
    if helper is None or not callable(helper):
        raise ValueError("Unknown molecule helper '{}'.".format(name))

    mapping = "BK" if name.endswith("bk") else "JW"
    print("", flush=True)
    print("Building {} {} Hamiltonian...".format(name, mapping), flush=True)
    molecule, _, _, reported_n_paulis, qubit_operator = helper()
    n_qubits = int(count_qubits(qubit_operator))
    all_terms = make_terms(qubit_operator, n_qubits)
    measurable_terms = [term for term in all_terms if term.pauli_tuple]
    if reported_n_paulis != len(measurable_terms):
        raise ValueError(
            "{} reports {} measurable terms, but {} were found.".format(
                name,
                reported_n_paulis,
                len(measurable_terms),
            )
        )

    memory = estimate_dense_memory_gib(
        len(measurable_terms),
        n_qubits,
    )
    print(
        "{}: qubits={}, measurable_terms={}, estimated_dense_memory={:.3f} GiB".format(
            name,
            n_qubits,
            len(measurable_terms),
            memory["total"],
        ),
        flush=True,
    )
    if args.max_memory_gib and memory["total"] > args.max_memory_gib:
        raise MemoryError(
            "{} requires an estimated {:.3f} GiB, exceeding "
            "--max-memory-gib={:.3f}.".format(
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
    energy = real_scalar(
        energy,
        "{} energy".format(name),
    )
    state_vector = np.asarray(state_vector, dtype=complex).reshape(-1)

    print(
        "{}: {} energy={:.16g}; building FC covariance moments with {} worker(s)...".format(
            name,
            args.wfn,
            energy,
            args.cov_workers,
        ),
        flush=True,
    )
    start = time.perf_counter()
    fc_covariances, _ = build_covariance_dictionary(
        measurable_terms,
        state_vector,
        n_qubits,
        args.cov_workers,
        args.cov_chunksize,
    )
    runtime = time.perf_counter() - start
    print(
        "{}: FC covariance_entries={} runtime_s={:.6f}".format(
            name,
            len(fc_covariances),
            runtime,
        ),
        flush=True,
    )
    return measurable_terms, fc_covariances


def route_directory(args, mapping, condition):
    return args.output_dir / "{}_{}".format(mapping, condition)


def plot_covariance_distributions(
    distributions,
    mapping,
    condition,
    args,
    peak_normalized,
):
    if not distributions:
        raise ValueError("At least one covariance distribution is required.")

    output_directory = route_directory(args, mapping, condition)
    output_directory.mkdir(parents=True, exist_ok=True)
    colors_by_name = {
        distribution.name: molecule_color(distribution.name)
        for distribution in distributions
    }
    saved_paths = []

    for attribute, filename_component, x_label, fixed_xlim in PLOT_SPECS:
        if peak_normalized:
            output_name = (
                "comparison_{}_{}_{}_offdiag_{}_unit_peak_density".format(
                    mapping,
                    args.wfn,
                    condition,
                    filename_component,
                )
            )
        else:
            if len(distributions) != 1:
                raise ValueError("Individual plots require exactly one molecule.")
            output_name = "{}_{}_{}_offdiag_{}".format(
                distributions[0].name,
                args.wfn,
                condition,
                filename_component,
            )

        paths = plot_distributions(
            distributions,
            attribute,
            x_label,
            output_directory / output_name,
            args.bw_adjust,
            args.dpi,
            fixed_xlim=fixed_xlim,
            peak_normalized=peak_normalized,
            bars=False,
            kde_bandwidth_factor=1.0,
            kde_gridsize=DEFAULT_KDE_GRIDSIZE,
            colors_by_name=colors_by_name,
            no_grid=args.no_grid,
            nonnegative=False,
        )
        saved_paths.extend(paths)
    return saved_paths


def process_mapping(mapping, molecule_names, args):
    fc_distributions = []
    qwc_distributions = []
    saved_paths = []

    for name in molecule_names:
        terms, fc_covariances = analyze_molecule(name, args)

        fc_distribution = build_distribution(
            name,
            "FC",
            fc_covariances,
        )
        fc_distributions.append(fc_distribution)
        saved_paths.extend(
            plot_covariance_distributions(
                [fc_distribution],
                mapping,
                "FC",
                args,
                peak_normalized=False,
            )
        )
        print(
            "{} FC: offdiag_covariances={}".format(
                name,
                fc_distribution.covariances_raw.size,
            ),
            flush=True,
        )

        if name not in QWC_EXCLUDED_SYSTEM_NAMES:
            qwc_covariances = qwc_covariance_dictionary(
                fc_covariances,
                terms,
            )
            qwc_distribution = build_distribution(
                name,
                "QWC",
                qwc_covariances,
            )
            qwc_distributions.append(qwc_distribution)
            saved_paths.extend(
                plot_covariance_distributions(
                    [qwc_distribution],
                    mapping,
                    "QWC",
                    args,
                    peak_normalized=False,
                )
            )
            print(
                "{} QWC: offdiag_covariances={}".format(
                    name,
                    qwc_distribution.covariances_raw.size,
                ),
                flush=True,
            )
            del qwc_covariances

        del fc_covariances

    saved_paths.extend(
        plot_covariance_distributions(
            fc_distributions,
            mapping,
            "FC",
            args,
            peak_normalized=True,
        )
    )
    saved_paths.extend(
        plot_covariance_distributions(
            qwc_distributions,
            mapping,
            "QWC",
            args,
            peak_normalized=True,
        )
    )
    return saved_paths


def main(argv=None):
    args = parse_args(argv)
    for route_name in ROUTE_NAMES:
        (args.output_dir / route_name).mkdir(parents=True, exist_ok=True)

    print("Covariance wavefunction={}".format(args.wfn))
    print("Off-diagonal covariance pairs only")
    print("Histogram bars=disabled")
    print("KDE bw_adjust={:.6g}".format(args.bw_adjust))
    print(
        "Route comparisons=unit-peak relative density "
        "(each visible KDE maximum is one)"
    )
    print("Output root={}".format(args.output_dir))

    saved_paths = []
    saved_paths.extend(process_mapping("JW", JW_SYSTEM_NAMES, args))
    saved_paths.extend(process_mapping("BK", BK_SYSTEM_NAMES, args))

    expected_figures = 26
    actual_figures = len(saved_paths) // 2
    if actual_figures != expected_figures:
        raise RuntimeError(
            "Expected {} figures, but generated {}.".format(
                expected_figures,
                actual_figures,
            )
        )

    print("")
    print(
        "Generated {} figures as {} PNG and {} SVG files under {}.".format(
            actual_figures,
            sum(path.suffix == ".png" for path in saved_paths),
            sum(path.suffix == ".svg" for path in saved_paths),
            args.output_dir,
        )
    )
    return saved_paths


if __name__ == "__main__":
    main()
