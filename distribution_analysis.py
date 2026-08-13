"""Compare Pauli-coefficient and covariance distributions across molecules.

For each requested Jordan-Wigner Hamiltonian, this script constructs the
selected reference wavefunction and the coefficient-free Pauli covariance
dictionary

    C_ij = <P_i P_j> - <P_i><P_j>

for the upper triangle of fully commuting, non-identity Pauli pairs.  It saves
four figures: raw and normalized covariance distributions, and raw and
normalized Hamiltonian-coefficient distributions.  A multi-molecule run also
saves four comparison figures in which every KDE is rescaled to unit peak
height.  Normalized covariances are correlation coefficients
C_ij / sqrt(C_ii C_jj); normalized coefficients are scaled by max_i |c_i|.
The constant identity term is excluded so both kinds of plots describe the
same measurable Pauli terms.  All plotted values retain their sign.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
from dataclasses import dataclass
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
from openfermion.linalg import get_sparse_operator
from openfermion.utils import count_qubits

import gflow_vqe.hamiltonians as hamlib
from gflow_vqe.utils import get_variance_wavefunction
from greedy import (
    JW_SYSTEM_NAMES,
    build_covariance_dictionary,
    default_cov_workers,
    make_terms,
)


DEFAULT_OUTPUT_DIRECTORY = "distribution_analysis_plots"
DEFAULT_BANDWIDTH_ADJUSTMENT = 0.8
DEFAULT_COVARIANCE_CHUNKSIZE = 128
DEFAULT_MAX_MEMORY_GIB = 8.0
REAL_TOLERANCE = 1.0e-9
ZERO_VARIANCE_TOLERANCE = 1.0e-12


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build Pauli covariance dictionaries and compare raw and "
            "normalized coefficient/covariance distributions."
        )
    )
    parser.add_argument(
        "molecules",
        nargs="+",
        choices=JW_SYSTEM_NAMES,
        help=(
            "One or more Jordan-Wigner molecule helpers. Multiple molecules "
            "are overlaid in each output figure."
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
            "Directory for generated figures "
            "(default: distribution_analysis_plots)."
        ),
    )
    parser.add_argument(
        "--format",
        dest="figure_format",
        choices=("png", "svg", "pdf"),
        default="png",
        help="Output figure format (default: png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution for raster output (default: 300).",
    )
    parser.add_argument(
        "--bw-adjust",
        type=float,
        default=DEFAULT_BANDWIDTH_ADJUSTMENT,
        help="Seaborn KDE bandwidth multiplier (default: 0.8).",
    )
    parser.add_argument(
        "--off-diagonal-only",
        action="store_true",
        help="Exclude diagonal variances C_ii from both covariance plots.",
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
    print("Building {} Jordan-Wigner Hamiltonian...".format(name), flush=True)
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
    covariances_raw, covariances_normalized, skipped = covariance_samples(
        covariance_dictionary,
        off_diagonal_only=args.off_diagonal_only,
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
    )
    print(
        "{}: covariance_entries={} runtime_s={:.6f}; plotted_raw={} "
        "plotted_normalized={} skipped_zero_variance={}".format(
            name,
            result.covariance_entries,
            result.covariance_runtime_s,
            result.covariances_raw.size,
            result.covariances_normalized.size,
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


def normalize_density_height(x_values, density, visible_xlim=None):
    """Scale a sampled KDE so its visible maximum is one."""

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
    return density / peak


def plot_distributions(
    distributions,
    attribute,
    x_label,
    output_path,
    bw_adjust,
    dpi,
    fixed_xlim=None,
    peak_normalized=False,
):
    """Overlay one independently normalized KDE per molecule."""

    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(10.5, 6.5))
    palette = sns.color_palette("colorblind", n_colors=len(distributions))
    fill = len(distributions) == 1 and not peak_normalized

    any_samples = False
    for color, distribution in zip(palette, distributions):
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
        elif has_kde_support(values):
            any_samples = True
            line_count = len(axis.lines)
            sns.kdeplot(
                x=values,
                ax=axis,
                label=label,
                color=color,
                fill=fill,
                alpha=0.25 if fill else 1.0,
                bw_adjust=bw_adjust,
                common_norm=False,
                warn_singular=False,
                linewidth=2.0,
            )
            if peak_normalized:
                if len(axis.lines) != line_count + 1:
                    raise RuntimeError("Seaborn did not produce one KDE curve.")
                curve = axis.lines[-1]
                curve.set_ydata(
                    normalize_density_height(
                        curve.get_xdata(),
                        curve.get_ydata(),
                        visible_xlim=fixed_xlim,
                    )
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
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("Relative density")
    else:
        axis.set_ylabel("Probability density")
    axis.legend(title="Molecule", frameon=True)
    figure.tight_layout()
    figure.savefig(
        output_path,
        format=output_path.suffix.lstrip("."),
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)
    print("Saved {}".format(output_path), flush=True)


def make_all_plots(distributions, args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = comparison_prefix([item.name for item in distributions], args.wfn)
    if args.off_diagonal_only:
        prefix += "_offdiag"
    suffix = ".{}".format(args.figure_format)
    plot_specs = (
        (
            "covariances_raw",
            "covariances_unnormalized",
            r"Covariance $C_{ij}$",
            None,
        ),
        (
            "covariances_normalized",
            "covariances_normalized",
            r"Normalized covariance $C_{ij}/\sqrt{C_{ii}C_{jj}}$",
            (-1.05, 1.05),
        ),
        (
            "coefficients_raw",
            "coefficients_unnormalized",
            r"Coefficient $c_i$",
            None,
        ),
        (
            "coefficients_normalized",
            "coefficients_normalized",
            r"Normalized coefficient $c_i/\max_j|c_j|$",
            (-1.05, 1.05),
        ),
    )

    paths = {}
    for key, filename_component, x_label, fixed_xlim in plot_specs:
        paths[key] = args.output_dir / "{}_{}{}".format(
            prefix,
            filename_component,
            suffix,
        )
        plot_distributions(
            distributions,
            key,
            x_label,
            paths[key],
            args.bw_adjust,
            args.dpi,
            fixed_xlim=fixed_xlim,
        )

    if len(distributions) > 1:
        for key, filename_component, x_label, fixed_xlim in plot_specs:
            peak_key = "{}_unit_peak_density".format(key)
            paths[peak_key] = args.output_dir / "{}_{}_unit_peak_density{}".format(
                prefix,
                filename_component,
                suffix,
            )
            plot_distributions(
                distributions,
                key,
                x_label,
                paths[peak_key],
                args.bw_adjust,
                args.dpi,
                fixed_xlim=fixed_xlim,
                peak_normalized=True,
            )
    return paths


def main(argv=None):
    args = parse_args(argv)
    print("Molecules={}".format(", ".join(args.molecules)))
    print("Covariance wavefunction={}".format(args.wfn))
    print(
        "Normalization: covariance=correlation coefficient; "
        "coefficient=max-absolute scaling."
    )

    distributions = []
    for molecule_name in args.molecules:
        distributions.append(analyze_molecule(molecule_name, args))

    paths = make_all_plots(distributions, args)
    print("")
    print("Generated {} figures in {}:".format(len(paths), args.output_dir))
    for path in paths.values():
        print("  {}".format(path))
    return distributions, paths


if __name__ == "__main__":
    main()
