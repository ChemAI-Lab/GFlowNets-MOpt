import argparse
import os
import math
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from openfermion.linalg import get_sparse_operator
from openfermion.utils import count_qubits
from tequila.grouping.binary_rep import BinaryHamiltonian, BinaryPauliString
from tequila.grouping.binary_utils import sorted_insertion_grouping, term_commutes_with_group
from tequila.hamiltonian import QubitHamiltonian

import gflow_vqe.hamiltonians as hamlib
from gflow_vqe.overlapping_helpers import (
    as_tequila_wavefunction,
    get_cov,
    iterative_coefficient_splitting_from_groups,
    prepare_cov_dict,
)
from gflow_vqe.utils import get_variance_wavefunction


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Variance-aware sorted insertion. Builds groups by Pauli-term covariance "
            "and refines standard sorted insertion groups by lowering eps^2 M."
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
        help="Wavefunction used to build the covariance dictionary (default: FCI).",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="fc",
        choices=("fc", "qwc"),
        help="Compatibility condition for groups: fully commuting or qubit-wise commuting (default: fc).",
    )
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=100,
        help="Maximum greedy sweeps for the SI-refinement version (default: 100).",
    )
    parser.add_argument(
        "--allow-new-groups",
        action="store_true",
        help="During SI refinement, also consider moving a term into a new singleton group.",
    )
    parser.add_argument(
        "--ordered-consider-new-groups",
        action="store_true",
        help="Let the ordered VarSI pass choose a new singleton group even when compatible groups exist.",
    )
    parser.add_argument(
        "--cov-workers",
        type=int,
        default=None,
        help="Number of worker processes for the VarSI-local parallel covariance builder (default: up to 8).",
    )
    parser.add_argument(
        "--cov-chunksize",
        type=int,
        default=128,
        help="Number of term pairs sent to each covariance worker task (default: 128).",
    )
    parser.add_argument(
        "--serial-cov-dict",
        action="store_true",
        help="Use the original serial prepare_cov_dict helper instead of the VarSI-local parallel builder.",
    )
    parser.add_argument(
        "--print-groups",
        action="store_true",
        help="Print the final Pauli groups for each VarSI variant.",
    )
    args = parser.parse_args(argv)
    args.func = getattr(hamlib, args.func_name, None)
    if args.func is None:
        raise ValueError("Unknown molecule '{}'".format(args.func_name))
    if args.max_sweeps < 1:
        parser.error("--max-sweeps must be at least 1.")
    if args.cov_workers is not None and args.cov_workers < 1:
        parser.error("--cov-workers must be at least 1.")
    if args.cov_chunksize < 1:
        parser.error("--cov-chunksize must be at least 1.")
    return args


def _is_identity_term(term):
    return not np.any(term.get_binary())


def measurable_terms(binary_hamiltonian):
    return [term for term in binary_hamiltonian.binary_terms if not _is_identity_term(term)]


def _clean_variance(value, tiny=1e-10):
    value = np.real_if_close(value, tol=1000)
    if hasattr(value, "imag") and abs(value.imag) > tiny:
        raise ValueError("Expected a real variance, got {}.".format(value))

    value = float(np.real(value))
    if value < 0.0 and abs(value) < tiny:
        return 0.0
    if value < 0.0:
        raise ValueError("Computed a negative variance: {}.".format(value))
    return value


def covariance_with_scaled_term(group, term, cov_dict):
    """
    Cov(H_group, c_i P_i) = sum_alpha c_alpha c_i Cov(P_alpha, P_i).
    """
    term_coeff = term.get_coeff()
    cov = 0.0
    for group_term in group:
        cov += group_term.get_coeff() * term_coeff * get_cov(group_term, term, cov_dict)
    return cov


def single_term_variance(term, cov_dict):
    coeff = term.get_coeff()
    return _clean_variance(coeff * coeff * get_cov(term, term, cov_dict))


def group_variance(group, cov_dict):
    variance = 0.0
    for term1 in group:
        for term2 in group:
            variance += term1.get_coeff() * term2.get_coeff() * get_cov(term1, term2, cov_dict)
    return _clean_variance(variance)


def variance_after_addition(group, term, cov_dict, current_variance=None):
    if current_variance is None:
        current_variance = group_variance(group, cov_dict)

    # Var(H_alpha + c_i P_i) = Var(H_alpha) + c_i^2 Var(P_i)
    #                           + 2 Cov(H_alpha, c_i P_i).
    return _clean_variance(
        current_variance
        + single_term_variance(term, cov_dict)
        + 2.0 * covariance_with_scaled_term(group, term, cov_dict)
    )


def variance_after_removal(group, term_index, cov_dict, current_variance):
    term = group[term_index]
    if len(group) == 1:
        return None

    group_without_term = group[:term_index] + group[term_index + 1 :]
    return _clean_variance(
        current_variance
        - single_term_variance(term, cov_dict)
        - 2.0 * covariance_with_scaled_term(group_without_term, term, cov_dict)
    )


def eps_sq_m_from_variances(variances):
    sqrt_sum = sum(math.sqrt(max(variance, 0.0)) for variance in variances)
    return sqrt_sum * sqrt_sum


def sample_ratios_from_variances(variances):
    weights = np.sqrt(np.asarray(variances, dtype=float))
    total = float(np.sum(weights))
    if total == 0.0:
        return np.ones(len(variances), dtype=float) / len(variances)
    return weights / total


def normalize_groups(groups):
    normalized = []
    for group in groups:
        if isinstance(group, BinaryHamiltonian):
            terms = list(group.binary_terms)
        else:
            terms = list(group)
        terms = [term for term in terms if not _is_identity_term(term)]
        if terms:
            normalized.append(terms)
    if not normalized:
        raise ValueError("No measurable Pauli terms were found.")
    return normalized


def _terms_compatible(term1, term2, condition):
    if condition == "fc":
        return term1.commute(term2)
    if condition == "qwc":
        return term1.qubit_wise_commute(term2)
    raise ValueError("Unsupported compatibility condition '{}'.".format(condition))


def validate_compatible_groups(groups, condition="fc"):
    normalized_groups = normalize_groups(groups)
    for group_idx, group in enumerate(normalized_groups):
        for term_idx, term in enumerate(group):
            for other_idx in range(term_idx + 1, len(group)):
                other = group[other_idx]
                if not _terms_compatible(term, other, condition):
                    raise ValueError(
                        "Incompatible terms found in group {} at positions {} and {} under condition '{}'.".format(
                            group_idx,
                            term_idx,
                            other_idx,
                            condition,
                        )
                    )
    return normalized_groups


def timed_call(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start


def timed_optional_call(func, *args, **kwargs):
    start = time.perf_counter()
    try:
        result = func(*args, **kwargs)
    except Exception as exc:
        return None, time.perf_counter() - start, exc
    return result, time.perf_counter() - start, None


_PARALLEL_COV_WFN = None


def _default_cov_workers():
    return max(1, min(8, os.cpu_count() or 1))


def _init_parallel_cov_worker(approx_wfn):
    global _PARALLEL_COV_WFN
    _PARALLEL_COV_WFN = as_tequila_wavefunction(approx_wfn)


def _iter_covariance_pair_chunks(term_data, chunksize):
    chunk = []
    for idx, (binary_1, key_1) in enumerate(term_data):
        for binary_2, key_2 in term_data[idx:]:
            chunk.append((binary_1, binary_2, key_1, key_2))
            if len(chunk) == chunksize:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def _parallel_covariance_chunk(pair_chunk):
    if _PARALLEL_COV_WFN is None:
        raise RuntimeError("Parallel covariance worker was not initialized with a reference wavefunction.")

    chunk_covariances = []
    for binary_1, binary_2, key_1, key_2 in pair_chunk:
        pauli_1 = BinaryPauliString(binary_1, 1.0)
        pauli_2 = BinaryPauliString(binary_2, 1.0)
        if not pauli_1.commute(pauli_2):
            continue

        op1 = QubitHamiltonian.from_paulistrings(pauli_1.to_pauli_strings())
        op2 = QubitHamiltonian.from_paulistrings(pauli_2.to_pauli_strings())
        covariance = _PARALLEL_COV_WFN.inner((op1 * op2)(_PARALLEL_COV_WFN)) - _PARALLEL_COV_WFN.inner(
            op1(_PARALLEL_COV_WFN)
        ) * _PARALLEL_COV_WFN.inner(op2(_PARALLEL_COV_WFN))
        chunk_covariances.append(((key_1, key_2), covariance))
    return chunk_covariances


def prepare_cov_dict_parallel(binary_hamiltonian, approx_wfn, max_workers=None, chunksize=128):
    """
    VarSI-local parallel version of prepare_cov_dict.

    The returned dictionary has the same keys and values as
    gflow_vqe.overlapping_helpers.prepare_cov_dict: ordered pairs of term
    binary tuples mapped to covariance values for commuting pairs only.
    """
    term_data = [(term.binary_tuple(), term.binary_tuple()) for term in binary_hamiltonian.binary_terms]
    if max_workers is None:
        max_workers = _default_cov_workers()

    cov_dict = {}
    pair_chunks = _iter_covariance_pair_chunks(term_data, chunksize)
    if max_workers == 1:
        _init_parallel_cov_worker(approx_wfn)
        for chunk_result in map(_parallel_covariance_chunk, pair_chunks):
            cov_dict.update(chunk_result)
        return cov_dict

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_parallel_cov_worker,
        initargs=(approx_wfn,),
    ) as executor:
        for chunk_result in executor.map(_parallel_covariance_chunk, pair_chunks):
            cov_dict.update(chunk_result)
    return cov_dict


def build_covariance_dictionary(binary_hamiltonian, wfn, use_serial=False, max_workers=None, chunksize=128):
    if use_serial:
        cov_dict, cov_time = timed_call(prepare_cov_dict, binary_hamiltonian, wfn)
        cov_label = "serial prepare_cov_dict"
    else:
        cov_workers = max_workers or _default_cov_workers()
        cov_dict, cov_time = timed_call(
            prepare_cov_dict_parallel,
            binary_hamiltonian,
            wfn,
            max_workers=cov_workers,
            chunksize=chunksize,
        )
        cov_label = "parallel prepare_cov_dict (workers={}, chunksize={})".format(
            cov_workers,
            chunksize,
        )
    return cov_dict, cov_time, cov_label


def variance_sorted_insertion_grouping(terms, cov_dict, condition="fc"):
    """
    Start from empty groups.

    The first group is seeded with the largest single-term variance. After that,
    every possible compatible insertion is scored by the total eps^2 M it would
    produce, and the lowest-scoring insertion is accepted. If a remaining term
    has no compatible existing group, opening a new group is considered for that
    term.
    """
    remaining_terms = sorted(terms, key=lambda term: single_term_variance(term, cov_dict), reverse=True)
    groups = []
    variances = []

    while remaining_terms:
        if not groups:
            term = remaining_terms.pop(0)
            groups.append([term])
            variances.append(single_term_variance(term, cov_dict))
            continue

        current_sqrt_sum = sum(math.sqrt(variance) for variance in variances)
        best_candidate = None

        for term_pos, term in enumerate(remaining_terms):
            compatible_candidates = []
            for group_idx, group in enumerate(groups):
                if not term_commutes_with_group(term, group, condition):
                    continue
                new_variance = variance_after_addition(group, term, cov_dict, variances[group_idx])
                sqrt_sum = current_sqrt_sum - math.sqrt(variances[group_idx]) + math.sqrt(new_variance)
                compatible_candidates.append((sqrt_sum * sqrt_sum, new_variance, group_idx))

            if not compatible_candidates:
                new_variance = single_term_variance(term, cov_dict)
                sqrt_sum = current_sqrt_sum + math.sqrt(new_variance)
                candidate = (sqrt_sum * sqrt_sum, new_variance, len(groups), term_pos)
            else:
                metric, new_variance, group_idx = min(
                    compatible_candidates,
                    key=lambda item: (item[0], item[1], item[2]),
                )
                candidate = (metric, new_variance, group_idx, term_pos)

            if best_candidate is None or candidate < best_candidate:
                best_candidate = candidate

        _, new_variance, group_idx, term_pos = best_candidate
        term = remaining_terms.pop(term_pos)
        if group_idx == len(groups):
            groups.append([term])
            variances.append(new_variance)
        else:
            groups[group_idx].append(term)
            variances[group_idx] = new_variance

    return groups


def variance_sorted_insertion_grouping_ordered(
    terms,
    cov_dict,
    condition="fc",
    consider_new_groups=False,
):
    """
    Variance-aware Sorted Insertion.

    Terms are processed in descending single-term variance.
    For each term, choose the compatible group that minimizes the
    resulting eps^2 M, or equivalently the increase in sum sqrt(V_g).
    """
    sorted_terms = sorted(
        terms,
        key=lambda term: single_term_variance(term, cov_dict),
        reverse=True,
    )

    groups = []
    variances = []

    for term in sorted_terms:
        term_variance = single_term_variance(term, cov_dict)
        current_sqrt_sum = sum(math.sqrt(v) for v in variances)

        best_metric = None
        best_group_idx = None
        best_new_variance = None

        for group_idx, group in enumerate(groups):
            if not term_commutes_with_group(term, group, condition):
                continue

            new_variance = variance_after_addition(
                group,
                term,
                cov_dict,
                current_variance=variances[group_idx],
            )

            new_sqrt_sum = (
                current_sqrt_sum
                - math.sqrt(variances[group_idx])
                + math.sqrt(new_variance)
            )
            metric = new_sqrt_sum * new_sqrt_sum

            if best_metric is None or metric < best_metric:
                best_metric = metric
                best_group_idx = group_idx
                best_new_variance = new_variance

        # The singleton option is theoretically unnecessary for exact
        # covariances and pure shot-count minimization, but useful for
        # noisy covariances or extended objectives.
        if best_metric is None or consider_new_groups:
            singleton_metric = (
                current_sqrt_sum + math.sqrt(term_variance)
            ) ** 2

            if best_metric is None or singleton_metric < best_metric:
                best_metric = singleton_metric
                best_group_idx = None
                best_new_variance = term_variance

        if best_group_idx is None:
            groups.append([term])
            variances.append(best_new_variance)
        else:
            groups[best_group_idx].append(term)
            variances[best_group_idx] = best_new_variance

    return groups


def _move_candidate_metric(
    groups,
    variances,
    source_idx,
    term_idx,
    dest_idx,
    cov_dict,
    current_sqrt_sum,
):
    term = groups[source_idx][term_idx]
    source_variance = variances[source_idx]
    source_new_variance = variance_after_removal(groups[source_idx], term_idx, cov_dict, source_variance)

    if dest_idx == len(groups):
        dest_old_variance = None
        dest_new_variance = single_term_variance(term, cov_dict)
    else:
        dest_old_variance = variances[dest_idx]
        dest_new_variance = variance_after_addition(groups[dest_idx], term, cov_dict, dest_old_variance)

    sqrt_sum = current_sqrt_sum - math.sqrt(source_variance)
    if source_new_variance is not None:
        sqrt_sum += math.sqrt(source_new_variance)

    if dest_old_variance is not None:
        sqrt_sum -= math.sqrt(dest_old_variance)
    sqrt_sum += math.sqrt(dest_new_variance)

    return sqrt_sum * sqrt_sum, source_new_variance, dest_new_variance


def _apply_move(groups, variances, move):
    source_idx, term_idx, dest_idx, source_new_variance, dest_new_variance = move
    term = groups[source_idx].pop(term_idx)

    if source_new_variance is None:
        del groups[source_idx]
        del variances[source_idx]
        if dest_idx > source_idx:
            dest_idx -= 1
    else:
        variances[source_idx] = source_new_variance

    if dest_idx == len(groups):
        groups.append([term])
        variances.append(dest_new_variance)
    else:
        groups[dest_idx].append(term)
        variances[dest_idx] = dest_new_variance


def refine_sorted_insertion_groups(
    initial_groups,
    cov_dict,
    condition="fc",
    max_sweeps=100,
    allow_new_groups=False,
    tiny=1e-10,
):
    """
    Start from Tequila sorted insertion groups and greedily move one term at a
    time whenever the move lowers total eps^2 M.
    """
    groups = [list(group) for group in normalize_groups(initial_groups)]
    variances = [group_variance(group, cov_dict) for group in groups]
    accepted_moves = 0

    for _ in range(max_sweeps):
        current_sqrt_sum = sum(math.sqrt(variance) for variance in variances)
        current_metric = current_sqrt_sum * current_sqrt_sum
        best_metric = current_metric
        best_move = None

        for source_idx, group in enumerate(groups):
            for term_idx, term in enumerate(group):
                dest_indices = [idx for idx in range(len(groups)) if idx != source_idx]
                if allow_new_groups:
                    dest_indices.append(len(groups))

                for dest_idx in dest_indices:
                    if dest_idx < len(groups) and not term_commutes_with_group(term, groups[dest_idx], condition):
                        continue

                    candidate_metric, source_new_variance, dest_new_variance = _move_candidate_metric(
                        groups,
                        variances,
                        source_idx,
                        term_idx,
                        dest_idx,
                        cov_dict,
                        current_sqrt_sum,
                    )
                    if candidate_metric < best_metric - tiny:
                        best_metric = candidate_metric
                        best_move = (
                            source_idx,
                            term_idx,
                            dest_idx,
                            source_new_variance,
                            dest_new_variance,
                        )

        if best_move is None:
            break

        _apply_move(groups, variances, best_move)
        accepted_moves += 1

    return groups, accepted_moves


def print_group_contents(label, groups):
    print("{} groups:".format(label))
    for idx, group in enumerate(normalize_groups(groups)):
        terms = []
        for term in group:
            terms.append(str(term.to_pauli_strings()))
        print("  Group {}: {}".format(idx, ", ".join(terms)))


def make_result(
    label,
    groups,
    wfn_cov_dict,
    fci_cov_dict,
    wfn_label,
    condition="fc",
    sample_ratios=None,
    runtime=None,
    extra=None,
):
    groups = validate_compatible_groups(groups, condition=condition)
    wfn_variances = [group_variance(group, wfn_cov_dict) for group in groups]
    fci_variances = [group_variance(group, fci_cov_dict) for group in groups]
    if sample_ratios is None:
        sample_ratios = sample_ratios_from_variances(wfn_variances)
    return {
        "label": label,
        "groups": groups,
        "eps_sq_m_wfn": eps_sq_m_from_variances(wfn_variances),
        "eps_sq_m_fci": eps_sq_m_from_variances(fci_variances),
        "wfn_label": wfn_label,
        "num_groups": len(groups),
        "sample_ratios": sample_ratios,
        "runtime": runtime,
        "extra": extra,
    }


def print_result(result):
    label = result["label"]
    print("{}:".format(label))
    print("  eps^2 M(wfn={})={:.12g}".format(result["wfn_label"], result["eps_sq_m_wfn"]))
    print("  eps^2 M(FCI)={:.12g}".format(result["eps_sq_m_fci"]))
    print("  Number of groups={}".format(result["num_groups"]))
    print("  Compatible groups=True")
    if result["runtime"] is not None:
        print("  Runtime (s)={:.6f}".format(result["runtime"]))
    #print("  Group variances={}".format(["{:.12g}".format(variance) for variance in variances]))
    #print("  Suggested sample ratios={}".format(["{:.8g}".format(ratio) for ratio in sample_ratios]))
    if result["extra"] is not None:
        print("  {}".format(result["extra"]))


def main(argv=None):
    args = parse_args(argv)
    mol, H, _, n_paulis, Hq = args.func()
    print("Molecule={}".format(args.func_name))
    print("Number of Pauli products to measure: {}".format(n_paulis))

    sparse_hamiltonian = get_sparse_operator(Hq)
    energy, variance_wfn = get_variance_wavefunction(
        mol,
        Hq,
        method=args.wfn,
        sparse_hamiltonian=sparse_hamiltonian,
    )
    print("{} Energy={}".format(args.wfn, energy))
    if args.wfn == "FCI":
        fci_energy = energy
        fci_wfn = variance_wfn
    else:
        fci_energy, fci_wfn = get_variance_wavefunction(
            mol,
            Hq,
            method="FCI",
            sparse_hamiltonian=sparse_hamiltonian,
        )
        print("FCI Energy={}".format(fci_energy))
    print("Compatibility condition={}".format(args.condition))
    print("Number of qubits={}".format(count_qubits(Hq)))

    binary_hamiltonian = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
    terms = measurable_terms(binary_hamiltonian)
    cov_workers = args.cov_workers or _default_cov_workers()
    cov_dict, cov_time, cov_label = build_covariance_dictionary(
        binary_hamiltonian,
        variance_wfn,
        use_serial=args.serial_cov_dict,
        max_workers=cov_workers,
        chunksize=args.cov_chunksize,
    )
    print(
        "{} covariance dictionary: {} entries built with {} in {:.6f} s".format(
            args.wfn,
            len(cov_dict),
            cov_label,
            cov_time,
        )
    )
    if args.wfn == "FCI":
        fci_cov_dict = cov_dict
        print("FCI reporting covariance dictionary: reused {} covariance dictionary".format(args.wfn))
    else:
        fci_cov_dict, fci_cov_time, fci_cov_label = build_covariance_dictionary(
            binary_hamiltonian,
            fci_wfn,
            use_serial=args.serial_cov_dict,
            max_workers=cov_workers,
            chunksize=args.cov_chunksize,
        )
        print(
            "FCI reporting covariance dictionary: {} entries built with {} in {:.6f} s".format(
                len(fci_cov_dict),
                fci_cov_label,
                fci_cov_time,
            )
        )

    failed_methods = []

    def run_ics(label, initial_groups):
        output, runtime, error = timed_optional_call(
            iterative_coefficient_splitting_from_groups,
            initial_groups,
            cov_dict,
            condition=args.condition,
        )
        if error is not None:
            failed_methods.append(
                {
                    "label": label,
                    "runtime": runtime,
                    "error": "{}: {}".format(type(error).__name__, error),
                }
            )
            return None, None, runtime
        groups, sample_size = output
        return groups, sample_size, runtime

    si_groups, si_time = timed_call(
        lambda: normalize_groups(sorted_insertion_grouping(terms, condition=args.condition))
    )
    si_groups = validate_compatible_groups(si_groups, condition=args.condition)

    si_ics_groups, si_ics_sample_size, si_ics_time = run_ics(
        "ICS initialized from sorted insertion groups",
        si_groups,
    )

    varsi_groups, varsi_time = timed_call(
        variance_sorted_insertion_grouping,
        terms,
        cov_dict,
        condition=args.condition,
    )
    varsi_groups = validate_compatible_groups(varsi_groups, condition=args.condition)

    varsi_ics_groups, varsi_ics_sample_size, varsi_ics_time = run_ics(
        "ICS initialized from VarSI greedy groups",
        varsi_groups,
    )

    ordered_varsi_groups, ordered_varsi_time = timed_call(
        variance_sorted_insertion_grouping_ordered,
        terms,
        cov_dict,
        condition=args.condition,
        consider_new_groups=args.ordered_consider_new_groups,
    )
    ordered_varsi_groups = validate_compatible_groups(ordered_varsi_groups, condition=args.condition)

    (ordered_refined_groups, ordered_refined_moves), ordered_refined_time = timed_call(
        refine_sorted_insertion_groups,
        ordered_varsi_groups,
        cov_dict,
        condition=args.condition,
        max_sweeps=args.max_sweeps,
        allow_new_groups=args.allow_new_groups,
    )
    ordered_refined_groups = validate_compatible_groups(ordered_refined_groups, condition=args.condition)

    ordered_varsi_ics_groups, ordered_varsi_ics_sample_size, ordered_varsi_ics_time = run_ics(
        "ICS initialized from VarSI ordered groups",
        ordered_varsi_groups,
    )

    ordered_refined_ics_groups, ordered_refined_ics_sample_size, ordered_refined_ics_time = run_ics(
        "ICS initialized from VarSI ordered+refined groups",
        ordered_refined_groups,
    )

    (refined_groups, accepted_moves), refined_time = timed_call(
        refine_sorted_insertion_groups,
        si_groups,
        cov_dict,
        condition=args.condition,
        max_sweeps=args.max_sweeps,
        allow_new_groups=args.allow_new_groups,
    )
    refined_groups = validate_compatible_groups(refined_groups, condition=args.condition)

    refined_ics_groups, refined_ics_sample_size, refined_ics_time = run_ics(
        "ICS initialized from VarSI-refined sorted insertion groups",
        refined_groups,
    )

    results = []

    def add_result(label, groups, runtime, sample_ratios=None, extra=None):
        results.append(
            make_result(
                label,
                groups,
                cov_dict,
                fci_cov_dict,
                args.wfn,
                condition=args.condition,
                sample_ratios=sample_ratios,
                runtime=runtime,
                extra=extra,
            )
        )

    add_result("Sorted insertion baseline", si_groups, si_time)
    if si_ics_groups is not None:
        add_result(
            "ICS initialized from sorted insertion groups",
            si_ics_groups,
            si_ics_time,
            sample_ratios=si_ics_sample_size,
        )
    add_result("VarSI greedy from empty groups", varsi_groups, varsi_time)
    if varsi_ics_groups is not None:
        add_result(
            "ICS initialized from VarSI greedy groups",
            varsi_ics_groups,
            varsi_ics_time,
            sample_ratios=varsi_ics_sample_size,
        )
    add_result("VarSI ordered from empty groups", ordered_varsi_groups, ordered_varsi_time)
    add_result(
        "VarSI ordered+refined",
        ordered_refined_groups,
        ordered_refined_time,
        extra="Accepted moves={}".format(ordered_refined_moves),
    )
    if ordered_varsi_ics_groups is not None:
        add_result(
            "ICS initialized from VarSI ordered groups",
            ordered_varsi_ics_groups,
            ordered_varsi_ics_time,
            sample_ratios=ordered_varsi_ics_sample_size,
        )
    if ordered_refined_ics_groups is not None:
        add_result(
            "ICS initialized from VarSI ordered+refined groups",
            ordered_refined_ics_groups,
            ordered_refined_ics_time,
            sample_ratios=ordered_refined_ics_sample_size,
        )
    add_result(
        "VarSI refinement from sorted insertion",
        refined_groups,
        refined_time,
        extra="Accepted moves={}".format(accepted_moves),
    )
    if refined_ics_groups is not None:
        add_result(
            "ICS initialized from VarSI-refined sorted insertion groups",
            refined_ics_groups,
            refined_ics_time,
            sample_ratios=refined_ics_sample_size,
        )

    for result in results:
        print("")
        print_result(result)

    if failed_methods:
        print("")
        print("Skipped methods:")
        for failed in failed_methods:
            print(
                "  {} failed after {:.6f} s: {}".format(
                    failed["label"],
                    failed["runtime"],
                    failed["error"],
                )
            )

    ranked_results = sorted(results, key=lambda result: result["eps_sq_m_fci"])
    print("")
    print("Ranking by eps^2 M(FCI) (lowest to highest):")
    for rank, result in enumerate(ranked_results, start=1):
        print(
            "  {}. {}: {:.12g} Groups: {}".format(
                rank,
                result["label"],
                result["eps_sq_m_fci"],
                result["num_groups"],
            )
        )

    if args.print_groups:
        for result in results:
            print("")
            print_group_contents(result["label"], result["groups"])


if __name__ == "__main__":
    main()
