from __future__ import annotations

"""
The core overlapping-grouping classes and covariance/sample-allocation utilities
in this module are adapted from Tequila's
`tequila/grouping/overlapping_methods.py` (tequila-basic 1.9.11).
Local additions in this file add support for user-specified initial groups and
for GFlowNet-style colorings produced by this repository.
"""

from copy import deepcopy
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np
import tequila as tq

from tequila.grouping.binary_rep import BinaryHamiltonian, BinaryPauliString
from tequila.grouping.binary_utils import sorted_insertion_grouping, term_commutes_with_group
from tequila.hamiltonian import QubitHamiltonian


def prepare_cov_dict(binary_hamiltonian: BinaryHamiltonian, approx_wfn):
    """
    Build the covariance dictionary expected by Tequila overlapping methods (ICS).

    Keys are ordered pairs of term binary tuples for commuting term pairs.
    """
    cov_dict = {}
    reference_wfn = tq.QubitWaveFunction(approx_wfn)

    for idx, term1 in enumerate(binary_hamiltonian.binary_terms):
        for term2 in binary_hamiltonian.binary_terms[idx:]:
            pauli_1 = BinaryPauliString(term1.get_binary(), 1.0)
            pauli_2 = BinaryPauliString(term2.get_binary(), 1.0)
            if not pauli_1.commute(pauli_2):
                continue

            op1 = QubitHamiltonian.from_paulistrings(pauli_1.to_pauli_strings())
            op2 = QubitHamiltonian.from_paulistrings(pauli_2.to_pauli_strings())
            covariance = reference_wfn.inner((op1 * op2)(reference_wfn)) - reference_wfn.inner(
                op1(reference_wfn)
            ) * reference_wfn.inner(op2(reference_wfn))
            cov_dict[(term1.binary_tuple(), term2.binary_tuple())] = covariance

    return cov_dict


def _is_identity_term(term) -> bool:
    return not np.any(term.get_binary())


def _group_to_terms(group):
    if isinstance(group, BinaryHamiltonian):
        return list(group.binary_terms)
    return list(group)


def _normalize_groups(groups, drop_identity=True):
    normalized = []
    for group in groups:
        terms = _group_to_terms(group)
        if drop_identity:
            terms = [term for term in terms if not _is_identity_term(term)]
        if terms:
            normalized.append(terms)
    return normalized


def _unique_terms(groups):
    unique = {}
    for group in groups:
        for term in group:
            key = term.binary_tuple()
            if key in unique:
                prev_coeff = unique[key].get_coeff()
                cur_coeff = term.get_coeff()
                if not np.isclose(prev_coeff, cur_coeff):
                    raise ValueError(
                        "Inconsistent coefficients found for term {}: {} vs {}.".format(
                            key,
                            prev_coeff,
                            cur_coeff,
                        )
                    )
                continue
            unique[key] = term
    return list(unique.values())


def _validate_nonoverlapping_groups(groups, reference_terms):
    if not groups:
        raise ValueError("At least one non-overlapping group is required.")

    group_keys = []
    for group in groups:
        group_keys.extend(term.binary_tuple() for term in group)

    if len(group_keys) != len(set(group_keys)):
        raise ValueError("Initial groups must be non-overlapping; at least one term appears more than once.")

    ref_keys = [term.binary_tuple() for term in reference_terms]
    if set(group_keys) != set(ref_keys):
        missing = sorted(set(ref_keys) - set(group_keys))
        extra = sorted(set(group_keys) - set(ref_keys))
        raise ValueError(
            "Initial groups do not match the reference terms. Missing={}, Extra={}.".format(
                missing,
                extra,
            )
        )


def extract_measurable_terms(binary_hamiltonian_or_terms):
    """
    Return all non-identity BinaryPauliStrings in their original order.
    """
    if isinstance(binary_hamiltonian_or_terms, BinaryHamiltonian):
        terms = binary_hamiltonian_or_terms.binary_terms
    else:
        terms = list(binary_hamiltonian_or_terms)
    return [term for term in terms if not _is_identity_term(term)]


def groups_from_gflow_grouping(grouping: Any, terms: Sequence[Any]):
    """
    Convert a GFlowNet-style coloring into non-overlapping BinaryPauliString groups.

    Supported inputs:
    - `networkx.Graph` with node attribute `color`
    - mapping `node -> color`
    - mapping `color -> [node, ...]`
    - sequence of colors ordered like `terms`
    """
    color_to_terms = {}

    def add_node(node, color):
        idx = int(node)
        if idx < 0 or idx >= len(terms):
            raise ValueError("Node index {} is out of range for {} terms.".format(idx, len(terms)))
        color_to_terms.setdefault(color, []).append(terms[idx])

    if isinstance(grouping, nx.Graph):
        for node, data in grouping.nodes(data=True):
            if "color" not in data:
                raise ValueError("Graph node {} is missing the 'color' attribute.".format(node))
            add_node(node, data["color"])
    elif isinstance(grouping, Mapping):
        if grouping and all(isinstance(value, (list, tuple, set)) for value in grouping.values()):
            for color, nodes in grouping.items():
                for node in nodes:
                    add_node(node, color)
        else:
            for node, color in grouping.items():
                add_node(node, color)
    else:
        colors = list(grouping)
        if len(colors) != len(terms):
            raise ValueError(
                "Coloring length {} does not match the number of measurable terms {}.".format(
                    len(colors),
                    len(terms),
                )
            )
        for node, color in enumerate(colors):
            add_node(node, color)

    groups = [color_to_terms[color] for color in sorted(color_to_terms)]
    _validate_nonoverlapping_groups(groups, terms)
    return groups


class OverlappingAuxiliary:
    """
    Class required for passing cov_dict and number of iterations to
    OverlappingGroups.
    """

    def __init__(self, cov_dict, n_iter=5):
        self.cov_dict = cov_dict
        self.n_iter = n_iter


class OverlappingGroupsWoFixed:
    """
    Tequila-derived helper used to eliminate one fixed coefficient per term when
    solving the overlapping-coefficient linear system.
    """

    def __init__(self, o_groups, o_terms, term_exists_in):
        def exclude_fixed_coeffs(o_groups, o_terms, term_exists_in):
            fixed_grp = []
            o_groups_wo_fixed = deepcopy(o_groups)
            term_exists_in_wo_fixed = deepcopy(term_exists_in)
            for term_idx, term in enumerate(o_terms):
                fixed_grp.append(term_exists_in[term_idx][-1])
                o_groups_wo_fixed[fixed_grp[term_idx]].remove(term)
                term_exists_in_wo_fixed[term_idx].remove(fixed_grp[term_idx])
            n_coeff_grp = np.array([len(lst) for lst in o_groups_wo_fixed])
            init_idx = [sum(n_coeff_grp[:i]) for i in range(len(o_groups_wo_fixed))]
            return fixed_grp, o_groups_wo_fixed, term_exists_in_wo_fixed, n_coeff_grp, init_idx

        def get_term_idxs(o_terms, o_groups_wo_fixed, term_exists_in_wo_fixed, init_idx):
            term_idxs = {}
            for term_idx, term in enumerate(o_terms):
                cur_idxs = {}
                for grp_idx in term_exists_in_wo_fixed[term_idx]:
                    cur_idxs[grp_idx] = init_idx[grp_idx] + o_groups_wo_fixed[grp_idx].index(term)
                term_idxs[term.binary_tuple()] = cur_idxs
            return term_idxs

        self.fixed_grp, self.o_groups, self.term_exists_in, self.n_coeff_grp, self.init_idx = exclude_fixed_coeffs(
            o_groups,
            o_terms,
            term_exists_in,
        )
        self.term_idxs = get_term_idxs(o_terms, self.o_groups, self.term_exists_in, self.init_idx)


def get_cov(term1, term2, cov_dict):
    """
    Return the covariance between commuting Pauli strings `term1` and `term2`.
    """
    if (term1.binary_tuple(), term2.binary_tuple()) in cov_dict:
        return cov_dict[(term1.binary_tuple(), term2.binary_tuple())]
    if (term2.binary_tuple(), term1.binary_tuple()) in cov_dict:
        return cov_dict[(term2.binary_tuple(), term1.binary_tuple())]
    raise KeyError("Covariance not found for terms {} and {}.".format(term1.binary_tuple(), term2.binary_tuple()))


def cov_term_w_group(term, group, cov_dict):
    cov = 0.0
    for grp_term in group:
        cov += grp_term.get_coeff() * get_cov(term, grp_term, cov_dict)
    return cov


def get_opt_sample_size(groups, cov_dict):
    """
    Allocate sample sizes optimally based on group variances.
    """
    weights = np.zeros(len(groups))
    for idx, group in enumerate(groups):
        cur_var = 0.0
        for term1 in group:
            for term2 in group:
                cur_var += term1.coeff * term2.coeff * get_cov(term1, term2, cov_dict)
        weights[idx] = np.sqrt(np.real(cur_var))
    return weights / np.sum(weights)


class OverlappingGroups:
    """
    Tequila-derived overlapping grouping implementation with two extra entry
    points:
    - `init_from_groups`: start from user-specified non-overlapping groups
    - `init_from_gflow_grouping`: start from a GFlowNet-style coloring
    """

    def __init__(self, no_groups, o_terms, term_exists_in):
        self.no_groups = no_groups
        self.o_terms = o_terms
        self.term_exists_in = term_exists_in
        self.o_groups = [[] for _ in range(len(no_groups))]
        for idx, term in enumerate(o_terms):
            for group_idx in term_exists_in[idx]:
                self.o_groups[group_idx].append(term)
        self.wo_fixed = OverlappingGroupsWoFixed(self.o_groups, self.o_terms, self.term_exists_in)

    @classmethod
    def init_from_binary_terms(cls, terms, condition="fc"):
        nonoverlapping_groups = sorted_insertion_grouping(terms, condition=condition)
        return cls.init_from_groups(nonoverlapping_groups, condition=condition, terms=terms)

    @classmethod
    def init_from_groups(cls, groups, condition="fc", terms=None):
        """
        Build the overlapping-group data structure from user-supplied
        non-overlapping groups instead of Tequila's internal sorted insertion.
        """
        nonoverlapping_groups = _normalize_groups(groups)
        if terms is None:
            terms = _unique_terms(nonoverlapping_groups)
        else:
            terms = extract_measurable_terms(terms)

        _validate_nonoverlapping_groups(nonoverlapping_groups, terms)

        newly_added = [[] for _ in range(len(nonoverlapping_groups))]
        sorted_terms = sorted(terms, key=lambda x: np.abs(x.coeff), reverse=True)
        overlapping_terms = []
        term_exists_in = []
        for term in sorted_terms:
            group_indices = []
            for idx, group in enumerate(nonoverlapping_groups):
                commute = term_commutes_with_group(term, group, condition) and term_commutes_with_group(
                    term,
                    newly_added[idx],
                    condition,
                )
                if commute:
                    group_indices.append(idx)
                    newly_added[idx].append(term)
            if len(group_indices) > 1:
                overlapping_terms.append(term.term_w_coeff(0.0))
                term_exists_in.append(group_indices)
        return cls(nonoverlapping_groups, overlapping_terms, term_exists_in)

    @classmethod
    def init_from_gflow_grouping(cls, grouping, terms, condition="fc"):
        nonoverlapping_groups = groups_from_gflow_grouping(grouping, extract_measurable_terms(terms))
        return cls.init_from_groups(nonoverlapping_groups, condition=condition)

    def optimize_pauli_coefficients(self, cov_dict, sample_size):
        def prep_mat_single_row(term, group_index):
            mat_single = np.zeros(np.sum(self.wo_fixed.n_coeff_grp))
            for term2 in self.o_groups[group_index]:
                term2_idx_dict = self.wo_fixed.term_idxs[term2.binary_tuple()]
                cov = np.real_if_close(get_cov(term, term2, cov_dict))
                if term2 in self.wo_fixed.o_groups[group_index]:
                    mat_single[term2_idx_dict[group_index]] -= cov / sample_size[group_index]
                else:
                    for idx in term2_idx_dict.values():
                        mat_single[idx] += cov / sample_size[group_index]
            return mat_single

        def prep_b_single_row(term, group_index):
            return np.real_if_close(
                cov_term_w_group(term, self.no_groups[group_index], cov_dict) / sample_size[group_index]
            )

        mat_size = np.sum(self.wo_fixed.n_coeff_grp)
        matrix = np.zeros((mat_size, mat_size))
        b = np.zeros((1, mat_size))
        row_idx = 0
        for grp_idx, grp in enumerate(self.wo_fixed.o_groups):
            for term1 in grp:
                matrix[row_idx] += prep_mat_single_row(term1, grp_idx)
                b[0, row_idx] += prep_b_single_row(term1, grp_idx)

                fixed_group_index = self.wo_fixed.fixed_grp[self.o_terms.index(term1)]
                matrix[row_idx] -= prep_mat_single_row(term1, fixed_group_index)
                b[0, row_idx] -= prep_b_single_row(term1, fixed_group_index)
                row_idx += 1
        sol = np.linalg.lstsq(matrix, b.T, rcond=None)[0]
        return sol.T[0]

    def overlapping_groups_from_coeff(self, coeff):
        def add_coeff_times_term(cur_coeff, term, group_index):
            added = False
            for term_idx, group_term in enumerate(final_overlapping_groups[group_index]):
                if group_term.binary_tuple() == term.binary_tuple():
                    final_overlapping_groups[group_index][term_idx].set_coeff(cur_coeff + group_term.get_coeff())
                    added = True
            if not added:
                final_overlapping_groups[group_index].append(term.term_w_coeff(cur_coeff))

        final_overlapping_groups = deepcopy(self.no_groups)
        for term_idx, term in enumerate(self.o_terms):
            fixed_group_coefficient = 0.0
            for grp_idx in self.wo_fixed.term_exists_in[term_idx]:
                cur_coeff = coeff[self.wo_fixed.init_idx[grp_idx] + self.wo_fixed.o_groups[grp_idx].index(term)]
                fixed_group_coefficient -= cur_coeff
                add_coeff_times_term(cur_coeff, term, grp_idx)

            fixed_idx = self.wo_fixed.fixed_grp[term_idx]
            add_coeff_times_term(fixed_group_coefficient, term, fixed_idx)
        return final_overlapping_groups

    def optimal_overlapping_groups(self, overlap_aux):
        cur_groups = self.no_groups
        for _ in range(overlap_aux.n_iter):
            cur_sample_size = get_opt_sample_size(cur_groups, overlap_aux.cov_dict)
            coeff = self.optimize_pauli_coefficients(overlap_aux.cov_dict, cur_sample_size)
            cur_groups = self.overlapping_groups_from_coeff(coeff)
        return cur_groups


def iterative_coefficient_splitting_from_groups(initial_groups, cov_dict, condition="fc"):
    """
    Run ICS starting from arbitrary non-overlapping groups.
    """
    overlap_aux = OverlappingAuxiliary(cov_dict)
    overlapping_groups = OverlappingGroups.init_from_groups(initial_groups, condition=condition)
    groups = overlapping_groups.optimal_overlapping_groups(overlap_aux)
    suggested_sample_size = get_opt_sample_size(groups, cov_dict)
    return [BinaryHamiltonian(group) for group in groups], suggested_sample_size


def iterative_coefficient_splitting_from_gflow_grouping(
    binary_hamiltonian,
    grouping,
    cov_dict,
    condition="fc",
):
    """
    Run ICS starting from a GFlowNet-compatible coloring produced by this repo.
    """
    measurable_terms = extract_measurable_terms(binary_hamiltonian)
    initial_groups = groups_from_gflow_grouping(grouping, measurable_terms)
    groups, suggested_sample_size = iterative_coefficient_splitting_from_groups(
        initial_groups,
        cov_dict,
        condition=condition,
    )
    return groups, suggested_sample_size
