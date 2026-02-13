import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from openfermion.linalg import get_sparse_operator
from openfermion.ops import QubitOperator
from tequila.hamiltonian import QubitHamiltonian


ColoringKey = Tuple[int, ...]
CovarianceKey = Tuple[int, int]


@dataclass(frozen=True)
class CovarianceRewardData:
    """Precomputed coefficients and pairwise covariances for Hamiltonian terms."""

    coefficients: Dict[int, float]
    covariance: Dict[CovarianceKey, float]
    n_terms: int
    tiny: float = 1e-12


def _to_coloring_key(colors: Sequence[int]) -> ColoringKey:
    return tuple(int(c) for c in colors)


def _safe_real(value: complex, tiny: float) -> float:
    value = complex(value)
    out = float(value.real)
    if abs(out) < tiny:
        return 0.0
    return out


def _extract_unit_pauli_and_coeff(pauli_string, tiny: float) -> Tuple[QubitOperator, float]:
    """
    Convert a tequila PauliString into:
    1) its unit-coefficient OpenFermion Pauli term P_j
    2) its scalar coefficient c_j
    """
    qh = QubitHamiltonian.from_paulistrings([pauli_string])
    of_op = qh.to_openfermion()
    non_constant_terms = [(term, coeff) for term, coeff in of_op.terms.items() if term != ()]
    if len(non_constant_terms) != 1:
        raise ValueError(
            "Expected one non-constant Pauli term per node, got {} terms".format(len(non_constant_terms))
        )
    pauli_term, coeff = non_constant_terms[0]
    coeff_real = _safe_real(coeff, tiny)
    return QubitOperator(pauli_term, 1.0), coeff_real


def build_covariance_reward_data(
    graph: nx.Graph,
    wfn,
    n_qubit: int,
    tiny: float = 1e-12,
) -> CovarianceRewardData:
    """
    Precompute covariance dictionary:
        Cov(P_j, P_k) = <P_j P_k> - <P_j><P_k>
    where P_j are unit-coefficient Pauli products and coefficients are stored separately.
    """
    n_terms = nx.number_of_nodes(graph)
    coefficients: Dict[int, float] = {}
    sparse_ops = {}
    expectations = {}

    for node in range(n_terms):
        pauli_obj = graph.nodes[node]["v"]
        unit_pauli, coeff = _extract_unit_pauli_and_coeff(pauli_obj, tiny=tiny)
        coefficients[node] = coeff
        op_sparse = get_sparse_operator(unit_pauli, n_qubits=n_qubit)
        sparse_ops[node] = op_sparse
        expectations[node] = np.vdot(wfn, op_sparse.dot(wfn))

    covariance: Dict[CovarianceKey, float] = {}
    for i in range(n_terms):
        for j in range(i, n_terms):
            exp_ij = np.vdot(wfn, sparse_ops[i].dot(sparse_ops[j].dot(wfn)))
            cov_ij = exp_ij - expectations[i] * expectations[j]
            cov_ij_real = _safe_real(cov_ij, tiny=tiny)
            covariance[(i, j)] = cov_ij_real
            covariance[(j, i)] = cov_ij_real

    return CovarianceRewardData(
        coefficients=coefficients,
        covariance=covariance,
        n_terms=n_terms,
        tiny=tiny,
    )


def build_covariance_dictionary(
    graph: nx.Graph,
    wfn,
    n_qubit: int,
    tiny: float = 1e-12,
) -> Dict[CovarianceKey, float]:
    """
    Convenience helper returning only Cov(P_j, P_k).
    """
    return build_covariance_reward_data(
        graph=graph,
        wfn=wfn,
        n_qubit=n_qubit,
        tiny=tiny,
    ).covariance.copy()


def covariance_reward_data_from_dicts(
    n_terms: int,
    coefficients: Mapping[int, float],
    covariance: Mapping[CovarianceKey, float],
    tiny: float = 1e-12,
) -> CovarianceRewardData:
    """
    Build CovarianceRewardData from user-provided dictionaries.
    Missing symmetric entries are mirrored automatically.
    """
    coeffs = {int(k): float(v) for k, v in coefficients.items()}
    cov: Dict[CovarianceKey, float] = {}
    for (i, j), value in covariance.items():
        ii, jj = int(i), int(j)
        vv = float(value)
        cov[(ii, jj)] = vv
        cov[(jj, ii)] = vv
    return CovarianceRewardData(coefficients=coeffs, covariance=cov, n_terms=int(n_terms), tiny=tiny)


class CovarianceRewardEngine:
    """
    Fast reward evaluator based on precomputed covariance values.
    Uses:
        Var(H_alpha) = sum_{j,k in alpha} c_j c_k Cov(P_j, P_k)
    and
        eps^2 M = (sum_alpha sqrt(Var(H_alpha)))^2
    """

    def __init__(
        self,
        graph: nx.Graph,
        reward_data: CovarianceRewardData,
        l0: float,
        l1: float,
        cache_limit: int = 100_000,
    ):
        self.graph = graph
        self.data = reward_data
        self.l0 = float(l0)
        self.l1 = float(l1)
        self.tiny = reward_data.tiny
        self.cache_limit = int(cache_limit)

        self._neighbors = [tuple(graph.neighbors(i)) for i in range(nx.number_of_nodes(graph))]
        self._reward_cache: MutableMapping[ColoringKey, float] = {}
        self._eps_cache: MutableMapping[ColoringKey, float] = {}

    def _trim_cache_if_needed(self) -> None:
        if len(self._reward_cache) > self.cache_limit:
            self._reward_cache.clear()
        if len(self._eps_cache) > self.cache_limit:
            self._eps_cache.clear()

    def _valid_coloring(self, key: ColoringKey) -> bool:
        for i, nbrs in enumerate(self._neighbors):
            ci = key[i]
            for j in nbrs:
                if ci == key[j]:
                    return False
        return True

    def _groups_from_coloring(self, key: ColoringKey) -> Dict[int, list]:
        groups: Dict[int, list] = {}
        for i, c in enumerate(key):
            groups.setdefault(c, []).append(i)
        return groups

    def _group_variance(self, nodes: Iterable[int]) -> float:
        nodes = list(nodes)
        coeffs = self.data.coefficients
        cov = self.data.covariance
        v = 0.0
        for i in nodes:
            ci = coeffs[i]
            for j in nodes:
                v += ci * coeffs[j] * cov[(i, j)]
        if v < self.tiny:
            return 0.0
        return float(v)

    def eps2m_from_coloring(self, colors: Sequence[int]) -> float:
        key = _to_coloring_key(colors)
        cached = self._eps_cache.get(key)
        if cached is not None:
            return cached

        groups = self._groups_from_coloring(key)
        root_sum = 0.0
        for nodes in groups.values():
            var_g = self._group_variance(nodes)
            root_sum += math.sqrt(max(var_g, 0.0))
        eps2m = float(root_sum * root_sum)
        self._eps_cache[key] = eps2m
        self._trim_cache_if_needed()
        return eps2m

    def reward_from_coloring(self, colors: Sequence[int]) -> float:
        key = _to_coloring_key(colors)
        cached = self._reward_cache.get(key)
        if cached is not None:
            return cached

        if self.l0 == 0.0 and self.l1 == 0.0:
            self._reward_cache[key] = 0.0
            return 0.0

        if not self._valid_coloring(key):
            self._reward_cache[key] = 0.0
            return 0.0

        reward = 0.0
        if self.l0 != 0.0:
            eps2m = self.eps2m_from_coloring(key)
            reward += self.l0 / max(eps2m, self.tiny)
        if self.l1 != 0.0:
            n_colors = len(set(key))
            reward += self.l1 * (len(key) - n_colors)

        reward = float(max(reward, 0.0))
        self._reward_cache[key] = reward
        self._trim_cache_if_needed()
        return reward

    def reward(self, colors: torch.Tensor) -> float:
        if isinstance(colors, torch.Tensor):
            return self.reward_from_coloring(colors.detach().cpu().tolist())
        return self.reward_from_coloring(colors)

    def edge_intermediate_reward(
        self,
        reward_prev: float,
        reward_next: float,
        scale: float = 1.0,
        clip_negative: bool = True,
    ) -> float:
        delta = float(reward_next - reward_prev)
        if clip_negative and delta < 0.0:
            delta = 0.0
        return float(scale * delta)

    def directional_log_reward_gain(
        self,
        reward_prev: float,
        reward_next: float,
    ) -> float:
        return float(math.log(max(reward_next, self.tiny)) - math.log(max(reward_prev, self.tiny)))
