import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from openfermion.linalg import get_sparse_operator
from tequila.grouping.binary_rep import BinaryHamiltonian

import gflow_vqe.hamiltonians as hamlib
from gflow_vqe.circuit_helpers import grouping_circuit_stats_tequila
from gflow_vqe.overlapping_helpers import (
    as_tequila_wavefunction,
    get_opt_sample_size,
    iterative_coefficient_splitting_from_gflow_grouping,
    prepare_cov_dict,
)
from gflow_vqe.utils import color_reward, get_variance_wavefunction


COLOR_DISTANCE_LATEX = (
    r"D_{\mathrm{color}}(x,y)=\frac{1}{N_P}\sum_{i=1}^{N_P}"
    r"\mathbf{1}\!\left[c_x(i)\ne c_y(i)\right]"
)

ICS_SUBSET_SELECTORS = ("measurement_25%", "two_qubit_25%")
ICS_REPORT_SET_SELECTORS = (
    "pareto_fronts",
    "pareto_fronts_plus_measurement_25%",
    "measurement_25%",
)

_ICS_WORKER_CONTEXT = None


@dataclass
class Candidate:
    candidate_id: int
    graph: object
    measurement: float
    num_groups: int
    two_qubit_gates: int
    signature: tuple
    comembership: np.ndarray
    color_vector: np.ndarray
    sources: list = field(default_factory=list)
    ics_measurement: float | None = None
    ics_runtime: float | None = None
    ics_error: str | None = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Collect composite-reward Pareto-front candidates and evaluate "
            "NO-vs-ICS quality, rank correlation, measurement/two-qubit "
            "scalarized regret, and grouping diversity."
        )
    )
    parser.add_argument(
        "func_name",
        choices=("H4", "LiH", "BeH2"),
        help="Molecule name. Expected folder layout: Molecule_Batch_Reward/L0_value/Molecule_sampled_graphs_metrics.p",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory containing <Molecule>_Batch_Reward (default: current directory).",
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=None,
        help="Override the batch directory. Defaults to <root>/<Molecule>_Batch_Reward.",
    )
    parser.add_argument(
        "--pareto-objectives",
        choices=("two-qubit",),
        default="two-qubit",
        help=(
            "Objectives used to compute each L0 folder Pareto front. "
            "This composite-reward analysis uses eps^2M and N_2q only."
        ),
    )
    parser.add_argument(
        "--weight-step",
        type=float,
        default=0.1,
        help="Grid step for scalarized weights over eps^2M and N_2q (default: 0.1).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional prefix for the candidate CSV output file.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help=(
            "Directory for compact Pareto/subset pickle artifacts. Defaults to --root. "
            "Files are named <Molecule>_ParetoFromL0Folders.p, <Molecule>_10Mset.p, etc."
        ),
    )
    parser.add_argument(
        "--refresh-pareto-cache",
        action="store_true",
        help="Rebuild <Molecule>_ParetoFromL0Folders.p even if it already exists.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes for ICS calculations. Defaults to os.cpu_count().",
    )
    args = parser.parse_args(argv)
    if args.weight_step <= 0 or args.weight_step > 1:
        parser.error("--weight-step must be in the interval (0, 1].")
    if args.num_workers is not None and args.num_workers < 1:
        parser.error("--num-workers must be at least 1.")
    return args


def pareto_front_min(points):
    if len(points) == 0:
        return np.array([], dtype=bool)
    is_pareto = np.ones(points.shape[0], dtype=bool)
    for idx, point in enumerate(points):
        if not is_pareto[idx]:
            continue
        dominated = np.any(np.all(points <= point, axis=1) & np.any(points < point, axis=1))
        if dominated:
            is_pareto[idx] = False
    return is_pareto


def objective_matrix(measurements, two_qubit_gates):
    return np.column_stack((measurements, two_qubit_gates.astype(float)))


def find_metric_files(molecule, batch_dir):
    if not batch_dir.exists():
        raise FileNotFoundError("Could not find batch directory '{}'.".format(batch_dir))

    exact_name = "{}_sampled_graphs_metrics.p".format(molecule)
    metric_files = []
    for child in sorted(batch_dir.iterdir()):
        if not child.is_dir():
            continue
        metric_path = child / exact_name
        if not metric_path.exists():
            matches = sorted(child.glob("*_sampled_graphs_metrics.p"))
            if not matches:
                continue
            metric_path = matches[0]
        metric_files.append(metric_path)

    if not metric_files:
        raise FileNotFoundError(
            "No '*_sampled_graphs_metrics.p' files were found under '{}'.".format(batch_dir)
        )
    return metric_files


def graph_path_from_metric_path(metric_path):
    name = metric_path.name
    if not name.endswith("_sampled_graphs_metrics.p"):
        raise ValueError("Unexpected metrics file name '{}'.".format(metric_path))
    graph_name = name.replace("_sampled_graphs_metrics.p", "_sampled_graphs.p")
    graph_path = metric_path.with_name(graph_name)
    if not graph_path.exists():
        raise FileNotFoundError(
            "Could not find graph file '{}' corresponding to '{}'.".format(graph_path, metric_path)
        )
    return graph_path


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_graphs_and_metrics(metric_path):
    graph_path = graph_path_from_metric_path(metric_path)
    graphs = load_pickle(graph_path)
    metrics = load_pickle(metric_path)
    if not isinstance(graphs, list):
        graphs = list(graphs)

    required_keys = ("measurements", "num_groups", "two_qubit_gates")
    if not isinstance(metrics, dict) or any(key not in metrics for key in required_keys):
        raise ValueError("Metrics file '{}' does not match the expected cache format.".format(metric_path))

    metrics = {key: np.asarray(metrics[key]) for key in required_keys}
    valid_indices = [idx for idx, graph in enumerate(graphs) if color_reward(graph) > 0]
    valid_graphs = [graphs[idx] for idx in valid_indices]
    if not valid_graphs:
        raise RuntimeError("No valid graphs were found in '{}'.".format(graph_path))

    if metrics["measurements"].shape[0] == len(valid_graphs):
        aligned_metrics = metrics
        aligned_indices = list(range(len(valid_graphs)))
    elif metrics["measurements"].shape[0] == len(graphs):
        aligned_metrics = {key: values[valid_indices] for key, values in metrics.items()}
        aligned_indices = valid_indices
    else:
        raise ValueError(
            "Metrics length in '{}' is {}, but graph file has {} graphs and {} valid graphs.".format(
                metric_path,
                metrics["measurements"].shape[0],
                len(graphs),
                len(valid_graphs),
            )
        )

    return valid_graphs, aligned_metrics, aligned_indices


def graph_colors(graph):
    colors = {}
    for node, data in graph.nodes(data=True):
        if "color" not in data:
            raise ValueError("Graph node {} is missing a 'color' attribute.".format(node))
        colors[node] = data["color"]
    return [colors[node] for node in sorted(colors)]


def color_vector(graph):
    return np.asarray(graph_colors(graph), dtype=object)


def comembership_vector_from_colors(colors):
    pairs = []
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            pairs.append(colors[i] == colors[j])
    return np.asarray(pairs, dtype=bool)


def comembership_vector(graph):
    return comembership_vector_from_colors(color_vector(graph))


def make_candidate(candidate_id, graph, measurement, num_groups, two_qubit_gates, source):
    colors = color_vector(graph)
    comembership = comembership_vector_from_colors(colors)
    return Candidate(
        candidate_id=candidate_id,
        graph=graph,
        measurement=float(measurement),
        num_groups=int(num_groups),
        two_qubit_gates=int(two_qubit_gates),
        signature=tuple(comembership.tolist()),
        comembership=comembership,
        color_vector=colors,
        sources=[source],
    )


def collect_candidates(molecule, batch_dir, pareto_objectives):
    candidates = []
    next_id = 0
    metric_files = find_metric_files(molecule, batch_dir)

    for metric_path in metric_files:
        graphs, metrics, original_indices = load_graphs_and_metrics(metric_path)
        measurements = metrics["measurements"].astype(float)
        num_groups = metrics["num_groups"].astype(int)
        two_qubit_gates = metrics["two_qubit_gates"].astype(int)
        points = objective_matrix(measurements, two_qubit_gates)
        pareto_mask = pareto_front_min(points)
        pareto_indices = np.flatnonzero(pareto_mask)

        print(
            "Loaded {} valid graphs from {}; folder Pareto points={}".format(
                len(graphs),
                metric_path.parent,
                len(pareto_indices),
            )
        )
        for idx in pareto_indices:
            graph = graphs[idx]
            candidates.append(
                make_candidate(
                    next_id,
                    graph,
                    measurements[idx],
                    num_groups[idx],
                    two_qubit_gates[idx],
                    "{}#{}".format(
                        metric_path.parent.as_posix(),
                        int(original_indices[idx]),
                    ),
                )
            )
            next_id += 1

    return deduplicate_candidates(candidates), metric_files


def collect_candidates_within_thresholds(metric_files, measurement_threshold, two_qubit_threshold):
    candidates = []
    next_id = 0

    for metric_path in metric_files:
        graphs, metrics, original_indices = load_graphs_and_metrics(metric_path)
        measurements = metrics["measurements"].astype(float)
        num_groups = metrics["num_groups"].astype(int)
        two_qubit_gates = metrics["two_qubit_gates"].astype(int)
        selected_count = 0

        for idx, graph in enumerate(graphs):
            if measurements[idx] > measurement_threshold and two_qubit_gates[idx] > two_qubit_threshold:
                continue
            candidates.append(
                make_candidate(
                    next_id,
                    graph,
                    measurements[idx],
                    num_groups[idx],
                    two_qubit_gates[idx],
                    "{}#{}".format(
                        metric_path.parent.as_posix(),
                        int(original_indices[idx]),
                    ),
                )
            )
            next_id += 1
            selected_count += 1

        print(
            "Loaded {} threshold candidates from {}.".format(
                selected_count,
                metric_path.parent,
            )
        )

    return deduplicate_candidates(candidates)


def candidate_to_record(candidate):
    return {
        "candidate_id": candidate.candidate_id,
        "graph": candidate.graph,
        "measurement": candidate.measurement,
        "num_groups": candidate.num_groups,
        "two_qubit_gates": candidate.two_qubit_gates,
        "signature": candidate.signature,
        "comembership": candidate.comembership,
        "color_vector": candidate.color_vector,
        "sources": list(candidate.sources),
        "ics_measurement": candidate.ics_measurement,
        "ics_runtime": candidate.ics_runtime,
        "ics_error": candidate.ics_error,
    }


def candidate_from_record(record):
    graph = record["graph"]
    colors = record.get("color_vector")
    if colors is None:
        colors = color_vector(graph)
    return Candidate(
        candidate_id=int(record["candidate_id"]),
        graph=graph,
        measurement=float(record["measurement"]),
        num_groups=int(record["num_groups"]),
        two_qubit_gates=int(record["two_qubit_gates"]),
        signature=tuple(record["signature"]),
        comembership=np.asarray(record["comembership"], dtype=bool),
        color_vector=np.asarray(colors, dtype=object),
        sources=list(record.get("sources", [])),
        ics_measurement=record.get("ics_measurement"),
        ics_runtime=record.get("ics_runtime"),
        ics_error=record.get("ics_error"),
    )


def save_candidate_records(path, molecule, candidates, metadata=None):
    payload = {
        "molecule": molecule,
        "metadata": metadata or {},
        "candidates": [candidate_to_record(candidate) for candidate in candidates],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle, pickle.HIGHEST_PROTOCOL)
    print("Saved {} candidates to {}".format(len(candidates), path))


def load_candidate_records(path):
    payload = load_pickle(path)
    if not isinstance(payload, dict) or "candidates" not in payload:
        raise ValueError("Candidate cache '{}' does not match the expected format.".format(path))
    candidates = [candidate_from_record(record) for record in payload["candidates"]]
    return candidates, payload


def collect_or_load_candidates(molecule, batch_dir, pareto_objectives, artifact_dir, refresh_cache=False):
    aggregate_path = artifact_dir / "{}_ParetoFromL0Folders.p".format(molecule)
    if aggregate_path.exists() and not refresh_cache:
        candidates, payload = load_candidate_records(aggregate_path)
        metadata = payload.get("metadata", {})
        cache_matches = (
            metadata.get("batch_dir") == batch_dir.as_posix()
            and metadata.get("pareto_objectives") == pareto_objectives
        )
        if cache_matches:
            metric_files = [Path(path) for path in metadata.get("metric_files", [])]
            print("Loaded {} compact Pareto candidates from {}".format(len(candidates), aggregate_path))
            return candidates, metric_files, aggregate_path
        print(
            "Ignoring stale Pareto cache {}; batch_dir/objectives do not match current arguments.".format(
                aggregate_path
            )
        )

    candidates, metric_files = collect_candidates(molecule, batch_dir, pareto_objectives)
    save_candidate_records(
        aggregate_path,
        molecule,
        candidates,
        metadata={
            "batch_dir": batch_dir.as_posix(),
            "pareto_objectives": pareto_objectives,
            "metric_files": [path.as_posix() for path in metric_files],
            "description": "Unique candidates collected from each L0 folder Pareto front.",
        },
    )
    return candidates, metric_files, aggregate_path


def deduplicate_candidates(candidates):
    unique = {}
    for candidate in candidates:
        existing = unique.get(candidate.signature)
        if existing is None:
            unique[candidate.signature] = candidate
            continue
        existing.sources.extend(candidate.sources)
        if (
            candidate.measurement,
            candidate.two_qubit_gates,
            candidate.num_groups,
        ) < (
            existing.measurement,
            existing.two_qubit_gates,
            existing.num_groups,
        ):
            existing.graph = candidate.graph
            existing.measurement = candidate.measurement
            existing.num_groups = candidate.num_groups
            existing.two_qubit_gates = candidate.two_qubit_gates
            existing.comembership = candidate.comembership
            existing.color_vector = candidate.color_vector

    unique_candidates = list(unique.values())
    for idx, candidate in enumerate(unique_candidates):
        candidate.candidate_id = idx
    return unique_candidates


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


def compute_group_metrics(groups, cov_dict, wfn):
    normalized_groups = [BinaryHamiltonian(list(group.binary_terms)) for group in groups]
    sample_size = get_opt_sample_size([group.binary_terms for group in normalized_groups], cov_dict)
    return optimal_allocation_metric(normalized_groups, sample_size, wfn)


def build_quantum_context(molecule_name):
    molecule_fn = getattr(hamlib, molecule_name, None)
    if molecule_fn is None:
        raise ValueError("Unknown molecule '{}'.".format(molecule_name))
    mol, H, _, n_paulis, Hq = molecule_fn()
    sparse_hamiltonian = get_sparse_operator(Hq)
    energy, fci_wfn = get_variance_wavefunction(
        mol,
        Hq,
        method="FCI",
        sparse_hamiltonian=sparse_hamiltonian,
    )
    binary_hamiltonian = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
    cov_dict = prepare_cov_dict(binary_hamiltonian, fci_wfn)
    return {
        "mol": mol,
        "H": H,
        "Hq": Hq,
        "n_paulis": n_paulis,
        "energy": energy,
        "wfn": fci_wfn,
        "binary_hamiltonian": binary_hamiltonian,
        "cov_dict": cov_dict,
    }


def sorted_insertion_baseline(context):
    binary_hamiltonian = context["binary_hamiltonian"]
    cov_dict = context["cov_dict"]
    wfn = context["wfn"]
    si_groups, _ = binary_hamiltonian.commuting_groups(
        options={"method": "si", "condition": "fc", "cov_dict": cov_dict}
    )
    measurement = compute_group_metrics(si_groups, cov_dict, wfn)
    group_mapping = {idx: list(group.binary_terms) for idx, group in enumerate(si_groups)}
    two_qubit_gates = int(grouping_circuit_stats_tequila(group_mapping).total_two_qubit_gates)
    return {
        "measurement": float(measurement),
        "num_groups": int(len(si_groups)),
        "two_qubit_gates": int(two_qubit_gates),
    }


def compute_ics_metric(candidate, context):
    if candidate.ics_measurement is not None or candidate.ics_error is not None:
        return
    start = time.perf_counter()
    try:
        groups, sample_size = iterative_coefficient_splitting_from_gflow_grouping(
            context["binary_hamiltonian"],
            candidate.graph,
            context["cov_dict"],
            condition="fc",
        )
        candidate.ics_measurement = optimal_allocation_metric(groups, sample_size, context["wfn"])
    except Exception as error:
        candidate.ics_error = "{}: {}".format(type(error).__name__, error)
    candidate.ics_runtime = time.perf_counter() - start


def _init_ics_worker(context):
    global _ICS_WORKER_CONTEXT
    _ICS_WORKER_CONTEXT = context


def _compute_ics_metric_worker(candidate):
    start = time.perf_counter()
    measurement = None
    error_message = None
    try:
        groups, sample_size = iterative_coefficient_splitting_from_gflow_grouping(
            _ICS_WORKER_CONTEXT["binary_hamiltonian"],
            candidate.graph,
            _ICS_WORKER_CONTEXT["cov_dict"],
            condition="fc",
        )
        measurement = optimal_allocation_metric(groups, sample_size, _ICS_WORKER_CONTEXT["wfn"])
    except Exception as error:
        error_message = "{}: {}".format(type(error).__name__, error)
    return {
        "candidate_id": candidate.candidate_id,
        "ics_measurement": measurement,
        "ics_error": error_message,
        "ics_runtime": time.perf_counter() - start,
    }


def compute_ics_metrics_parallel(candidates, context, num_workers):
    worker_context = {
        "binary_hamiltonian": context["binary_hamiltonian"],
        "cov_dict": context["cov_dict"],
        "wfn": context["wfn"],
    }
    pending = [
        candidate
        for candidate in candidates
        if candidate.ics_measurement is None and candidate.ics_error is None
    ]
    if not pending:
        print("Computed ICS for {}/{} candidates.".format(len(candidates), len(candidates)))
        return

    completed = len(candidates) - len(pending)
    if num_workers <= 1:
        for candidate in pending:
            compute_ics_metric(candidate, worker_context)
            completed += 1
            if completed % 10 == 0 or completed == len(candidates):
                print("Computed ICS for {}/{} candidates.".format(completed, len(candidates)))
        return

    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_ics_worker,
        initargs=(worker_context,),
    ) as executor:
        futures = {
            executor.submit(_compute_ics_metric_worker, candidate): candidate
            for candidate in pending
        }
        for future in as_completed(futures):
            submitted_candidate = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {
                    "candidate_id": submitted_candidate.candidate_id,
                    "ics_measurement": None,
                    "ics_error": "{}: {}".format(type(error).__name__, error),
                    "ics_runtime": None,
                }
            candidate = candidates_by_id[result["candidate_id"]]
            candidate.ics_measurement = result["ics_measurement"]
            candidate.ics_error = result["ics_error"]
            candidate.ics_runtime = result["ics_runtime"]
            completed += 1
            if completed % 10 == 0 or completed == len(candidates):
                print("Computed ICS for {}/{} candidates.".format(completed, len(candidates)))


def rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    pos = 0
    while pos < len(values):
        end = pos + 1
        while end < len(values) and values[order[end]] == values[order[pos]]:
            end += 1
        avg_rank = 0.5 * (pos + 1 + end)
        ranks[order[pos:end]] = avg_rank
        pos = end
    return ranks


def spearman_correlation(x_values, y_values):
    if len(x_values) < 2:
        return float("nan")
    x_ranks = rankdata(x_values)
    y_ranks = rankdata(y_values)
    x_std = float(np.std(x_ranks))
    y_std = float(np.std(y_ranks))
    if x_std == 0.0 or y_std == 0.0:
        return float("nan")
    return float(np.corrcoef(x_ranks, y_ranks)[0, 1])


def simplex_weight_grid(step):
    n_steps = int(round(1.0 / step))
    if not np.isclose(n_steps * step, 1.0):
        raise ValueError("--weight-step must evenly divide 1.0; got {}.".format(step))
    return [(i / n_steps, (n_steps - i) / n_steps) for i in range(n_steps + 1)]


def hardware_regret_analysis(candidates, x_no, si_metrics, weight_step):
    weights = simplex_weight_grid(weight_step)
    regrets = []
    different_best = 0
    worst = None

    def scalar_cost(candidate, weight):
        w_m, w_2q = weight
        return (
            w_m * candidate.measurement / si_metrics["measurement"]
            + w_2q * candidate.two_qubit_gates / si_metrics["two_qubit_gates"]
        )

    for weight in weights:
        costs = [(scalar_cost(candidate, weight), candidate) for candidate in candidates]
        best_cost, best_candidate = min(costs, key=lambda item: item[0])
        no_cost = scalar_cost(x_no, weight)
        regret = (no_cost - best_cost) / best_cost if best_cost != 0 else 0.0
        regrets.append(regret)
        if best_candidate.signature != x_no.signature:
            different_best += 1
        if worst is None or regret > worst["regret"]:
            worst = {
                "weight": weight,
                "regret": float(regret),
                "best_candidate_id": best_candidate.candidate_id,
                "best_cost": float(best_cost),
                "x_no_cost": float(no_cost),
            }

    return {
        "weight_count": len(weights),
        "mean_regret": float(np.mean(regrets)) if regrets else float("nan"),
        "max_regret": float(np.max(regrets)) if regrets else float("nan"),
        "different_best_count": int(different_best),
        "worst": worst,
    }


def ics_scalarized_regret_analysis(candidates, x_no, si_metrics, weight_step):
    successful = [candidate for candidate in candidates if candidate.ics_measurement is not None]
    if not successful or x_no.ics_measurement is None:
        return {
            "weight_count": 0,
            "mean_regret": float("nan"),
            "max_regret": float("nan"),
            "different_best_count": 0,
            "worst": None,
        }

    weights = simplex_weight_grid(weight_step)
    regrets = []
    different_best = 0
    worst = None

    def scalar_cost(candidate, weight):
        w_m, w_2q = weight
        return (
            w_m * candidate.ics_measurement / si_metrics["measurement"]
            + w_2q * candidate.two_qubit_gates / si_metrics["two_qubit_gates"]
        )

    for weight in weights:
        costs = [(scalar_cost(candidate, weight), candidate) for candidate in successful]
        best_cost, best_candidate = min(costs, key=lambda item: item[0])
        no_cost = scalar_cost(x_no, weight)
        regret = (no_cost - best_cost) / best_cost if best_cost != 0 else 0.0
        regrets.append(regret)
        if best_candidate.signature != x_no.signature:
            different_best += 1
        if worst is None or regret > worst["regret"]:
            worst = {
                "weight": weight,
                "regret": float(regret),
                "best_candidate_id": best_candidate.candidate_id,
                "best_cost": float(best_cost),
                "x_no_cost": float(no_cost),
            }

    return {
        "weight_count": len(weights),
        "mean_regret": float(np.mean(regrets)) if regrets else float("nan"),
        "max_regret": float(np.max(regrets)) if regrets else float("nan"),
        "different_best_count": int(different_best),
        "worst": worst,
    }


def candidate_metric_summary(candidate):
    if candidate is None:
        return None
    return {
        "candidate_id": candidate.candidate_id,
        "measurement": candidate.measurement,
        "num_groups": candidate.num_groups,
        "two_qubit_gates": candidate.two_qubit_gates,
        "ics_measurement": candidate.ics_measurement,
        "ics_runtime": candidate.ics_runtime,
        "ics_error": candidate.ics_error,
    }


def ics_metrics_report(selector, candidates, x_no, si_metrics, weight_step):
    successful = [candidate for candidate in candidates if candidate.ics_measurement is not None]
    failed = [candidate for candidate in candidates if candidate.ics_error is not None]
    x_ics_best = min(successful, key=lambda candidate: candidate.ics_measurement) if successful else None
    if x_no.ics_measurement is None or x_ics_best is None:
        delta_ics = float("nan")
        delta_ics_percent = float("nan")
    else:
        delta_ics = x_no.ics_measurement - x_ics_best.ics_measurement
        delta_ics_percent = 100.0 * delta_ics / x_no.ics_measurement

    rho = spearman_correlation(
        [candidate.measurement for candidate in successful],
        [candidate.ics_measurement for candidate in successful],
    )

    return {
        "selector": selector,
        "candidate_ids": [candidate.candidate_id for candidate in candidates],
        "count": len(candidates),
        "successful_count": len(successful),
        "failed_count": len(failed),
        "x_no": candidate_metric_summary(x_no),
        "x_ics_best": candidate_metric_summary(x_ics_best),
        "delta_ics": float(delta_ics),
        "delta_ics_percent": float(delta_ics_percent),
        "spearman_rho": rho,
        "spearman_count": len(successful),
        "ics_scalarized_regret": ics_scalarized_regret_analysis(candidates, x_no, si_metrics, weight_step),
    }


def normalized_hamming_distance(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        raise ValueError("Cannot compare vectors with shapes {} and {}.".format(left.shape, right.shape))
    if left.size == 0:
        return 0.0
    return float(np.mean(left != right))


def pairwise_distance_summary(candidates, vector_attr):
    if len(candidates) < 2:
        return 0.0, 0.0
    distances = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            distances.append(
                normalized_hamming_distance(
                    getattr(candidates[i], vector_attr),
                    getattr(candidates[j], vector_attr),
                )
            )
    return float(np.mean(distances)), float(np.max(distances))


def subset_diversity(selector_name, subset):
    if not subset:
        return {
            "selector": selector_name,
            "count": 0,
            "mean_distance": float("nan"),
            "max_distance": float("nan"),
            "mean_comembership_distance": float("nan"),
            "max_comembership_distance": float("nan"),
            "mean_color_distance": float("nan"),
            "max_color_distance": float("nan"),
            "min_groups": None,
            "max_groups": None,
            "min_two_qubit": None,
            "max_two_qubit": None,
        }
    mean_comembership, max_comembership = pairwise_distance_summary(subset, "comembership")
    mean_color, max_color = pairwise_distance_summary(subset, "color_vector")
    return {
        "selector": selector_name,
        "count": len(subset),
        "mean_distance": mean_comembership,
        "max_distance": max_comembership,
        "mean_comembership_distance": mean_comembership,
        "max_comembership_distance": max_comembership,
        "mean_color_distance": mean_color,
        "max_color_distance": max_color,
        "min_groups": int(min(candidate.num_groups for candidate in subset)),
        "max_groups": int(max(candidate.num_groups for candidate in subset)),
        "min_two_qubit": int(min(candidate.two_qubit_gates for candidate in subset)),
        "max_two_qubit": int(max(candidate.two_qubit_gates for candidate in subset)),
    }


def diversity_subsets(candidates, x_no):
    best_measurement = x_no.measurement
    best_two_qubit = min(candidate.two_qubit_gates for candidate in candidates)
    return {
        "measurement_10%": [
            candidate for candidate in candidates if candidate.measurement <= 1.10 * best_measurement
        ],
        "measurement_25%": [
            candidate for candidate in candidates if candidate.measurement <= 1.25 * best_measurement
        ],
        "two_qubit_10%": [
            candidate for candidate in candidates if candidate.two_qubit_gates <= 1.10 * best_two_qubit
        ],
        "two_qubit_25%": [
            candidate for candidate in candidates if candidate.two_qubit_gates <= 1.25 * best_two_qubit
        ],
    }


def build_ics_candidate_set(analysis_candidates, pareto_candidates, subsets):
    pareto_signatures = {candidate.signature for candidate in pareto_candidates}
    subset_signatures = {
        selector: {candidate.signature for candidate in subsets[selector]}
        for selector in ICS_SUBSET_SELECTORS
    }
    selected_signatures = set(pareto_signatures)
    for signatures in subset_signatures.values():
        selected_signatures.update(signatures)

    ics_candidates = []
    selectors_by_candidate_id = {}
    for candidate in analysis_candidates:
        if candidate.signature not in selected_signatures:
            continue
        selectors = []
        if candidate.signature in pareto_signatures:
            selectors.append("folder_pareto")
        for selector in ICS_SUBSET_SELECTORS:
            if candidate.signature in subset_signatures[selector]:
                selectors.append(selector)
        ics_candidates.append(candidate)
        selectors_by_candidate_id[candidate.candidate_id] = selectors

    return ics_candidates, selectors_by_candidate_id


def candidates_with_signatures(candidates, signatures):
    return [candidate for candidate in candidates if candidate.signature in signatures]


def build_ics_report_sets(analysis_candidates, pareto_candidates, subsets):
    pareto_signatures = {candidate.signature for candidate in pareto_candidates}
    measurement_25_signatures = {candidate.signature for candidate in subsets["measurement_25%"]}
    return {
        "pareto_fronts": candidates_with_signatures(analysis_candidates, pareto_signatures),
        "pareto_fronts_plus_measurement_25%": candidates_with_signatures(
            analysis_candidates,
            pareto_signatures | measurement_25_signatures,
        ),
        "measurement_25%": subsets["measurement_25%"],
    }


def ics_subset_summary(selector, subset):
    successful = [candidate for candidate in subset if candidate.ics_measurement is not None]
    failed = [candidate for candidate in subset if candidate.ics_error is not None]
    best = min(successful, key=lambda candidate: candidate.ics_measurement) if successful else None
    return {
        "selector": selector,
        "count": len(subset),
        "successful_count": len(successful),
        "failed_count": len(failed),
        "best": None
        if best is None
        else {
            "candidate_id": best.candidate_id,
            "measurement": best.measurement,
            "num_groups": best.num_groups,
            "two_qubit_gates": best.two_qubit_gates,
            "ics_measurement": best.ics_measurement,
        },
    }


def diversity_analyses(subsets):
    return [subset_diversity(selector, subset) for selector, subset in subsets.items()]


def save_diversity_subset_artifacts(molecule, artifact_dir, subsets):
    name_map = {
        "measurement_10%": "{}_10Mset.p".format(molecule),
        "measurement_25%": "{}_25Mset.p".format(molecule),
        "two_qubit_10%": "{}_10_2qSet.p".format(molecule),
        "two_qubit_25%": "{}_25_2qSet.p".format(molecule),
    }
    paths = {}
    for selector, subset in subsets.items():
        path = artifact_dir / name_map[selector]
        save_candidate_records(
            path,
            molecule,
            subset,
            metadata={
                "selector": selector,
                "description": (
                    "Compact candidate subset used for label-invariant co-membership, "
                    "label-sensitive color-vector diversity, and 25% ICS workset reporting."
                ),
            },
        )
        paths[selector] = path
    return paths


def save_ics_result_artifact(molecule, artifact_dir, ics_candidates, ics_candidate_selectors, ics_report_sets):
    path = artifact_dir / "{}_ICSResults.p".format(molecule)
    save_candidate_records(
        path,
        molecule,
        ics_candidates,
        metadata={
            "description": (
                "Compact candidate records for all ICS calculations. Candidate records include "
                "original metrics, graph objects, ICS measurements, runtimes, and errors."
            ),
            "ics_selectors": {
                candidate_id: list(selectors)
                for candidate_id, selectors in ics_candidate_selectors.items()
            },
            "report_sets": {
                selector: [candidate.candidate_id for candidate in candidates]
                for selector, candidates in ics_report_sets.items()
            },
        },
    )
    return path


def write_outputs(prefix, candidates, ics_candidate_ids=None, ics_candidate_selectors=None):
    prefix = Path(prefix)
    candidates_path = prefix.with_suffix(".candidates.csv")
    ics_candidate_ids = set(ics_candidate_ids or [])
    ics_candidate_selectors = ics_candidate_selectors or {}

    with open(candidates_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "candidate_id",
                "measurement",
                "num_groups",
                "two_qubit_gates",
                "ics_measurement",
                "ics_error",
                "ics_selected",
                "ics_selectors",
                "sources",
            ),
        )
        writer.writeheader()
        for candidate in candidates:
            selectors = ics_candidate_selectors.get(candidate.candidate_id, [])
            writer.writerow(
                {
                    "candidate_id": candidate.candidate_id,
                    "measurement": candidate.measurement,
                    "num_groups": candidate.num_groups,
                    "two_qubit_gates": candidate.two_qubit_gates,
                    "ics_measurement": candidate.ics_measurement,
                    "ics_error": candidate.ics_error,
                    "ics_selected": candidate.candidate_id in ics_candidate_ids,
                    "ics_selectors": ";".join(selectors),
                    "sources": ";".join(candidate.sources),
                }
            )

    print("")
    print("Saved candidate table to {}".format(candidates_path))


def print_diversity_result(result):
    print(
        "  {selector}: |S|={count}, D_co mean/max={mean_comembership_distance:.6g}/{max_comembership_distance:.6g}, "
        "D_color mean/max={mean_color_distance:.6g}/{max_color_distance:.6g}, "
        "N_G min/max={min_groups}/{max_groups}, N_2q min/max={min_two_qubit}/{max_two_qubit}".format(
            **result
        )
    )


def main(argv=None):
    args = parse_args(argv)
    batch_dir = args.batch_dir if args.batch_dir is not None else args.root / "{}_Batch_Reward".format(args.func_name)
    artifact_dir = args.artifact_dir if args.artifact_dir is not None else args.root
    worker_count = args.num_workers if args.num_workers is not None else (os.cpu_count() or 1)
    pareto_candidates, metric_files, aggregate_path = collect_or_load_candidates(
        args.func_name,
        batch_dir,
        args.pareto_objectives,
        artifact_dir,
        refresh_cache=args.refresh_pareto_cache,
    )
    if not pareto_candidates:
        raise RuntimeError("No Pareto-front candidates were collected.")
    if not metric_files:
        metric_files = find_metric_files(args.func_name, batch_dir)

    x_no = min(pareto_candidates, key=lambda candidate: candidate.measurement)
    best_two_qubit = min(candidate.two_qubit_gates for candidate in pareto_candidates)
    measurement_25_threshold = 1.25 * x_no.measurement
    two_qubit_25_threshold = 1.25 * best_two_qubit

    threshold_candidates = collect_candidates_within_thresholds(
        metric_files,
        measurement_25_threshold,
        two_qubit_25_threshold,
    )
    analysis_candidates = deduplicate_candidates(pareto_candidates + threshold_candidates)
    subsets = diversity_subsets(analysis_candidates, x_no)
    ics_report_sets = build_ics_report_sets(analysis_candidates, pareto_candidates, subsets)
    ics_candidates, ics_candidate_selectors = build_ics_candidate_set(
        analysis_candidates,
        pareto_candidates,
        subsets,
    )
    ics_candidate_ids = {candidate.candidate_id for candidate in ics_candidates}

    print("")
    print("Collected {} unique Pareto candidates from {} L0 folders.".format(len(pareto_candidates), len(metric_files)))
    print("Compact Pareto candidate file={}".format(aggregate_path))
    print("25% measurement threshold={:.12g}".format(measurement_25_threshold))
    print("25% two-qubit threshold={:.12g}".format(two_qubit_25_threshold))
    print("Analysis candidate set=Pareto union 25% threshold points ({} candidates)".format(len(analysis_candidates)))
    print(
        "ICS candidate set=folder Pareto union measurement_25% union two_qubit_25% "
        "({} candidates)".format(len(ics_candidates))
    )
    print("Additional ICS report sets:")
    for selector in ICS_REPORT_SET_SELECTORS:
        print("  {}: {} candidates".format(selector, len(ics_report_sets[selector])))

    print("")
    print("Building FCI Hamiltonian context for ICS/SI baseline...")
    context = build_quantum_context(args.func_name)
    print(
        "FCI energy={}, Pauli products={}, covariance entries={}".format(
            context["energy"],
            context["n_paulis"],
            len(context["cov_dict"]),
        )
    )
    si_metrics = sorted_insertion_baseline(context)
    print(
        "SI baseline: eps^2M={:.12g}, N_G={}, N_2q={}".format(
            si_metrics["measurement"],
            si_metrics["num_groups"],
            si_metrics["two_qubit_gates"],
        )
    )

    print("")
    print(
        "Single-best NO grouping x_NO*: id={}, eps^2M_NO={:.12g}, N_G={}, N_2q={}".format(
            x_no.candidate_id,
            x_no.measurement,
            x_no.num_groups,
            x_no.two_qubit_gates,
        )
    )

    compute_ics_metrics_parallel(ics_candidates, context, worker_count)

    successful_ics = [candidate for candidate in ics_candidates if candidate.ics_measurement is not None]
    if x_no.ics_measurement is None:
        delta_ics = float("nan")
        x_ics_best = None
        print("ICS from x_NO* failed: {}".format(x_no.ics_error))
    elif successful_ics:
        x_ics_best = min(successful_ics, key=lambda candidate: candidate.ics_measurement)
        delta_ics = 100.0 * (x_no.ics_measurement - x_ics_best.ics_measurement) / x_no.ics_measurement
    else:
        x_ics_best = None
        delta_ics = float("nan")

    print("")
    print("NO-vs-ICS comparison:")
    print("  eps^2M_ICS(x_NO*)={}".format(x_no.ics_measurement))
    if x_ics_best is not None:
        print(
            "  best sampled ICS: id={}, eps^2M_ICS={:.12g}, NO eps^2M={:.12g}".format(
                x_ics_best.candidate_id,
                x_ics_best.ics_measurement,
                x_ics_best.measurement,
            )
        )
    print("  Delta_ICS (%)={}".format(delta_ics))

    ics_metric_reports = {
        selector: ics_metrics_report(
            selector,
            ics_report_sets[selector],
            x_no,
            si_metrics,
            args.weight_step,
        )
        for selector in ICS_REPORT_SET_SELECTORS
    }
    print("")
    print("ICS metrics by candidate set:")
    for selector in ICS_REPORT_SET_SELECTORS:
        result = ics_metric_reports[selector]
        best = result["x_ics_best"]
        regret_result = result["ics_scalarized_regret"]
        if best is None:
            print(
                "  {}: successful={}/{}, failed={}".format(
                    selector,
                    result["successful_count"],
                    result["count"],
                    result["failed_count"],
                )
            )
            continue
        print(
            "  {}: successful={}/{}, failed={}, best id={}, "
            "Delta_ICS={:.12g}%, rho={}, ICS regret mean/max={:.12g}/{:.12g}".format(
                selector,
                result["successful_count"],
                result["count"],
                result["failed_count"],
                best["candidate_id"],
                result["delta_ics_percent"],
                result["spearman_rho"],
                regret_result["mean_regret"],
                regret_result["max_regret"],
            )
        )

    ics_subset_summaries = {
        selector: ics_subset_summary(selector, subsets[selector])
        for selector in ICS_SUBSET_SELECTORS
    }
    print("")
    print("ICS results on 25% subsets:")
    for selector in ICS_SUBSET_SELECTORS:
        result = ics_subset_summaries[selector]
        best = result["best"]
        if best is None:
            print(
                "  {}: successful={}/{}, failed={}".format(
                    selector,
                    result["successful_count"],
                    result["count"],
                    result["failed_count"],
                )
            )
        else:
            print(
                "  {}: successful={}/{}, failed={}, best id={}, eps^2M_ICS={:.12g}".format(
                    selector,
                    result["successful_count"],
                    result["count"],
                    result["failed_count"],
                    best["candidate_id"],
                    best["ics_measurement"],
                )
            )

    corr_candidates = [candidate for candidate in ics_candidates if candidate.ics_measurement is not None]
    rho = spearman_correlation(
        [candidate.measurement for candidate in corr_candidates],
        [candidate.ics_measurement for candidate in corr_candidates],
    )
    print("")
    print("Rank correlation:")
    print("  Spearman rho(NO eps^2M, ICS eps^2M)={} over {} candidates".format(rho, len(corr_candidates)))

    regret = hardware_regret_analysis(pareto_candidates, x_no, si_metrics, args.weight_step)
    print("")
    print("Measurement/two-qubit scalarized regret:")
    print("  weights evaluated={}".format(regret["weight_count"]))
    print("  mean regret={:.12g}".format(regret["mean_regret"]))
    print("  max regret={:.12g}".format(regret["max_regret"]))
    print("  count with x_w* != x_NO*={}".format(regret["different_best_count"]))
    if regret["worst"] is not None:
        print(
            "  worst weight=(w_M={:.3g}, w_2q={:.3g}), best id={}, regret={:.12g}".format(
                regret["worst"]["weight"][0],
                regret["worst"]["weight"][1],
                regret["worst"]["best_candidate_id"],
                regret["worst"]["regret"],
            )
        )

    subset_paths = save_diversity_subset_artifacts(args.func_name, artifact_dir, subsets)
    diversity = diversity_analyses(subsets)
    print("")
    print("Label-invariant co-membership diversity:")
    print("Color-vector distance:")
    print("  {}".format(COLOR_DISTANCE_LATEX))
    for result in diversity:
        print_diversity_result(result)
    print("Saved diversity subset files:")
    for selector, path in subset_paths.items():
        print("  {} -> {}".format(selector, path))

    failed_ics = [candidate for candidate in ics_candidates if candidate.ics_error is not None]
    if failed_ics:
        print("")
        print("ICS failures:")
        for candidate in failed_ics:
            print("  id={}: {}".format(candidate.candidate_id, candidate.ics_error))

    ics_results_path = save_ics_result_artifact(
        args.func_name,
        artifact_dir,
        ics_candidates,
        ics_candidate_selectors,
        ics_report_sets,
    )
    print("Saved ICS calculation results to {}".format(ics_results_path))

    if args.output_prefix:
        write_outputs(
            args.output_prefix,
            analysis_candidates,
            ics_candidate_ids=ics_candidate_ids,
            ics_candidate_selectors=ics_candidate_selectors,
        )


if __name__ == "__main__":
    main()
