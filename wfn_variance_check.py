from gflow_vqe.utils import *
from gflow_vqe.hamiltonians import *
from gflow_vqe.gflow_utils import *


def _to_real_if_close(value, tiny=1e-10):
    if hasattr(value, "imag") and abs(value.imag) < tiny:
        return float(value.real)
    return value


def main():
    args = parse_driver_args()
    molecule = args.func
    if molecule is None:
        raise ValueError("Unknown molecule '{}'".format(args.func_name))

    reward_wfn_method = args.wfn
    print("Molecule={}".format(args.func_name))
    print("Wavefunction for variance check={}".format(reward_wfn_method))

    mol, H, Hferm, n_paulis, Hq = molecule()
    print("Number of Pauli products to measure: {}".format(n_paulis))

    sparse_hamiltonian = get_sparse_operator(Hq)
    fci_energy, fci_wfn = get_variance_wavefunction(mol, Hq, method="FCI", sparse_hamiltonian=sparse_hamiltonian)
    if reward_wfn_method == "FCI":
        reward_wfn = fci_wfn
        reward_wfn_energy = fci_energy
    else:
        reward_wfn_energy, reward_wfn = get_variance_wavefunction(
            mol,
            Hq,
            method=reward_wfn_method,
            sparse_hamiltonian=sparse_hamiltonian,
        )

    print("FCI Energy={}".format(_to_real_if_close(fci_energy)))
    if reward_wfn_method != "FCI":
        print("{} Energy={}".format(reward_wfn_method, _to_real_if_close(reward_wfn_energy)))

    n_q = count_qubits(Hq)
    print("Number of qubits={}".format(n_q))

    binary_H = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
    terms = get_terms(binary_H)
    comp_matrix = FC_CompMatrix(terms)
    Gc = obj_to_comp_graph(terms, comp_matrix)
    n_terms = nx.number_of_nodes(Gc)
    print("Number of Hamiltonian terms={}".format(n_terms))

    # Deterministic valid coloring for variance check.
    graph = Gc.copy()
    color_map = nx.coloring.greedy_color(graph, strategy="largest_first")
    nx.set_node_attributes(graph, color_map, "color")
    print("Greedy coloring max color={}".format(max_color(graph)))

    eps2m_reward = get_groups_measurement(graph, reward_wfn, n_q)
    eps2m_fci = get_groups_measurement(graph, fci_wfn, n_q)
    print("eps^2 M using {} = {}".format(reward_wfn_method, eps2m_reward))
    print("eps^2 M using FCI = {}".format(eps2m_fci))

    # Sanity-check reward helpers and get_groups_measurement usage.
    print("meas_reward({}) = {}".format(reward_wfn_method, meas_reward(graph, reward_wfn, n_q)))
    print("meas_reward(FCI) = {}".format(meas_reward(graph, fci_wfn, n_q)))
    print("custom_reward({}) [l0=1,l1=1] = {}".format(reward_wfn_method, custom_reward(graph, reward_wfn, n_q, 1, 1)))

    h_color = extract_hamiltonian_by_color(graph)
    groups = generate_groups(h_color)
    print("Number of measurement groups={}".format(len(groups)))
    print("First grouped variances (selected wfn vs FCI):")
    for i, group in enumerate(groups[: min(5, len(groups))]):
        sparse_group = get_sparse_operator(group, n_qubits=n_q)
        var_reward = _to_real_if_close(variance(sparse_group, reward_wfn))
        var_fci = _to_real_if_close(variance(sparse_group, fci_wfn))
        print("  Group {}: {} variance={}, FCI variance={}".format(i, reward_wfn_method, var_reward, var_fci))

    print("Variance check completed.")


if __name__ == "__main__":
    main()
