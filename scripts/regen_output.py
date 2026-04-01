from gflow_vqe.utils import *
from gflow_vqe.hamiltonians import *
from gflow_vqe.gflow_utils import *
from gflow_vqe.result_analysis import *
import os
import pickle


def main():
    ########################
    # Hamiltonian definition
    # and initialization
    ########################
    driver_args = parse_driver_args()
    molecule = driver_args.func
    if molecule is None:
        raise ValueError("Unknown molecule '{}'".format(driver_args.func_name))

    reward_wfn_method = driver_args.wfn
    print("Training reward wavefunction={}".format(reward_wfn_method))

    mol, H, Hferm, n_paulis, Hq = molecule()
    print("Number of Pauli products to measure: {}".format(n_paulis))

    ############################
    # Get wavefunctions for reward + FCI analysis
    ############################
    sparse_hamiltonian = get_sparse_operator(Hq)
    fci_energy, fci_wfn = get_variance_wavefunction(
        mol,
        Hq,
        method="FCI",
        sparse_hamiltonian=sparse_hamiltonian,
    )
    print("FCI Energy={}".format(fci_energy))

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
        print("{} Energy={}".format(reward_wfn_method, reward_wfn_energy))

    n_q = count_qubits(Hq)
    print("Number of Qubits={}".format(n_q))

    fig_name = driver_args.func_name
    l0 = 0
    l1 = 1
    l2 = 0

    ##################################
    # Load graphs from file
    ##################################
    sampled_graphs_path = fig_name + "_sampled_graphs.p"
    if not os.path.isfile(sampled_graphs_path):
        raise FileNotFoundError(
            "Sampled graphs file '{}' not found. Run driver_sv.py first.".format(sampled_graphs_path)
        )

    with open(sampled_graphs_path, "rb") as f:
        sampled_graphs = pickle.load(f)

    if not isinstance(sampled_graphs, list):
        sampled_graphs = list(sampled_graphs)

    print("Number of Graphs in file: {}".format(len(sampled_graphs)))

    print("Sampled graphs saved to file: {}".format(sampled_graphs_path))
    ##################################################################################
    ## Done with the training loop, now we can analyze results.#######################
    ##################################################################################
    check_sampled_graphs_wf_plot(fig_name, sampled_graphs, reward_wfn, fci_wfn, n_q, l0=l0, l1=l1, l2=l2)
    histogram_all_fci(fig_name, sampled_graphs, fci_wfn, n_q)


if __name__ == "__main__":
    main()
