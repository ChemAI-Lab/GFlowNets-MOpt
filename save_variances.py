from gflow_vqe.utils import *
from gflow_vqe.hamiltonians import *
from gflow_vqe.gflow_utils import *
from gflow_vqe.result_analysis import *
from gflow_vqe.training import *
from openfermion import commutator
import multiprocessing

########################
#Hamiltonian definition#
# and initialization   #
########################
molecule = parser()
mol, H, Hferm, n_paulis, Hq = molecule()
print("Number of Pauli products to measure: {}".format(n_paulis)) 
############################
# Get FCI wfn for variance #
############################

sparse_hamiltonian = get_sparse_operator(Hq)
energy, fci_wfn = get_ground_state(sparse_hamiltonian)
n_q = count_qubits(Hq)

fig_name = "Batch_Results"

##################################
# Load graphs from file!! ##########
##################################

with open(fig_name + "_sampled_graphs.p", 'rb') as f:
    sampled_graphs = pickle.load(f)
#Sorting graphs.

print("Number of Graphs in file: {}".format(len(sampled_graphs)))
#Serial version
all_measurements = np.array([get_groups_measurement(g, fci_wfn, n_q) for g in sampled_graphs])
# Parallel version
# def parallel_get_measurements(sampled_graphs, fci_wfn, n_q):
#     with multiprocessing.Pool() as pool:
#         results = pool.starmap(get_groups_measurement, [(g, fci_wfn, n_q) for g in sampled_graphs])
#     return np.array(results)

#all_measurements = parallel_get_measurements(sampled_graphs, fci_wfn, n_q)

print("All Measurement produced")

with open(fig_name + "_variances.p", 'wb') as f:
    pickle.dump(all_measurements, f, pickle.HIGHEST_PROTOCOL)

print("All Measurement saved")


