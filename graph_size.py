from gflow_vqe.utils import *
from gflow_vqe.hamiltonians import *
from gflow_vqe.gflow_utils import *
from gflow_vqe.result_analysis import *
from gflow_vqe.training import *
import time

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
########################
#Hamiltonian definition#
# and initialization   #
########################
molecule = parser()
mol, H, Hferm, n_paulis, Hq = molecule()
print("Number of Pauli products to measure: {}".format(n_paulis))
###############
# For loaded Hams
#mol="lih"
#Hq, H = load_qubit_hamiltonian(mol)
#print("Number of Pauli products to measure: {}".format(len(Hq.terms) - 1))

############################
# Get FCI wfn for variance #
############################

sparse_hamiltonian = get_sparse_operator(Hq)
energy, fci_wfn = get_ground_state(sparse_hamiltonian)
print("Energy={}".format(energy))
n_q = count_qubits(Hq)
print("Number of Qubits={}".format(n_q))
#Get list of Hamiltonian terms and generate complementary graph
binary_H = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
terms=get_terms(binary_H)
CompMatrix=FC_CompMatrix(terms)
#CompMatrix=QWC_CompMatrix(terms)
Gc=obj_to_comp_graph(terms, CompMatrix)
n_terms=nx.number_of_nodes(Gc)
print("Results for FC grouping")
print("Number of terms in the Hamiltonian: {}".format(n_terms))
print(Gc)

# Average number of neighbors (average degree)
avg_neighbors = sum(dict(Gc.degree()).values()) / Gc.number_of_nodes()

print("Average number of neighbors per node: {:.2f}".format(avg_neighbors))

CompMatrix_qwc=QWC_CompMatrix(terms)
Gc_qwc=obj_to_comp_graph(terms, CompMatrix_qwc)
n_terms=nx.number_of_nodes(Gc)
print("Results for QWC grouping")
print("Number of terms in the Hamiltonian: {}".format(n_terms))
print(Gc_qwc)

# Average number of neighbors (average degree)
avg_neighbors = sum(dict(Gc_qwc.degree()).values()) / Gc_qwc.number_of_nodes()

print("Average number of neighbors per node: {:.2f}".format(avg_neighbors))

