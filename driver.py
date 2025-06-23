from gflow_vqe.utils import *
from gflow_vqe.hamiltonians import *
from gflow_vqe.gflow_utils import *
from gflow_vqe.result_analysis import *
from gflow_vqe.training import *
import time

assert torch.__version__.startswith('2.1') and 'cu121' in torch.__version__, "The Colab torch version has changed, you may need to edit the !pip install cell to install matching torch_geometric versions"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
########################
#Hamiltonian definition#
# and initialization   #
########################
molecule = parser()
t0 = time.time()
mol, H, Hferm, n_paulis, Hq = molecule()
print("Number of Pauli products to measure: {}".format(n_paulis))
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
###########################
# Parameters for GFlowNets#
###########################

n_hid_units = 512
n_episodes = 1000
learning_rate = 3e-4
update_freq = 10
seed = 45
fig_name = "H2"

print("For all experiments, our hyperparameters will be:")
print("    + n_hid_units={}".format(n_hid_units))
print("    + n_episodes={}".format(n_episodes))
print("    + learning_rate={}".format(learning_rate))
print("    + update_freq={}".format(update_freq))


##################################
# Training Loop!! ################
##################################

#sampled_graphs, losses = colored_initial_flow_match_training(Gc, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed, fci_wfn, n_q, fig_name)
#sampled_graphs, losses = precolored_flow_match_training(Gc, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed, fci_wfn, n_q, fig_name)
#sampled_graphs, losses = pure_flow_match_training(Gc, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed, fci_wfn, n_q, fig_name)
#sampled_graphs, losses = colored_initial_TB_training(Gc, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed, fci_wfn, n_q, fig_name)
#sampled_graphs, losses = precolored_TB_training(Gc, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed, fci_wfn, n_q, fig_name)
#sampled_graphs, losses = pure_TB_training(Gc, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed, fci_wfn, n_q, fig_name)
#sampled_graphs, losses = seq_TB_training(Gc, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed, fci_wfn, n_q, fig_name)
sampled_graphs, losses = GIN_TB_training(Gc, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed, fci_wfn, n_q, fig_name)

##################################
#Timing###########################
t1 = time.time()
print(f"Training time: {t1 - t0:.2f} seconds")
##################################
# Save graphs to file!! ##########
##################################

with open(fig_name + "_sampled_graphs.p", 'wb') as f:
    pickle.dump(sampled_graphs, f, pickle.HIGHEST_PROTOCOL)

##################################################################################
## Done with the training loop, now we can analyze results.#######################
##################################################################################
#check_sampled_graphs_vqe_plot(fig_name, sampled_graphs) #Prints commutativity graphs for best performing groupings
#check_sampled_graphs_vqe(sampled_graphs)
#check_sampled_graphs_fci(sampled_graphs, fci_wfn, n_q)
check_sampled_graphs_fci_plot(fig_name, sampled_graphs, fci_wfn, n_q)
plot_loss_curve(fig_name, losses, title="Loss over Training Iterations")
#histogram_last(sampled_graphs)
#histogram_all(fig_name,sampled_graphs)
histogram_all_fci(fig_name,sampled_graphs,fci_wfn,n_q)
