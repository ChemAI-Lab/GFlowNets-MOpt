#from tqdm import tqdm, trange
#from torch.distributions.categorical import Categorical
from utils import *
from hamiltonians import *
from gflow_utils import *
from result_analysis import *
from training import *

assert torch.__version__.startswith('2.1') and 'cu121' in torch.__version__, "The Colab torch version has changed, you may need to edit the !pip install cell to install matching torch_geometric versions"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
########################
#Hamiltonian definition#
# and initialization   #
########################
molecule = parser()
mol, H, Hferm, n_paulis, Hq = molecule()
print("Number of Pauli products to measure: {}".format(n_paulis))

#Get list of Hamiltonian terms and generate complementary graph
binary_H = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
terms=get_terms(binary_H)
CompMatrix=FC_CompMatrix(terms)
Gc=obj_to_comp_graph(terms, CompMatrix)
n_terms=nx.number_of_nodes(Gc)
###########################
# Parameters for GFlowNets#
###########################

n_hid_units = 128
n_episodes = 1000
learning_rate = 3e-4
update_freq = 10
seed = 45

print("For all experiments, our hyperparameters will be:")
print("    + n_hid_units={}".format(n_hid_units))
print("    + n_episodes={}".format(n_episodes))
print("    + learning_rate={}".format(learning_rate))
print("    + update_freq={}".format(update_freq))


##################################
# Training Loop!! ################
##################################

sampled_graphs, losses = precolored_flow_match_training(Gc, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed)

##################################################################################
## Done with the training loop, now we can analyze results.#######################
##################################################################################
check_sampled_graphs_vqe_plot(sampled_graphs)
plot_loss_curve(losses, title="Loss over Training Iterations")
#histogram_last(sampled_graphs)
histogram_all(sampled_graphs)
#plt.show()
