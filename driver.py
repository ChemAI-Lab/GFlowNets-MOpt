from tqdm import tqdm, trange
from torch.distributions.categorical import Categorical
from utils import *
from hamiltonians import *
from gflow_utils import *
from result_analysis import *

assert torch.__version__.startswith('2.1') and 'cu121' in torch.__version__, "The Colab torch version has changed, you may need to edit the !pip install cell to install matching torch_geometric versions"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
########################
#Hamiltonian definition#
# and initialization   #
########################

mol, H, Hferm, n_paulis, Hq = LiH()
print("Number of Pauli products to measure: {}".format(n_paulis))

#Get list of Hamiltonian terms
binary_H = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
terms=get_terms(binary_H)
CompMatrix=FC_CompMatrix(terms)
Gc=obj_to_comp_graph(terms, CompMatrix)
n_terms=nx.number_of_nodes(Gc)
###########################
# Parameters for GFlowNets#
###########################

n_hid_units = 512
n_episodes = 100
learning_rate = 3e-4
seed = 45

print("For all experiments, our hyperparameters will be:")
print("    + n_hid_units={}".format(n_hid_units))
print("    + n_episodes={}".format(n_episodes))
print("    + learning_rate={}".format(learning_rate))

##################################
# Training Loop!! ################
##################################
set_seed(seed)

# Instantiate model n_hid_units optimizer
F_sa = FlowModel(n_hid_units,n_terms)
opt = torch.optim.Adam(F_sa.parameters(), learning_rate)

# Accumulate losses here and take a
# gradient step every `update_freq` episode (at the end of each trajectory).
losses, sampled_graphs = [], []
minibatch_loss = 0
update_freq = 10
# Dictionary to store the color assigned to each node

tbar = trange(n_episodes, desc="Training iter")
for episode in tbar:
    state = Gc  # Each episode starts with the initially colored graph
    color_map = nx.coloring.greedy_color(state, strategy="random_sequential")
    nx.set_node_attributes(state, color_map, 'color')
    bound=max(color_map.values())
    edge_flow_preds = F_sa(graph_to_tensor(state))  # Predict F(s, a).
    #print(edge_flow_preds)
    for t in range(nx.number_of_nodes(state)):  # All trajectories as length the number of nodes

        #Mask calculator
        #print(t)
        new_state = state.copy()
        mask = calculate_forward_mask_from_state(new_state, t, bound)
        edge_flow_preds = edge_flow_preds * mask
        # Sample the action and compute the new state.
        policy = edge_flow_preds / edge_flow_preds.sum()
        # We want to sample from the policy here (a Categorical distribution...)
        action = Categorical(probs=policy).sample()
        #print('Action {}'.format(action))
        new_state.nodes[t]['color'] = action.item()
        #print(new_state.nodes[t]['color'])

        # To compute the loss, we'll first enumerate the parents, then compute
        # the edge flows F(s, a) of each parent, indexing to get relevant flows.
        parent_states, parent_actions = graph_parents(new_state)
        ps = torch.stack([graph_to_tensor(p) for p in parent_states])
        pa = torch.tensor(parent_actions).long()
        parent_edge_flow_preds = F_sa(ps)[torch.arange(len(parent_states)), pa]

        if t == nx.number_of_nodes(state)-1:  # End of trajectory.
            # We calculate the reward and set F(s,a) = 0 \forall a, since there
            # are no children of this state.
            #print(nx.get_node_attributes(new_state, "color"))
            #reward = color_reward(new_state)
            reward = vqe_reward(new_state)
            edge_flow_preds = torch.zeros(nx.number_of_nodes(state))
            #print(reward)
        else:
            # We compute F(s, a) and set the reward to zero.
            reward = 0
            edge_flow_preds = F_sa(graph_to_tensor(new_state))


        minibatch_loss += flow_matching_loss(  # Accumulate.
            parent_edge_flow_preds,
            edge_flow_preds,
            reward,
        )
        state = new_state  # Continue iterating.

    # We're done with the episode, add the graph to the list, and if we are at an
    # update episode, take a gradient step.
    sampled_graphs.append(state)
    if episode % update_freq == 0:

        # Normalize accumulated loss.
        minibatch_loss = minibatch_loss / (update_freq)
        losses.append(minibatch_loss.item())
        tbar.set_description("Training iter (reward={:.3f})".format(reward))
        minibatch_loss.backward()
        opt.step()
        opt.zero_grad()
        minibatch_loss = 0
##################################################################################
## Done with the training loop, now we can analyze results.#######################
##################################################################################
check_sampled_graphs_vqe_plot(sampled_graphs)
plot_loss_curve(losses, title="Loss over Training Iterations")
plt.show()
