import torch
import torch.multiprocessing as mp
from tqdm import tqdm, trange
from gflow_vqe.utils import *
from gflow_vqe.gflow_utils import *
from torch.distributions.categorical import Categorical
from gflow_vqe.hamiltonians import *
from gflow_vqe.result_analysis import *
from multiprocessing import Manager
import time

def test_reward(graph, wfn, n_qubit):
    """Reward is based on the number of colors we have. The lower cliques the better.
    Invalid configs give 0. Additionally, employs 1/eps^2M where M is the number of Measurements
    to achieve accuracy \eps as reward function. The lower number of shots, the better."""
    if is_not_valid(graph):
        return 0
    else:
        reward= 1/get_groups_measurement(graph, wfn, n_qubit)#color_reward(graph) + 10**3/get_groups_measurement(graph, wfn, n_qubit)

    return reward

def train_episode(rank, graph, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed, wfn, n_q):
    """Training function for each process. Trajectory balance"""
    #torch.cuda.set_device(0)  # Use GPU 0 for all processes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    #torch.cuda.set_per_process_memory_fraction(0.9 / mp.cpu_count(), device=0)  # Limit memory usage

    # Set the seed for reproducibility
    set_seed(seed + rank)

    # Create the model and move it to the GPU
    model = TBModel(n_hid_units, n_terms).to(device)

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    minibatch_loss = 0
    losses, sampled_graphs, logZs = [], [], []
    tbar = trange(n_episodes // mp.cpu_count(), desc=f"Process {rank} Training")
    color_map = nx.coloring.greedy_color(graph, strategy="random_sequential")
    bound=max(color_map.values())+2
    for episode in tbar:
        state = graph.copy()  
        P_F_s, P_B_s = model(graph_to_tensor(state).to(device), n_terms)  # Forward and backward policy
        total_log_P_F, total_log_P_B = 0, 0

        for t in range(nx.number_of_nodes(state)):
            new_state = state.copy()
            mask = calculate_forward_mask_from_state(new_state, t, bound).to(device)
            P_F_s = torch.where(mask, P_F_s, -100)  # Removes invalid forward actions.
            categorical = Categorical(logits=P_F_s)
            action = categorical.sample()
            new_state.nodes[t]['color'] = action.item()
            total_log_P_F += categorical.log_prob(action)

            if t == nx.number_of_nodes(state) - 1:  # End of trajectory
                reward = test_reward(new_state, wfn, n_q)
            else:
                reward = 0

            P_F_s, P_B_s = model(graph_to_tensor(new_state).to(device), n_terms)
            mask = calculate_backward_mask_from_state(new_state, t, bound).to(device)
            #P_B_s = torch.clamp(P_B_s, min=-1e6, max=1e6)
            #P_B_s = torch.where(torch.isnan(P_B_s), torch.full_like(P_B_s, -100), P_B_s)
            P_B_s = torch.where(mask, P_B_s, -100)
            total_log_P_B += Categorical(logits=P_B_s).log_prob(action)

            state = new_state

        sampled_graphs.append(state)
        minibatch_loss += trajectory_balance_loss(
            model.logZ,
            total_log_P_F,
            total_log_P_B,
            reward,
        )

        if episode % update_freq == 0:
            logZs.append(model.logZ.item())
            minibatch_loss.backward()
            # Gradient clipping. Interesting to avoid exploding gradients. Good to know, not necessarily needed here
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            opt.zero_grad()
            losses.append(minibatch_loss.item())
            minibatch_loss = 0

    return sampled_graphs, losses
    
def train_wrapper(rank, shared_results, *args):
    sampled_graphs, losses = train_episode(rank, *args)
    print(f"Process {rank} finished training.")
    shared_results.append((sampled_graphs, losses))

def train_episode_seq(rank, graph, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed, wfn, n_q):
    """Training function for each process. Trajectory balance"""
    #torch.cuda.set_device(0)  # Use GPU 0 for all processes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    #torch.cuda.set_per_process_memory_fraction(0.9 / mp.cpu_count(), device=0)  # Limit memory usage

    # Set the seed for reproducibility
    set_seed(seed + rank)

    # Instantiate model and optimizer
    model = TBModel_seq(n_hid_units,n_terms)
    opt = torch.optim.Adam(model.parameters(),  learning_rate)

    # Accumulate losses here and take a
    # gradient step every `update_freq` episode (at the end of each trajectory).
    losses, sampled_graphs, logZs = [], [], []
    minibatch_loss = 0
    # Determine upper limit
    color_map = nx.coloring.greedy_color(graph, strategy="random_sequential")
    bound=max(color_map.values())+ math.floor(0.1*max(color_map.values())) #10

    tbar = trange(n_episodes, desc="Training iter")
    for episode in tbar:
        state = graph  # Each episode starts with the initially colored graph
        P_F_s = model(graph_to_tensor(state),n_terms)  # Forward and backward policy
        total_log_P_F = 0
        
        for t in range(nx.number_of_nodes(state)):  # All trajectories as length the number of nodes

            #Mask calculator
            new_state = state.copy()
            mask = calculate_forward_mask_from_state(new_state, t, bound)
            P_F_s = torch.where(mask, P_F_s, -100)  # Removes invalid forward actions.
            #P_F_s = torch.where(torch.isnan(P_F_s), torch.full_like(P_F_s, -100), P_F_s)
            # Sample the action and compute the new state.
            # Here P_F is logits, so we use Categorical to compute a softmax.
            categorical = Categorical(logits=P_F_s)
            action = categorical.sample()
            #print('Action {}'.format(action))
            new_state.nodes[t]['color'] = action.item()
            total_log_P_F += categorical.log_prob(action)  # Accumulate the log_P_F sum.

            #If a trajectory is complete. in TB we don't need to calculate parents.
            if t == nx.number_of_nodes(state)-1:  # End of trajectory.
            # We calculate the reward
                reward = meas_reward(new_state,wfn,n_q)
                #reward = vqe_reward(new_state)
            
            # We recompute P_F and P_B for new_state.
            P_F_s = model(graph_to_tensor(new_state),n_terms)

            state = new_state  # Continue iterating.

        # We're done with the trajectory, let's compute its loss. Since the reward
        # can sometimes be zero, instead of log(0) we'll clip the log-reward to -20.
        minibatch_loss += trajectory_balance_loss_seq(
            model.logZ,
            total_log_P_F,
            reward,
        )

    # We're done with the episode, add the graph to the list, and if we are at an
    # update episode, take a gradient step.
        sampled_graphs.append(state)
        if episode % update_freq == 0:
            losses.append(minibatch_loss.item())
            logZs.append(model.logZ.item())
            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            minibatch_loss = 0
            # torch.save({
            # 'epoch': episode,
            # 'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': opt.state_dict(),
            # 'loss': losses,
            # }, fig_name + "_pureTBmodel_seq.pth")

    return sampled_graphs, losses
    
def train_wrapper_seq(rank, shared_results, *args):
    sampled_graphs, losses = train_episode_seq(rank, *args)
    print(f"Process {rank} finished training.")
    shared_results.append((sampled_graphs, losses))

def main(molecule):
    num_processes = mp.cpu_count()  # Number of processes to spawn
    t0 = time.time()   
    mol, H, Hferm, n_paulis, Hq = molecule()
    print("Number of Pauli products to measure: {}".format(n_paulis))
    ############################
    # Get FCI wfn for variance #
    ############################

    sparse_hamiltonian = get_sparse_operator(Hq)
    energy, fci_wfn = get_ground_state(sparse_hamiltonian)
    n_q = count_qubits(Hq)
    print("Energy={}".format(energy))
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
    n_episodes = 1000 #num_processes*100
    learning_rate = 1e-3
    update_freq = 10
    seed = 45
    fig_name = "Test"

    print("For all experiments, our hyperparameters will be:")
    print("    + n_hid_units={}".format(n_hid_units))
    print("    + n_episodes={}".format(n_episodes))
    print("    + learning_rate={}".format(learning_rate))
    print("    + update_freq={}".format(update_freq))
    print("Training in {} processors".format(num_processes))
   # Use a Manager to create shared lists
    with Manager() as manager:
        shared_results = manager.list()

        mp.spawn(
            train_wrapper_seq, #train_wrapper or train_wrapper_seq
            args=(shared_results, Gc, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed, fci_wfn, n_q),
            nprocs=num_processes,
            join=True,
        )

        # Aggregate results from all processes
        all_sampled_graphs = []
        all_losses = []
        for sampled_graphs, losses in shared_results:
            all_sampled_graphs.extend(sampled_graphs)
            all_losses.extend(losses)
    t1 = time.time()
    print(f"Training time: {t1 - t0:.2f} seconds")
    
    print("Training completed successfully")
    # Save all sampled graphs to a file
    with open(fig_name + "_sampled_graphs.p", "wb") as f:
        pickle.dump(all_sampled_graphs, f, pickle.HIGHEST_PROTOCOL)

    print(f"All sampled graphs saved to {fig_name}_sampled_graphs.p")

    # Perform analysis similar to driver.py
    check_sampled_graphs_fci_plot(fig_name, all_sampled_graphs, fci_wfn, n_q)
    plot_loss_curve(fig_name, all_losses, title="Loss over Training Iterations")
    histogram_all_fci(fig_name, all_sampled_graphs, fci_wfn, n_q)

molecule = parser()
if __name__ == "__main__":
    main(molecule)