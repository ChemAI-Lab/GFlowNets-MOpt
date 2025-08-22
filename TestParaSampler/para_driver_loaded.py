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


def train_episode(rank, graph, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed, wfn, n_q):
    """Training function for each process. Uses trajectory balance with sequential colors P_B=1"""
    #torch.cuda.set_device(0)  # Use GPU 0 for all processes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    #torch.cuda.set_per_process_memory_fraction(0.9 / mp.cpu_count(), device=0)  # Limit memory usage

    # Set the seed for reproducibility
    set_seed(seed + rank)

    # Create the model and move it to the GPU
    model = TBModel_seq(n_hid_units, n_terms).to(device)

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
        P_F_s = model(graph_to_tensor(state).to(device), n_terms)  # Forward and backward policy
        total_log_P_F= 0

        for t in range(nx.number_of_nodes(state)):
            new_state = state.copy()
            mask = calculate_forward_mask_from_state(new_state, t, bound).to(device)
            P_F_s = torch.where(mask, P_F_s, -100)  # Removes invalid forward actions.
            #P_F_s = torch.clamp(P_F_s, min=-1e6, max=1e6)
            P_F_s = torch.where(torch.isnan(P_F_s), torch.full_like(P_F_s, -100), P_F_s)
            # if torch.isnan(P_F_s).any():
            #     print(f"NaN detected in P_F_s at process {rank}, episode {episode}")
            #     print(f"P_F_s: {P_F_s}")
            #     raise ValueError("NaN detected in P_F_s")
            categorical = Categorical(logits=P_F_s)
            action = categorical.sample()
            new_state.nodes[t]['color'] = action.item()
            total_log_P_F += categorical.log_prob(action)

            if t == nx.number_of_nodes(state) - 1:  # End of trajectory
                reward = meas_reward(new_state, wfn, n_q)
            else:
                reward = 0

            P_F_s = model(graph_to_tensor(new_state).to(device), n_terms)
            #mask = calculate_backward_mask_from_state(new_state, t, bound).to(device)
            #P_B_s = torch.clamp(P_B_s, min=-1e6, max=1e6)
            #P_B_s = torch.where(torch.isnan(P_B_s), torch.full_like(P_B_s, -100), P_B_s)
            #P_B_s = torch.where(mask, P_B_s, -100)
            #total_log_P_B += Categorical(logits=P_B_s).log_prob(action)

            state = new_state

        sampled_graphs.append(state)
        minibatch_loss += trajectory_balance_loss_seq(
            model.logZ,
            total_log_P_F,
            reward,
        )

        if episode % update_freq == 0:
            losses.append(minibatch_loss.item())
            logZs.append(model.logZ.item())    
            minibatch_loss.backward()
            # Gradient clipping. Interesting to avoid exploding gradients. Good to know, not necessarily needed here
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            opt.zero_grad()
            minibatch_loss = 0

    return sampled_graphs, losses
    
def train_wrapper(rank, shared_results, *args):
    sampled_graphs, losses = train_episode(rank, *args)
    print(f"Process {rank} finished training.")
    shared_results.append((sampled_graphs, losses))

def main_loaded(Hq,H):
    num_processes = mp.cpu_count()  # Number of processes to spawn
    t0 = time.time()   
    print("Number of Pauli products to measure: {}".format(len(Hq.terms) - 1))    ############################
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
    n_episodes = num_processes*120
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
            train_wrapper,
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

#This driver takes Hamiltonians from npj Quantum Inf 9, 14 (2023). https://doi.org/10.1038/s41534-023-00683-y
# MOLECULES = ["h2", "lih", "beh2", "h2o", "nh3", "n2"]
if __name__ == "__main__":
    mol="lih"
    Hq, H = load_qubit_hamiltonian(mol)    
    main_loaded(Hq, H)