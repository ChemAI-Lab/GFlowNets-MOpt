import torch
import torch.multiprocessing as mp
from tqdm import tqdm, trange
from gflow_vqe.utils import *
from gflow_vqe.gflow_utils import *
from torch.distributions.categorical import Categorical
from gflow_vqe.hamiltonians import *

def train_episode(rank, graph, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed, wfn, n_q):
    """Training function for each process."""
    torch.cuda.set_device(0)  # Use GPU 0 for all processes
    torch.cuda.set_per_process_memory_fraction(0.9 / mp.cpu_count(), device=0)  # Limit memory usage

    # Set the seed for reproducibility
    set_seed(seed + rank)

    # Create the model and move it to the GPU
    model = TBModel(n_hid_units, n_terms).to(0)

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    minibatch_loss = 0
    losses = []
    tbar = trange(n_episodes // mp.cpu_count(), desc=f"Process {rank} Training")
    color_map = nx.coloring.greedy_color(graph, strategy="random_sequential")
    bound=max(color_map.values())+2
    for episode in tbar:
        state = graph.copy()  
        P_F_s, P_B_s = model(graph_to_tensor(state).to(0), n_terms)  # Forward and backward policy
        total_log_P_F, total_log_P_B = 0, 0

        for t in range(nx.number_of_nodes(state)):
            new_state = state.copy()
            mask = calculate_forward_mask_from_state(new_state, t, bound).to(0)
            P_F_s = torch.where(mask, P_F_s, -100)  # Removes invalid forward actions.
            categorical = Categorical(logits=P_F_s)
            action = categorical.sample()
            new_state.nodes[t]['color'] = action.item()
            total_log_P_F += categorical.log_prob(action)

            if t == nx.number_of_nodes(state) - 1:  # End of trajectory
                reward = meas_reward(new_state, wfn, n_q)
            else:
                reward = 0

            P_F_s, P_B_s = model(graph_to_tensor(new_state).to(0), n_terms)
            mask = calculate_backward_mask_from_state(new_state, t, bound).to(0)
            P_B_s = torch.where(mask, P_B_s, -100)
            total_log_P_B += Categorical(logits=P_B_s).log_prob(action)

            state = new_state

        minibatch_loss += trajectory_balance_loss(
            model.logZ,
            total_log_P_F,
            total_log_P_B,
            reward,
        )

        if episode % update_freq == 0:
            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(minibatch_loss.item())
            minibatch_loss = 0

    return losses

def main(molecule):
    num_processes = mp.cpu_count()  # Number of processes to spawn
    print(num_processes)
   
    mol, H, Hferm, n_paulis, Hq = molecule()
    print("Number of Pauli products to measure: {}".format(n_paulis))
    ############################
    # Get FCI wfn for variance #
    ############################

    sparse_hamiltonian = get_sparse_operator(Hq)
    energy, fci_wfn = get_ground_state(sparse_hamiltonian)
    n_q = count_qubits(Hq)
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
    fig_name = "LiH"

    print("For all experiments, our hyperparameters will be:")
    print("    + n_hid_units={}".format(n_hid_units))
    print("    + n_episodes={}".format(n_episodes))
    print("    + learning_rate={}".format(learning_rate))
    print("    + update_freq={}".format(update_freq))
    print("Training in {} processors".format(num_processes))

    mp.spawn(
        train_episode,
        args=(Gc, n_terms, n_hid_units, n_episodes, learning_rate, update_freq, seed, fci_wfn, n_q),
        nprocs=num_processes,
        join=True,
    )

molecule = parser()
if __name__ == "__main__":
    main(molecule)