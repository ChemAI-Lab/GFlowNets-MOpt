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

def train_episode(rank, model, graph, n_terms, update_freq, seed, wfn, n_q):
    """Training function for each process. Uses trajectory balance with sequential colors P_B=1"""
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    set_seed(seed + rank)

    # Move the model to the appropriate device
    model.to(device)

    minibatch_loss = 0
    losses, sampled_graphs, logZs = [], [], []
    #tbar = trange(update_freq, desc=f"Process {rank} Training")  # Run only update_freq episodes
    #color_map = nx.coloring.greedy_color(graph, strategy="random_sequential")
    color_map = nx.coloring.greedy_color(graph, strategy="largest_first")

    bound = max(color_map.values()) + 2

    #for episode in tbar:
    for _ in range(update_freq):
        state = graph.copy()
        P_F_s, P_B_s = model(graph_to_tensor(state),n_terms)
        # Forward and backward policy
        total_log_P_F, total_log_P_B = 0, 0


        for t in range(nx.number_of_nodes(state)):
            new_state = state.copy()
            mask = calculate_forward_mask_from_state(new_state, t, bound).to(device)
            P_F_s = torch.where(mask, P_F_s, -100)  # Removes invalid forward actions.
            #P_F_s = torch.where(torch.isnan(P_F_s), torch.full_like(P_F_s, -100), P_F_s)
            categorical = Categorical(logits=P_F_s)
            action = categorical.sample()
            new_state.nodes[t]['color'] = action.item()
            total_log_P_F += categorical.log_prob(action)

            if t == nx.number_of_nodes(state) - 1:  # End of trajectory
                reward = test_reward(new_state, wfn, n_q)
            else:
                reward = 0

            P_F_s, P_B_s = model(graph_to_tensor(new_state),n_terms)
            mask = calculate_backward_mask_from_state(new_state, t, bound).to(device)
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

    losses.append(minibatch_loss.item())
    return sampled_graphs, losses


def train_wrapper(rank, shared_results, model, graph, n_terms, update_freq, seed, wfn, n_q):
    sampled_graphs, losses = train_episode(rank, model, graph, n_terms, update_freq, seed, wfn, n_q)
    #print(f"Process {rank} finished training.")
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

    binary_H = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
    terms = get_terms(binary_H)
    CompMatrix = FC_CompMatrix(terms)
    Gc = obj_to_comp_graph(terms, CompMatrix)
    n_terms = nx.number_of_nodes(Gc)

    n_hid_units = 512
    n_episodes = num_processes * 10
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
        all_sampled_graphs = []
        all_losses = []

        for iteration in range(n_episodes // update_freq):

            # Create the model and optimizer
            #model = TBModel_seq(n_hid_units, n_terms)
            model = embTBModel(n_hid_units, n_terms)
            model.share_memory()  # Share the model's parameters across processes
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            # Spawn processes for training
            mp.spawn(
                train_wrapper,
                args=(shared_results, model, Gc, n_terms, update_freq, seed, fci_wfn, n_q),
                nprocs=num_processes,
                join=True,
            )

            # Aggregate losses from all processes
            total_loss = 0
            total_samples = 0
            for sampled_graphs, losses in shared_results:
                all_sampled_graphs.extend(sampled_graphs)
                all_losses.extend(losses)
                total_loss += sum(losses)
                total_samples += len(sampled_graphs)

            # Perform optimization in the main process
            total_loss_tensor = torch.tensor(total_loss, requires_grad=True)
            total_loss_tensor.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Print optimization status
            print(f"Iteration {iteration + 1}: Optimized parameters with {total_samples} samples.")

            # Clear shared results for the next round
            shared_results[:] = []

    t1 = time.time()
    print(f"Training time: {t1 - t0:.2f} seconds")

    print("Training completed successfully")
    # Save all sampled graphs to a file
    with open(fig_name + "_sampled_graphs.p", "wb") as f:
        pickle.dump(all_sampled_graphs, f, pickle.HIGHEST_PROTOCOL)

    print(f"Total number of sampled graphs: {len(all_sampled_graphs)}")
    print(f"All sampled graphs saved to {fig_name}_sampled_graphs.p")

    # Perform analysis similar to driver.py
    check_sampled_graphs_fci_plot(fig_name, all_sampled_graphs, fci_wfn, n_q)
    plot_loss_curve(fig_name, all_losses, title="Loss over Training Iterations")
    histogram_all_fci(fig_name, all_sampled_graphs, fci_wfn, n_q)


molecule = parser()
if __name__ == "__main__":
    main(molecule)
