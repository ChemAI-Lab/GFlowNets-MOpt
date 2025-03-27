from gflow_vqe.utils import *
from gflow_vqe.hamiltonians import *
from gflow_vqe.gflow_utils import *
from gflow_vqe.result_analysis import *
from gflow_vqe.training import *
from openfermion import commutator
import multiprocessing


assert torch.__version__.startswith('2.1') and 'cu121' in torch.__version__, "The Colab torch version has changed, you may need to edit the !pip install cell to install matching torch_geometric versions"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
# Load graphs to file!! ##########
##################################

with open(fig_name + "_sampled_graphs.p", 'rb') as f:
    sampled_graphs = pickle.load(f)
#Sorting graphs.

print("Number of Graphs in file: {}".format(len(sampled_graphs)))
unique_graphs=[]
seen_color_dicts = set()
##################################################################################
## Verifying the code works correctly.############################################
##################################################################################
def check_commutativity(operator):
    terms = list(operator.terms.keys())  # Extract Pauli words
    num_terms = len(terms)

    commuting_pairs = []
    non_commuting_pairs = []

    for i in range(num_terms):
        for j in range(i + 1, num_terms):  # Avoid redundant checks
            term1 = QubitOperator(terms[i])
            term2 = QubitOperator(terms[j])

            if commutator(term1, term2) == QubitOperator():
                commuting_pairs.append((terms[i], terms[j]))
            else:
                non_commuting_pairs.append((terms[i], terms[j]))

    return commuting_pairs, non_commuting_pairs

# for graph in sorted(sampled_graphs, key=lambda i: get_groups_measurement(i, fci_wfn, n_q), reverse=False):
#     color_dict = frozenset(nx.get_node_attributes(graph, "color").items())
#     if color_dict not in seen_color_dicts:
#         seen_color_dicts.add(color_dict)
#         unique_graphs.append(graph)

# tiny = 1e-14

# h_color=extract_hamiltonian_by_color(unique_graphs[0])
# groups = generate_groups(h_color)
# sqrt_var=0
# for g in groups:
#     sparse_group=get_sparse_operator(g,n_qubits=n_q)
#     var=variance(sparse_group,fci_wfn)
#     #print("Group={}".format(g))
#     #print(type(g.terms))
#     commuting, non_commuting = check_commutativity(g)
#     print("Non-commuting pairs:", non_commuting)
#     print("Var_group={}".format(var))
#     if var.imag < tiny:
#         var = var.real

#     if var.real < tiny:
#         var=0

#     sqrt_var+=math.sqrt(var)

# eps_sq_M=sqrt_var**2
#print("esp^2M={}".format(eps_sq_M))

# num_blocks = len(sampled_graphs) // 100  # Number of full blocks

# averages = []
# std_devs = []

# for block_idx in range(num_blocks):
#     block_measurements = []
    
#     for i in range(block_idx * 100, (block_idx + 1) * 100):
#         block_measurements.append(get_groups_measurement(sampled_graphs[i], fci_wfn, n_q))

#     # Select the 10 graphs with the lowest measurements
#     lowest_measurements = sorted(block_measurements)[:10]

#     # Compute average and standard deviation
#     avg = np.mean(lowest_measurements)
#     std_dev = np.std(lowest_measurements)

#     averages.append(avg)
#     std_devs.append(std_dev)

#all_measurements = np.array([get_groups_measurement(g, fci_wfn, n_q) for g in sampled_graphs])
def parallel_get_measurements(sampled_graphs, fci_wfn, n_q):
    with multiprocessing.Pool() as pool:
        results = pool.starmap(get_groups_measurement, [(g, fci_wfn, n_q) for g in sampled_graphs])
    return np.array(results)

all_measurements = parallel_get_measurements(sampled_graphs, fci_wfn, n_q)

with open(fig_name + "_variances.p", 'wb') as f:
    pickle.dump(all_measurements, f, pickle.HIGHEST_PROTOCOL)

print("All Measurement produced")
num_blocks = len(sampled_graphs) // 10  # Number of full blocks

averages = np.zeros(num_blocks)
std_devs = np.zeros(num_blocks)

#This section takes the average per block
# for block_idx in range(num_blocks):
#     # Extract block of 100 measurements
#     block_measurements = all_measurements[block_idx * 100 : (block_idx + 1) * 100]

#     # Get the 5 smallest values efficiently
#     lowest_measurements = np.partition(block_measurements, 5)[:5]  
#     print("Measurement block sorted")


#     # Compute average and standard deviation
#     averages[block_idx] = np.mean(lowest_measurements)
#     std_devs[block_idx] = np.std(lowest_measurements)

#This block now takes the cumulative average.

best_measurements = []  # Stores the best 10 measurements seen so far

for block_idx in range(num_blocks):
    # Extract block of 10 measurements
    block_measurements = all_measurements[block_idx * 10 : (block_idx + 1) * 10]

    # Update the list of best 10 measurements seen so far
    best_measurements.extend(block_measurements)
    best_measurements = sorted(best_measurements)[:10]  # Keep only the 10 lowest values

    print("Updated best 10 measurements up to this block")

    # Compute cumulative average and standard deviation
    averages[block_idx] = np.mean(best_measurements)
    std_devs[block_idx] = np.std(best_measurements)

# Plotting with error bars
plt.figure(figsize=(12, 8))
x_values = np.arange(num_blocks)  # Block indices
yerr=std_devs

#plt.errorbar(x_values, averages, yerr, fmt='o', capsize=5, label="Measurement Average")
lower_bound = averages - std_devs
upper_bound = averages + std_devs

plt.tight_layout()
plt.plot(x_values, averages, 'o-', label="Measurement Average", ms=4, markevery=3)
plt.fill_between(x_values, lower_bound, upper_bound, alpha=0.3, label="Standard Deviation")
# Add horizontal black line
plt.axhline(y=1.11, color='black', linestyle='--', linewidth=3)
plt.xlim(0, 300)
plt.xlabel("Iterations",fontsize=20)
plt.ylabel("Measurement Average",fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=14)  # Adjust label size
#plt.title("Average of Lowest 10 Measurements per Optimization Step")
#plt.legend()
plt.grid(False)
plt.savefig("Average_top10_BeH2_fb.svg", format='svg', dpi=600)
plt.savefig("Average_top10_BeH2_fb.png", format='png', dpi=600)


##################################################################################
## Done with the training loop, now we can analyze results.#######################
##################################################################################
#check_sampled_graphs_vqe_plot(fig_name, sampled_graphs) #Prints commutativity graphs for best performing groupings
#check_sampled_graphs_vqe(sampled_graphs)
#check_sampled_graphs_fci(sampled_graphs, fci_wfn, n_q)
#check_sampled_graphs_fci_plot(fig_name, sampled_graphs, fci_wfn, n_q)
#histogram_all_fci(fig_name,sampled_graphs,fci_wfn,n_q)
