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

fig_name = "LiH"

##################################
# Load graphs from file!! ##########
##################################

with open(fig_name + "_sampled_graphs.p", 'rb') as f:
    sampled_graphs = pickle.load(f)
#Sorting graphs.
print("Number of Graphs in file: {}".format(len(sampled_graphs)))
# Step 2: Evaluate rewards
points = []
for g in sampled_graphs:
    r1 = get_groups_measurement(g, fci_wfn, n_q)
    r2 = n_paulis-color_reward(g)
    points.append((r1, r2))

points = np.array(points)

# Step 3: Pareto front computation
def pareto_front(points):
    is_pareto = np.ones(points.shape[0], dtype=bool)
    for i, p in enumerate(points):
        if is_pareto[i]:
            # Check if any point dominates p
            is_dominated = np.any(
                np.all(points <= p, axis=1) & np.any(points < p, axis=1)
            )
            if is_dominated:
                is_pareto[i] = False
    return is_pareto

mask = pareto_front(points)
pareto_points = points[mask]

# Step 4: Plot
plt.scatter(points[:, 0], points[:, 1], alpha=0.5, label="Sampled graphs")
plt.scatter(pareto_points[:, 0], pareto_points[:, 1], color="red", label="Pareto front")

# Optionally connect Pareto front points in sorted order
pareto_sorted = pareto_points[np.argsort(pareto_points[:, 0])]
plt.plot(pareto_sorted[:, 0], pareto_sorted[:, 1], color="red")

plt.xlabel("M_reward")
plt.ylabel("Color_reward")
plt.legend()
plt.savefig("pareto_front.svg", format="svg", dpi=600)
plt.show()


