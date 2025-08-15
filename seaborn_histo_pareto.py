from gflow_vqe.utils import *
from gflow_vqe.hamiltonians import *
from gflow_vqe.gflow_utils import *
from gflow_vqe.result_analysis import *
from gflow_vqe.training import *
from openfermion import commutator
import multiprocessing
import seaborn as sns
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
#sampled_graphs = sampled_graphs[4000:]
print("Number of Graphs in file: {}".format(len(sampled_graphs)))
# Step 2: Evaluate rewards
points = []
for g in sampled_graphs:
    r1 = get_groups_measurement(g, fci_wfn, n_q)
    r2 =n_paulis-color_reward(g)
    points.append((r1, r2))

points = np.array(points)
x = points[:, 0]
y = points[:, 1]

# 3) Pareto front (minimization)
def pareto_front_min(pts):
    is_pareto = np.ones(pts.shape[0], dtype=bool)
    for i, p in enumerate(pts):
        if not is_pareto[i]:
            continue
        dominated = np.any(np.all(pts <= p, axis=1) & np.any(pts < p, axis=1))
        if dominated:
            is_pareto[i] = False
    return is_pareto

mask = pareto_front_min(points)
pareto_sorted = points[mask][np.argsort(points[mask][:, 0])]

# 4) Seaborn JointGrid: scatter + KDE marginals
sns.set_theme(style="whitegrid")
g = sns.JointGrid(x=x, y=y, height=7.5, space=0)

# Joint scatter
sns.scatterplot(x=x, y=y, ax=g.ax_joint, alpha=0.5, s=30, edgecolor=None)

# Pareto front line in RED with markers
g.ax_joint.plot(
    pareto_sorted[:, 0], pareto_sorted[:, 1],
    color="orange", marker="o", markersize=8, linewidth=1.5, label="Pareto front"
) #color = "red", "#FF8C00"
g.ax_joint.legend(loc="best")
g.set_axis_labels("$R_G$", "$N_P-R_G$")

# KDE marginals (filled)
sns.kdeplot(x=x, ax=g.ax_marg_x, fill=True,color="purple")
sns.kdeplot(y=y, ax=g.ax_marg_y, fill=True,color="green")

# Optional: cleaner marginal axes
g.ax_marg_x.set_ylabel("")
g.ax_marg_y.set_xlabel("")
g.ax_marg_x.tick_params(axis="x", labelbottom=False)
g.ax_marg_y.tick_params(axis="y", labelleft=False)

g.ax_joint.plot(
    0.276, 20, 
    marker="d", color="red", markersize=5, linestyle="None", label="SI"
)

# Add text label near the point
g.ax_joint.text(
    0.276 + 0.005, 20 + 0.3,  # small offset so it doesn't overlap
    "SI", color="red", fontsize=9
)
# 5) Save SVG at 600 dpi
g.figure.savefig("pareto_joint_all.svg", format="svg", dpi=600, bbox_inches="tight")
