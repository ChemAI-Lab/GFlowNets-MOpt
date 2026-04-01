import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle

from gflow_vqe.utils import *
from gflow_vqe.hamiltonians import *
from gflow_vqe.gflow_utils import *
from gflow_vqe.result_analysis import *
from gflow_vqe.training import *
from openfermion import commutator

########################
# Hamiltonian + data
########################
molecule = parser()
mol, H, Hferm, n_paulis, Hq = molecule()
print("Number of Pauli products to measure: {}".format(n_paulis))

sparse_hamiltonian = get_sparse_operator(Hq)
energy, fci_wfn = get_ground_state(sparse_hamiltonian)
n_q = count_qubits(Hq)

fig_name = "BeH2"
with open(fig_name + "_sampled_graphs.p", 'rb') as f:
    sampled_graphs = pickle.load(f)

#sampled_graphs = sampled_graphs[:50]

print("Number of Graphs in file: {}".format(len(sampled_graphs)))

# Compute points once
points = []
for g in sampled_graphs:
    r1 = get_groups_measurement(g, fci_wfn, n_q)
    r2 = n_paulis - color_reward(g)
    points.append((r1, r2))
points = np.array(points)
x_all = points[:, 0]
y_all = points[:, 1]

########################
# Pareto helper
########################
def pareto_front_min(pts):
    if len(pts) == 0:
        return np.array([], dtype=bool)
    is_pareto = np.ones(pts.shape[0], dtype=bool)
    for i, p in enumerate(pts):
        if not is_pareto[i]:
            continue
        dominated = np.any(np.all(pts <= p, axis=1) & np.any(pts < p, axis=1))
        if dominated:
            is_pareto[i] = False
    return is_pareto

# ---- Build the JointGrid ONCE ----
sns.set_theme(style="whitegrid")
# seed with one point just to create the layout; we'll clear it anyway
g = sns.JointGrid(x=x_all[:1], y=y_all[:1], height=7.5, space=0)
fig = g.figure  # <- the single figure we animate

# Fixed ranges & labels
#g.ax_joint.set_xlim(0.55, 1.8)
#g.ax_joint.set_ylim(9, 27)
#g.set_axis_labels(r"$\epsilon^2M(x)$", r"$N_P-R_G(x)$", fontsize=14)

# ---- Frame sizes (10, 20, ..., N) ----
STEP = 5
frame_sizes = list(range(STEP, len(x_all) + 1, STEP))
if frame_sizes[-1] != len(x_all):
    frame_sizes.append(len(x_all))

def update(frame_idx):
    # how many samples to show in this frame
    n = frame_sizes[frame_idx]
    x = x_all[:n]
    y = y_all[:n]
    pts = np.column_stack((x, y))

    # Clear axes (DON'T create new figures!)
    g.ax_joint.cla()
    g.ax_marg_x.cla()
    g.ax_marg_y.cla()

    g.set_axis_labels(r"$\epsilon^2M(x)$", r"$N_P-R_G(x)$", fontsize=14)

    # Joint scatter
    sns.scatterplot(x=x, y=y, ax=g.ax_joint, alpha=0.5, s=30, edgecolor=None)

    # Pareto front
    mask = pareto_front_min(pts)
    #pareto_sorted = points[mask][np.argsort(points[mask][:, 0])]    
    #g.ax_joint.plot(
    #pareto_sorted[:, 0], pareto_sorted[:, 1],
    #color="orange", marker="o", markersize=8, linewidth=1.5, label="Pareto front"
    #) #color = "red", "#FF8C00"
    pf = pts[mask]
    if len(pf) > 0:
        pf = pf[np.argsort(pf[:, 0])]
        g.ax_joint.plot(pf[:, 0], pf[:, 1], color="orange", marker="o",
                        markersize=6, linewidth=1.5, label="Pareto front")

    # Marginal KDEs
    #if len(x) > 1 and np.std(x) > 0:
    sns.kdeplot(x=x, ax=g.ax_marg_x, fill=True, color="purple", bw_adjust=0.8)
    #if len(y) > 1 and np.std(y) > 0:
    sns.kdeplot(y=y, ax=g.ax_marg_y, fill=True, color="green",  bw_adjust=0.8)
    
    # Optional: cleaner marginal axes
    g.ax_marg_x.set_ylabel("")
    g.ax_marg_y.set_xlabel("")
    g.ax_marg_x.tick_params(axis="x", labelbottom=False)
    g.ax_marg_y.tick_params(axis="y", labelleft=False)

    g.ax_joint.plot(
    0.614, 18,
    marker="D", color="red", markersize=7, linestyle="None", label="SI"
    )
    g.ax_joint.legend(loc="best")

    # Fixed axis ranges
    g.ax_joint.set_xlim(0.55, 1.8)
    g.ax_joint.set_ylim(9, 27)

    # Keep marginals aligned to joint limits
    g.ax_marg_x.set_xlim(g.ax_joint.get_xlim())
    g.ax_marg_y.set_ylim(g.ax_joint.get_ylim())

    # Clean marginal axes
    #g.ax_marg_x.set_ylabel("")
    #g.ax_marg_y.set_xlabel("")
    #g.ax_marg_x.tick_params(axis="x", labelbottom=False)
    #g.ax_marg_y.tick_params(axis="y", labelleft=False)
    # Keep marginals aligned to joint limits
    #g.ax_marg_x.set_xlim(g.ax_joint.get_xlim())
    #g.ax_marg_y.set_ylim(g.ax_joint.get_ylim())
    g.ax_marg_y.set_xticks([])   # removes numbers from top marginal
    g.ax_marg_x.set_yticks([])   # removes numbers from right marginal
    g.ax_marg_y.set_xticklabels([])   # removes numbers from top marginal
    g.ax_marg_x.set_yticklabels([])   # removes numbers from right marginal
    fig.subplots_adjust(bottom=0.10, left=0.10, right=0.98, top=0.98)

    return (g.ax_joint, g.ax_marg_x, g.ax_marg_y)

# Make the animation FAST (short delay) and use the single seaborn figure
ani = animation.FuncAnimation(
    fig, update, frames=len(frame_sizes), interval=30, blit=False, repeat=False
)

# Save GIF at higher fps for snappier playback
ani.save("pareto_joint_progress.gif", writer="pillow", dpi=300, fps=30)

# Optional: close the figure after saving
plt.close(fig)

