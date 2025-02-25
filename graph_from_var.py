from gflow_vqe.utils import *
from gflow_vqe.hamiltonians import *
from gflow_vqe.gflow_utils import *
from gflow_vqe.result_analysis import *
from gflow_vqe.training import *

fig_name = "Batch_Results"

with open(fig_name + "_variances.p", 'rb') as f:
    all_measurements = pickle.load(f)

num_blocks = len(all_measurements) // 10  # Number of full blocks

averages = np.zeros(num_blocks)
std_devs = np.zeros(num_blocks)
best_measurements = []  # Stores the best 10 measurements seen so far

for block_idx in range(num_blocks):
    # Extract block of 10 measurements
    block_measurements = all_measurements[block_idx * 10 : (block_idx + 1) * 10]

    # Update the list of best 10 measurements seen so far
    best_measurements.extend(block_measurements)
    best_measurements = sorted(best_measurements)[:10]  # Keep only the 10 lowest values

    # Compute cumulative average and standard deviation
    averages[block_idx] = np.mean(best_measurements)
    std_devs[block_idx] = np.std(best_measurements)

# Plotting with error bars
plt.figure(figsize=(12, 8))
x_values = np.arange(num_blocks)  # Block indices
yerr=std_devs
#plt.tight_layout()
#plt.errorbar(x_values, averages, yerr, fmt='o', capsize=5, label="Measurement Average")
lower_bound = averages - std_devs
upper_bound = averages + std_devs

plt.plot(x_values, averages, '#9467bd' ,marker='d', linestyle='-', label="Measurement Average", ms=8, markevery=3) #ms=5 for o and 8 for p,d,etc
plt.fill_between(x_values, lower_bound, upper_bound, color='#9467bd', alpha=0.3, label="Standard Deviation")
# Add horizontal black line

y_val=18.8 #GMA Value 
plt.axhline(y=y_val, color='black', linestyle='--', linewidth=3)
plt.xlim(-1, 49)
#plt.ylim(0.95,1.85)
plt.xlabel("Iterations",fontsize=20)
#plt.ylabel("Measurement Average",fontsize=20)
plt.ylabel(r"${\cal M} \;\;\; [10^{6}]$",fontsize=28)
plt.tick_params(axis='both', which='major', labelsize=20)  # Adjust label size
#plt.title("Average of Lowest 10 Measurements per Optimization Step")
#plt.legend()
plt.grid(False)
plt.savefig("Average_top10_H2OjwQWC.svg", format='svg', dpi=600)
plt.savefig("Average_top10_H2OjwQWC.png", format='png', dpi=600)

##################
#Color Codes and Markers
# *  H2: '#1f77b4'  "o"
# *  H4: '#ff7f0e'  "^"
# * LiH: '#2ca02c'  "s"
# * BeH2: '#d62728' "p"
# * H2O: '#9467bd'  "d"
# * N2: '#8c564b'   "h"
# * NH3: '#e377c2'  "X"