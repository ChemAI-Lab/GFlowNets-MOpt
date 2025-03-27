The purpose of this code is to generate groups of compatible operators of a Hamiltonian of molecular systems based on the estimated number of shots.
GFlowNet generates different samples where the ones with higher probability are based on a reward function.

Before you run make sure to have the proper packages installed in requirements.txt 
or
Copy and paste the commands in pip_installs.txt ensuring that there are no conflicts during installation (additional packages may be required depending on your system).

Reward functions can be
1) color_reward based only on the number of colors 
2) vqe_reward which contains the estimated number of measurements.
3) meas_reward which in addition uses exact variances to get the measurement number.

A mask function located in gflow_vqe/gflow_utils.py is employed to ensure that the generated graphs are valid and to limit the solution space, this limit can be changed by the user as required by modifying the line "mask[lower_bound+1:] = 0". 

To generate commutativity graphs of the best performing groupings, modify the driver.py file calling 
check_sampled_graphs_method_plot instead of check_sampled_graphs_method. Lines are commented for user convenience. For method=fci, full ci variances are employed while the vqe option uses only an estimator for them.

NN parameters and optimizer state are saved for each molecule on the .pth file.
All sampled graphs are saved as fig_name_sampled_graphs.p for data analysis or posterior use in quantum computing software.

To call this function use:

python driver.py molecule > out.log

Where molecule can be H2, H4, H6, LiH, BeH2, N2. The default bond distance is 1 Angstrom, this can be modified on the gflow_vqe/hamiltonians.py file. 

On driver.py we can change parameters for GFlowNets like:
fig_name, Training rate, number of hid_uinits, number of episodes, update_freq and the random seed.

Using driver_loaded_hams.py we can load hamiltonians from the Hamiltonian library used in npj Quantum Inf 9, 14 (2023). https://doi.org/10.1038/s41534-023-00683-y for direct comparison. Note: Depending on the geometry at which the Hamiltonian was generated, the number of terms may change. A clear example is NH3 where the Hamiltonian included in hamiltonians.py produces 2617 terms instead of 3608 from the loaded one despite having the same bond distances and angles.

Training models (Flow matching and trajectory balance) are in gflow_vqe/training.py. 
We have 3 training protocols:
Pure -> Start with a graph with the worst possible coloring.
Colored_initial -> Start with a colored graph always.
Precolored -> Every epoch give a new randomly colored graph. 

Pending implementations!!

1) Add reduced parent calculation since we are using sequential coloring. Multicoloring or group coloring.
2) Measurement requirements using approximate Variances from ccsd/cisd
3) Parallelization of training to improve performance.
