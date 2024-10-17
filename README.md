The purpose of this code is to generate groups of compatible operators of a Hamiltonian of molecular systems based on the estimated number of shots.
GFlowNet generates different samples where the ones with higher probability are based on a reward function.

Before you run make sure to have the proper packages installed in requirements.txt 
or
Copy and paste the commands in pip_installs.txt ensuring that there are no conflicts during installation (additional packages may be required depending on your system).

Reward functions can be color_reward based only on the number of colors or vqe_reward which contains the estimated number of measurements.
A mask function located in gflow_vqe/gflow_utils.py is employed to ensure that the generated graphs are valid and to limit the solution space, this limit can be changed by the user as required by modifying the line "mask[lower_bound+1:] = 0". 

To generate commutativity graphs of the best performing groupings, modify the driver.py file calling 
check_sampled_graphs_vqe_plot instead of check_sampled_graphs_vqe. Lines are commented for user convenience.

To call this function use:

python driver.py molecule > out.log

Where molecule can be H2, H4, H6, LiH, BeH2, N2. The default bond distance is 1 Angstrom, this can be modified on the gflow_vqe/hamiltonians.py file. 

On driver.py we can change parameters for GFlowNets like:
fig_name, Training rate, number of hid_uinits, number of episodes, update_freq and the random seed, 

Training models (Flow matching, trajectory balance etc) are in gflow_vqe/training.py. 
Currently only the flow matching training is available, trajectory balance will be implemented for the full journal version.

If the flow model architecture needs to be modified, this can be done in gflow_vqe/gflow_utils.py.

Pending implementations!!
1) Use a .json file as input for driver.py
2) Measurement requirements using Variances from ccsd/cisd/fci wavefunctions.
3) Saving to files the generated dictionaries (save the best performing groupings only or save all sampled but ordered)
4) Trajectory balance!!
