Before you run make sure to have the proper packages installed in either requirements.txt or copy and paste the commands in pip_installs.txt.


The purpose of this code is to generate a grouping of a Hamiltonian of molecular systems based on the estimated number of shots.
GFlowNet generates different samples where the ones with higher probability are based on the reward function.
Reward functions can be color_reward based only on the number of colors or vqe_reward which contains the estimated number of measurements.

To call this function use python driver.py molecule
Where molecule can be H2, H4, H6, LiH, BeH2, N2. The default bond distance is 1 Angstrom. This can be modified on the hamiltonians.py file.

On driver.py we can change parameters for GFlowNets like training rate, number of hid_uinits, number of episodes, update_freq and the random seed. 

Training models (Flow matching, trajectory balance etc) are in training.py

If the flow model architecture needs to be modified, this can be done through gflow_utils.py.

Pending implementations!!
Trajectory balance!!
To do:
Clean the requirements.txt
