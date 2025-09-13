The purpose of this code is to generate groupings of molecular Hamiltonians for their implementation in Quantum Computers based on the number of measurements required to reach an accuracy $\varepsilon$ and the number of groups.
GFlowNet generates different samples where the ones with higher probability are based on a reward function.

After cloning, install the package as:
pip install -e .

Before you run make sure to have the proper packages installed in requirements.txt 
or
For most users installing: matplotlib, seaborn, pennylane, tequila-basic, pyscf, torch and torch_geometric would be enough.

Reward functions can be
1) ==color_reward== based only on the number of colors.
2) ==vqe_reward== which contains the estimated number of measurements, used in https://arxiv.org/abs/2410.16041
3) ==meas_reward== which uses exact variances to get the measurement number and $\lambda_0=1$, $\lambda_1=1$ in the reward.
4) ==my_reward== which uses exact variances to get the measurement number and a $\lambda_0=10^3$ value
5) ==custom_reward== which uses exact variances to get the measurement number and the user can pass the $\lambda_0$, $\lambda_1$ values.

Verify before running the reward function employed by the training protocol. The trianing functions are inside gflow_vqe/training.py. The list is available in the driver.py file with a small description of the models employed for each of them and the loss function implemented.

For the results of the paper "Discrete Flow-Based Generative Models for Measurement Optimization in Quantum Computing", we employed the training functions:
==GIN_TB_training== corresponding to the GINE model described in the text.
==coeff_GIN_TB_training== corresponding to the GINE$_w$ model described in the text.
==coeff_GIN_TB_training_custom_reward==
All of them using the ==my_reward== or ==custom_reward== Reward functions. The upper bound for the search space is generated through a greedy coloring algorithm with a random sequential strategy and increased +2

A mask function located in gflow_vqe/gflow_utils.py is employed to ensure that the generated graphs are valid and to limit the solution space, this limit can be changed by the user as required by employing the ==coeff_GIN_TB_training_wbound== 

To generate commutativity graphs of the best performing groupings, modify the driver.py file calling 
check_sampled_graphs_method_plot instead of check_sampled_graphs_method. Lines are commented for user convenience. For method=fci, full ci variances are employed while the vqe option uses only an estimator for them.

NN parameters and optimizer state are saved for each molecule on the .pth file.
All sampled graphs are saved as fig_name_sampled_graphs.p for data analysis or posterior use in quantum computing software.

To call this function use:

python driver.py molecule > out.log

Where molecule can be H2, H4, LiH, BeH2, H2O, N2. The default bond distance is 1 Angstrom. This can be modified on the gflow_vqe/hamiltonians.py file. 

On driver.py we can change parameters for GFlowNets like:
fig_name, Training rate, number of hid_uinits, number of episodes, embedding dimension, update_freq and the random seed. We leave options for GPU usage although we saw no real benefit. 

Experimental! Parallel training implemented, we have multiple models (1/process) in para driver and single-model versions where the updates occur on each processor or by collecting the results and updating outside the sampling parallel loop 