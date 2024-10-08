Before you run make sure to have the proper packages installed.
!pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install botorch gpytorch typeguard==2.13.3 linear_operator==0.5.2 jaxtyping pyro-ppl --no-deps
!pip install torch_geometric==2.4.0 torch_sparse torch_scatter torch-cluster rdkit gitpython omegaconf wandb --find-links https://data.pyg.org/whl/torch-2.1.2+cu121.html
!pip install --no-deps git+https://github.com/recursionpharma/gflownet.git@f106cde
!pip install git+https://github.com/quantumlib/OpenFermion.git@master#egg=openfermion
!pip install --upgrade git+https://github.com/aspuru-guzik-group/tequila.git@devel
!pip install qulacs
!pip install PySCF
!pip install pennylane

The purpose of this code is to generate a grouping of a Hamiltonian of molecular systems based on the estimated number of shots.
GFlowNet generates different samples where the ones with higher probability are based on the reward function.
Reward functions can be color_reward based only on the number of colors or vqe_reward which contains the estimated number of measurements.

To call this function use python driver.py molecule
Where molecule can be H2, H4, H6, LiH, BeH2, N2. The default bond distance is 1 Angstrom. This can be modified on the hamiltonians.py file.

On driver.py we can change parameters for GFlowNets like training rate, number of hid_uinits, number of episodes and the random seed. 

If the flow model architecture needs to be modified, this can be done through gflow_utils.py.

Pending implementations!!
Trajectory balance!!

