from gflow_vqe.utils import *
from gflow_vqe.hamiltonians import *
from gflow_vqe.gflow_utils import *
from gflow_vqe.result_analysis import *
from gflow_vqe.training import *
from openfermion import commutator

molecule = parser()
mol, H, Hferm, n_paulis, Hq = molecule()
# For loaded Hams
#mol="lih"
#Hq, H = load_qubit_hamiltonian(mol)
#print("Number of Pauli products to measure: {}".format(len(Hq.terms) - 1))

sparse_hamiltonian = get_sparse_operator(Hq)
energy, psi = get_ground_state(sparse_hamiltonian)
n_q = count_qubits(Hq)
binary_H = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
options = {"method":"si", "condition": "fc"}
commuting_parts = binary_H.commuting_groups(options=options)[0]
groups=[]
for i in range(len(commuting_parts)):
    commuting_parts[i]=commuting_parts[i].to_qubit_hamiltonian()
    groups.append(commuting_parts[i].to_openfermion())

tiny=1e-12
sqrt_var=0
for g in groups:
    sparse_group=get_sparse_operator(g,n_qubits=n_q)
    var=variance(sparse_group,psi)

    if var.imag < tiny:
         var = var.real

    if var.real < tiny:
        var=0
    sqrt_var+=math.sqrt(var)

eps_sq_M=sqrt_var**2
print("\eps^2 M={} FC".format(eps_sq_M))
print("Number of groups FC",len(groups))

options = {"method":"si", "condition": "qwc"}
commuting_parts = binary_H.commuting_groups(options=options)[0]
groups=[]
for i in range(len(commuting_parts)):
    commuting_parts[i]=commuting_parts[i].to_qubit_hamiltonian()
    groups.append(commuting_parts[i].to_openfermion())

tiny=1e-12
sqrt_var=0
for g in groups:
    sparse_group=get_sparse_operator(g,n_qubits=n_q)
    var=variance(sparse_group,psi)

    if var.imag < tiny:
         var = var.real

    if var.real < tiny:
        var=0
    sqrt_var+=math.sqrt(var)

eps_sq_M=sqrt_var**2
print("\eps^2 M={} QWC".format(eps_sq_M))
print("Number of groups QWC",len(groups))

