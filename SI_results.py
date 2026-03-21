from gflow_vqe.utils import *
from gflow_vqe.hamiltonians import *
from gflow_vqe.gflow_utils import *
from gflow_vqe.result_analysis import *
from gflow_vqe.training import *
from gflow_vqe.overlapping_helpers import as_tequila_wavefunction, prepare_cov_dict
from openfermion import commutator
from tequila.grouping.binary_rep import BinaryHamiltonian


def _to_real_if_close(value, tiny=1e-12):
    if hasattr(value, "imag") and abs(value.imag) < tiny:
        return float(value.real)
    return value


def build_openfermion_groups(commuting_parts):
    groups = []
    for part in commuting_parts:
        groups.append(part.to_qubit_hamiltonian().to_openfermion())
    return groups


def equal_allocation_metric(groups, wfn, n_qubits, tiny=1e-12):
    sqrt_var = 0
    for group in groups:
        sparse_group = get_sparse_operator(group, n_qubits=n_qubits)
        var = variance(sparse_group, wfn)
        if var.imag < tiny:
            var = var.real
        if var.real < tiny:
            var = 0
        sqrt_var += math.sqrt(var)
    return sqrt_var ** 2


def optimal_allocation_metric(commuting_parts, suggested_sample_size, fci_wfn, tiny=1e-12):
    measurement_metric = 0
    wf_fci = as_tequila_wavefunction(fci_wfn)

    for idx, part in enumerate(commuting_parts):
        op = part.to_qubit_hamiltonian()
        var_part = wf_fci.inner((op * op)(wf_fci)) - wf_fci.inner(op(wf_fci)) ** 2
        if hasattr(var_part, "imag") and abs(var_part.imag) < tiny:
            var_part = var_part.real
        measurement_metric += var_part / suggested_sample_size[idx]

    return _to_real_if_close(measurement_metric, tiny=tiny)


def print_equal_allocation_result(label, binary_hamiltonian, n_qubits, fci_wfn, method="si", condition=None):
    if condition is None:
        condition = label.lower()
    options = {"method": method, "condition": condition}
    commuting_parts = binary_hamiltonian.commuting_groups(options=options)[0]
    groups = build_openfermion_groups(commuting_parts)
    eps_sq_m = equal_allocation_metric(groups, fci_wfn, n_qubits)
    print("eps^2 M={} {}".format(_to_real_if_close(eps_sq_m), label))
    print("Number of groups {} {}".format(label, len(groups)))


def print_optimal_allocation_result(label, options, binary_hamiltonian, fci_wfn):
    commuting_parts, suggested_sample_size = binary_hamiltonian.commuting_groups(options=options)
    measurement_metric = optimal_allocation_metric(commuting_parts, suggested_sample_size, fci_wfn)
    print("Required number of measurements={} {}".format(measurement_metric, label))
    print("Number of groups {} {}".format(label, len(commuting_parts)))


def main():
    ##############Block for normal Hamiltonians###########################
    args = parse_driver_args()
    molecule = args.func
    if molecule is None:
        raise ValueError("Unknown molecule '{}'".format(args.func_name))

    mol, H, Hferm, n_paulis, Hq = molecule()
    print("Number of Pauli products to measure: {}".format(n_paulis))
    ######################################################################
    ##############Block for loaded Hamiltonians###########################
    #This driver takes Hamiltonians from npj Quantum Inf 9, 14 (2023). https://doi.org/10.1038/s41534-023-00683-y
    # MOLECULES = ["h2", "lih", "beh2", "h2o", "nh3", "n2"]
    # mol="lih"
    # Hq, H = load_qubit_hamiltonian(mol)
    # print("Number of Pauli products to measure: {}".format(len(Hq.terms) - 1))
    ######################################################################
    sparse_hamiltonian = get_sparse_operator(Hq)
    _, fci_wfn = get_ground_state(sparse_hamiltonian)
    n_q = count_qubits(Hq)
    binary_hamiltonian = BinaryHamiltonian.init_from_qubit_hamiltonian(H)

    print_equal_allocation_result("FC", binary_hamiltonian, n_q, fci_wfn)
    print_equal_allocation_result("QWC", binary_hamiltonian, n_q, fci_wfn)
    print_equal_allocation_result("RLF", binary_hamiltonian, n_q, fci_wfn, method="rlf", condition="fc")

    cov_dict = prepare_cov_dict(binary_hamiltonian, fci_wfn)

    options_si_optimal = {"method": "si", "condition": "fc", "cov_dict": cov_dict}
    print_optimal_allocation_result(
        "SI-Optimal Measurement allocation",
        options_si_optimal,
        binary_hamiltonian,
        fci_wfn,
    )

    options_ics = {"method": "ics", "condition": "fc", "cov_dict": cov_dict}
    print_optimal_allocation_result("ICS", options_ics, binary_hamiltonian, fci_wfn)


if __name__ == "__main__":
    main()
