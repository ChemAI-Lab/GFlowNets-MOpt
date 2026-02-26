import argparse
import math

from gflow_vqe.hamiltonians import *


def _get_molecule_builder(name):
    fn = globals().get(name)
    if fn is None or not callable(fn):
        raise ValueError("Unknown molecule '{}'".format(name))
    return fn


def _non_identity_pauli_words(qubit_hamiltonian):
    """Return the set of Pauli words (OpenFermion term keys) excluding the identity term."""
    return {term for term in qubit_hamiltonian.terms.keys() if term != ()}


def _non_identity_pauli_coeffs(qubit_hamiltonian):
    """Return a dict term -> coefficient excluding the identity term."""
    return {term: coeff for term, coeff in qubit_hamiltonian.terms.items() if term != ()}


def _load_pauli_words(molecule_name):
    builder = _get_molecule_builder(molecule_name)
    _, _, _, _, hq = builder()
    return _non_identity_pauli_words(hq)


def _load_pauli_coeffs(molecule_name):
    builder = _get_molecule_builder(molecule_name)
    _, _, _, _, hq = builder()
    return _non_identity_pauli_coeffs(hq)


def main():
    parser = argparse.ArgumentParser(
        description="Compare common Pauli words between two molecular qubit Hamiltonians."
    )
    parser.add_argument("molecule_a", type=str, help="First molecule function name (e.g., BeH2)")
    parser.add_argument("molecule_b", type=str, help="Second molecule function name (e.g., H4)")
    args = parser.parse_args()

    coeffs_a = _load_pauli_coeffs(args.molecule_a)
    coeffs_b = _load_pauli_coeffs(args.molecule_b)
    paulis_a = set(coeffs_a.keys())
    paulis_b = set(coeffs_b.keys())

    n_a = len(paulis_a)
    n_b = len(paulis_b)
    common = paulis_a & paulis_b
    n_common = len(common)
    shared_terms_sorted = sorted(common, key=repr)

    # Euclidean (L2) distance between coefficient vectors restricted to shared Pauli words.
    # Works for real or complex coefficients.
    coeff_distance_l2 = math.sqrt(
        sum(abs(coeffs_a[t] - coeffs_b[t]) ** 2 for t in shared_terms_sorted)
    )
    coeff_abs_diffs = [abs(coeffs_a[t] - coeffs_b[t]) for t in shared_terms_sorted]
    min_coeff_diff = min(coeff_abs_diffs) if coeff_abs_diffs else 0.0
    max_coeff_diff = max(coeff_abs_diffs) if coeff_abs_diffs else 0.0
    avg_coeff_diff = (sum(coeff_abs_diffs) / len(coeff_abs_diffs)) if coeff_abs_diffs else 0.0

    if n_a <= n_b:
        smaller_name, smaller_n = args.molecule_a, n_a
        bigger_name, bigger_n = args.molecule_b, n_b
    else:
        smaller_name, smaller_n = args.molecule_b, n_b
        bigger_name, bigger_n = args.molecule_a, n_a

    overlap_prop = (n_common / smaller_n) if smaller_n > 0 else 0.0

    print("Pauli-word overlap (non-identity terms, coefficients ignored)")
    print("{}: {} Pauli words".format(args.molecule_a, n_a))
    print("{}: {} Pauli words".format(args.molecule_b, n_b))
    print("Common Pauli words: {}".format(n_common))
    print(
        "Proportion of smaller Hamiltonian ({}) present in bigger ({}): {:.6f} ({:.2f}%)".format(
            smaller_name,
            bigger_name,
            overlap_prop,
            100.0 * overlap_prop,
        )
    )
    print(
        "Smaller/Bigger sizes: {}/{}".format(
            smaller_n,
            bigger_n,
        )
    )
    print(
        "Coefficient vector L2 distance on shared Pauli words (dimension {}): {}".format(
            n_common,
            coeff_distance_l2,
        )
    )
    print(
        "Min/Max absolute coefficient difference on shared Pauli words: {} / {}".format(
            min_coeff_diff,
            max_coeff_diff,
        )
    )
    print(
        "Average absolute coefficient difference on shared Pauli words: {}".format(
            avg_coeff_diff,
        )
    )


if __name__ == "__main__":
    main()
