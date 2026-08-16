"""Print fully commuting graph sizes and random-sequential color bounds."""

import math

import networkx as nx
from tequila.grouping.binary_rep import BinaryHamiltonian

from gflow_vqe.hamiltonians import BeH2, H2, H2O, H4, LiH, N2, SiO
from gflow_vqe.utils import (
    FC_CompMatrix,
    get_terms,
    obj_to_comp_graph,
)


MOLECULES = (
    ("H2", H2),
    ("H4", H4),
    ("LiH", LiH),
    ("BeH2", BeH2),
    ("H2O", H2O),
    ("N2", N2),
    ("SiO", SiO),
)
N_RANDOM_COLORINGS = 10


def calculate_color_bound(molecule_name, molecule_builder):
    """Build one FC incompatibility graph and print its requested bound."""

    _, H, _, _, _ = molecule_builder()
    binary_H = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
    terms = get_terms(binary_H)
    CompMatrix = FC_CompMatrix(terms)
    Gc = obj_to_comp_graph(terms, CompMatrix)
    n_terms = nx.number_of_nodes(Gc)
    print(
        "Number of terms in the Hamiltonian: {}".format(n_terms),
        flush=True,
    )

    bounds = []
    for _ in range(N_RANDOM_COLORINGS):
        color_map = nx.coloring.greedy_color(
            Gc,
            strategy="random_sequential",
        )
        n_colors = len(set(color_map.values()))
        bounds.append(n_colors + 2)

    average_bound = math.ceil(sum(bounds) / len(bounds))
    print("Individual color bounds for {}: {}".format(molecule_name, bounds))
    return n_terms, bounds, average_bound


def main():
    average_bounds = []
    for molecule_name, molecule_builder in MOLECULES:
        _, _, average_bound = calculate_color_bound(
            molecule_name,
            molecule_builder,
        )
        average_bounds.append((molecule_name, average_bound))

    print(
        "\nCeiling-rounded average color bounds over {} random_sequential "
        "colorings:".format(N_RANDOM_COLORINGS)
    )
    for molecule_name, average_bound in average_bounds:
        print("{}: {}".format(molecule_name, average_bound))


if __name__ == "__main__":
    main()
