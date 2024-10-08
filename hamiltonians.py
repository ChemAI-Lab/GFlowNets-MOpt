from openfermion.transforms import *
from openfermion.ops import FermionOperator, QubitOperator, BinaryCode, BinaryPolynomial
import tequila as tq
from tequila.hamiltonian import QubitHamiltonian, paulis
from tequila.grouping.binary_rep import BinaryHamiltonian
from tequila.grouping import *
import argparse

def parser():
    parser = argparse.ArgumentParser(description="Parse function name from terminal")
    
    # Define the argument for function name
    parser.add_argument(
        "func_name",
        type=str,
        help="The name of the molecule to call (e.g., H2, H4, H6, LiH, BeH2, N2)"
    )

    args = parser.parse_args()
    
    # Get the function dynamically
    func = globals().get(args.func_name)

    return func

def H2():
    '''
    Return a test hamiltonian.
    '''
    trafo = "JordanWigner"
    mol = tq.chemistry.Molecule(
                            geometry="H 0.0 0.0 0.0 \n H 0.0 0.0 1.",
                            basis_set="sto3g",
                            transformation=trafo,
                            backend='pyscf'
                                 )
    H = mol.make_hamiltonian()
    Hq = H.to_openfermion()
    Hferm = reverse_jordan_wigner(Hq)
    return mol, H, Hferm, len(Hq.terms) - 1, Hq #Minus 1 since it always contain a constant term that we don't need to measure.

def LiH():
    '''
    Return a test hamiltonian.
    '''
    trafo = "JordanWigner"
    mol = tq.chemistry.Molecule(
                            geometry="Li 0.0 0.0 0.0 \n H 0.0 0.0 1.",
                            basis_set="sto3g",
                            transformation=trafo,
                            backend='pyscf'
                                 )
    H = mol.make_hamiltonian()
    Hq = H.to_openfermion()
    Hferm = reverse_jordan_wigner(Hq)
    return mol, H, Hferm, len(Hq.terms) - 1, Hq #Minus 1 since it always contain a constant term that we don't need to measure.

def N2():
    '''
    Return a test hamiltonian.
    '''
    trafo = "JordanWigner"
    mol = tq.chemistry.Molecule(
                            geometry="N 0.0 0.0 0.0 \n N 0.0 0.0 1.0",
                            basis_set="sto3g",
                            transformation=trafo,
                            backend='pyscf'
                                 )
    H = mol.make_hamiltonian()
    Hq = H.to_openfermion()
    Hferm = reverse_jordan_wigner(Hq)
    return mol, H, Hferm, len(Hq.terms) - 1, Hq #Minus 1 since it always contain a constant term that we don't need to measure.

def H4():
    '''
    Return a test hamiltonian.
    '''
    trafo = "JordanWigner"
    mol = tq.chemistry.Molecule(
                            geometry="H 0.0 0.0 0.0 \n H 0.0 0.0 1.0 \n H 0.0 0.0 2.0 \n H 0.0 0.0 3.0 ",
                            basis_set="sto3g",
                            transformation=trafo,
                            backend='pyscf'
                                 )
    H = mol.make_hamiltonian()
    Hq = H.to_openfermion()
    Hferm = reverse_jordan_wigner(Hq)
    return mol, H, Hferm, len(Hq.terms) - 1, Hq #Minus 1 since it always contain a constant term that we don't need to measure.

def H6():
    '''
    Return a test hamiltonian.
    '''
    trafo = "JordanWigner"
    mol = tq.chemistry.Molecule(
                            geometry="H 0.0 0.0 0.0 \n H 0.0 0.0 1.0 \n H 0.0 0.0 2.0 \n H 0.0 0.0 3.0 \n H 0.0 0.0 4.0 \n H 0.0 0.0 5.0",
                            basis_set="sto3g",
                            transformation=trafo,
                            backend='pyscf'
                                 )
    H = mol.make_hamiltonian()
    Hq = H.to_openfermion()
    Hferm = reverse_jordan_wigner(Hq)
    return mol, H, Hferm, len(Hq.terms) - 1, Hq #Minus 1 since it always contain a constant term that we don't need to measure.

def BeH2():
    '''
    Return a test hamiltonian.
    '''
    trafo = "JordanWigner"
    mol = tq.chemistry.Molecule(
                            geometry="Be 0.0 0.0 0.0 \n H 0.0 0.0 1.0 \n H 0.0 0.0 -1.0 ",
                            basis_set="sto3g",
                            transformation=trafo,
                            backend='pyscf'
                                 )
    H = mol.make_hamiltonian()
    Hq = H.to_openfermion()
    Hferm = reverse_jordan_wigner(Hq)
    return mol, H, Hferm, len(Hq.terms) - 1, Hq #Minus 1 since it always contain a constant term that we don't need to measure.

