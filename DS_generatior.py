#Creation of electronic Hamiltonians of LiH, H4 (rectangle) and H4(line)

#Atomic distances:
    #LiH and H4(line) [0.5A, 3.3A]
    #H4(rectangle) [0.5A, 2.0A]x[0.5A, 2.0A]

#Use PySCF and OpenFermion
import numpy as np
from openfermion.chem import MolecularData as md
from openfermionpyscf import run_pyscf
from openfermion.hamiltonians import generate_hamiltonian
from openfermion.transforms import jordan_wigner

#Perform Hartree-Fock calculation, STO-3G minimal basis
#Construct fermionic sec-quantized Hamiltonian for all molecules and config
#Hamiltonian mapped to sum of Pauli operators by Jordan-Wigner transformation H(r)
#Electric Hamiltonians for all molecules turn into 8 qubit hamiltonians

bas = 'sto-3g'

#LiH 
LiH_ham = []
LiH_dis = np.linspace(0.5, 3.3, 100)
for d in LiH_dis:
    molecule = md(
        geometry = [['Li', (0, 0, 0)], ['H', (0, 0, d)]], 
        basis = bas,
        multiplicity = 1
    )

    mol = run_pyscf(molecule, run_scf=1, run_mp2=0, run_cisd=0, run_ccsd=0, run_fci=0)

    reduced_one_body = mol.one_body_integrals[:4, :4]
    reduced_two_body = mol.two_body_integrals[:4, :4, :4, :4]
    
    hamiltonian = generate_hamiltonian(
        one_body_integrals=reduced_one_body,
        two_body_integrals=reduced_two_body,
        constant=mol.nuclear_repulsion
    )

    LiH_ham.append(hamiltonian)

#H4 lin
H4lin_ham = []
H4lin_dis = np.linspace(0.5, 3.3, 100)
for d in H4lin_dis:
    molecule = md(
        geometry = [['H', (0, 0, 0)], ['H', (0, 0, d)], ['H', (0, 0, 2*d)], ['H', (0, 0, 3*d)]], 
        basis = bas,
        multiplicity = 1
    )

    mol = run_pyscf(molecule, run_scf=1, run_mp2=0, run_cisd=0, run_ccsd=0, run_fci=0)
    
    hamiltonian = generate_hamiltonian(
        one_body_integrals=mol.one_body_integrals,
        two_body_integrals=mol.two_body_integrals,
        constant=mol.nuclear_repulsion
    )

    H4lin_ham.append(hamiltonian)

#H4 rec
H4rec_ham = []
H4rec_dis = np.linspace(0.5, 2.0, 10)
for d1 in H4rec_dis:
    for d2 in H4rec_dis:
        molecule = md(
            geometry = [['H', (0, 0, 0)], ['H', (d1, 0, 0)], ['H', (0, d2, 0)], ['H', (d1, d2, 0)]], 
            basis = bas,
            multiplicity = 1
        )

        mol = run_pyscf(molecule, run_scf=1, run_mp2=0, run_cisd=0, run_ccsd=0, run_fci=0)
        
        hamiltonian = generate_hamiltonian(
            one_body_integrals=mol.one_body_integrals,
            two_body_integrals=mol.two_body_integrals,
            constant=mol.nuclear_repulsion
        )

        H4rec_ham.append(hamiltonian)

#Jordan-Wigner transformation
for i in range(len(LiH_ham)):
    LiH_ham[i] = jordan_wigner(LiH_ham[i])

for i in range(len(H4lin_ham)):
    H4lin_ham[i] = jordan_wigner(H4lin_ham[i])
    
for i in range(len(H4rec_ham)):
    H4rec_ham[i] = jordan_wigner(H4rec_ham[i])    

#For noiseless simulation, GS of H(r) is prepared by exact diagonalization
from openfermion.linalg import get_sparse_operator
from scipy.sparse.linalg import eigsh

LiH_eigenvalues = []
LiH_eigenvectors = []

#LiH



#Compute quantities of excited state properties to be predicted (formula 2)

#Diagonalization of H(r) 

#Fit values into [-1, 1] range

#30 training data points and 50 test points for LiH

#30 training data points and 50 test points for H4 (linear)

#250 training data points and 1250 test points for H4 (rectangle)



