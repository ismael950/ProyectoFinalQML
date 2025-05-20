#Creation of electronic Hamiltonians of LiH, H4 (rectangle) and H4(line)

#Atomic distances:
    #LiH [0.5A, 3.3A]

#Use PySCF and OpenFermion
import numpy as np
import pyscf
from pyscf.scf.dhf import dip_moment
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
LiH_molecules = []

for d in LiH_dis:
    molecule = md(
        geometry = [['Li', (0, 0, 0)], ['H', (0, 0, d)]], 
        basis = bas,
        multiplicity = 1
    )

    mol = run_pyscf(molecule, run_scf=1, run_mp2=0, run_cisd=0, run_ccsd=0, run_fci=0)
    LiH_molecules.append(mol)

    reduced_one_body = mol.one_body_integrals[:4, :4]
    reduced_two_body = mol.two_body_integrals[:4, :4, :4, :4]
    
    hamiltonian = generate_hamiltonian(
        one_body_integrals=reduced_one_body,
        two_body_integrals=reduced_two_body,
        constant=mol.nuclear_repulsion
    )

    LiH_ham.append(hamiltonian)

for i in range(len(LiH_ham)):
    LiH_ham[i] = jordan_wigner(LiH_ham[i])

from openfermion.linalg import get_sparse_operator
from scipy.sparse.linalg import eigsh

#LiH
LiH_eigenvalues = []
LiH_eigenvectors = []

for H in LiH_ham:
    SH = get_sparse_operator(H)
    evals, evcts = eigsh(SH, k=3, which = 'SA')
    LiH_eigenvalues.append(evals)
    GS=[]
    for i in evcts:
        GS.append(i[0])
    LiH_eigenvectors.append(GS)


#Compute quantities of excited state properties to be predicted (formula 2) deltaE1 and deltaE2
deltas = []
d1min = 1000000000
d1max = -1000000000
d2min = 1000000000
d2max = -1000000000

for eivls in LiH_eigenvalues:
    d1 = eivls[1]-eivls[0]
    d2 = eivls[2]-eivls[0]
    d1min = min(d1min, d1)
    d1max = max(d1max, d1)
    d2min = min(d2min, d2)
    d2max = max(d2max, d2)
    deltas.append([d1, d2])

#Fit values into [-1, 1] range
for i in range(len(deltas)):
    deltas[i][0] = 2*(deltas[i][0]-d1min)/(d1max-d1min)-1
    deltas[i][1] = 2*(deltas[i][1]-d2min)/(d2max-d2min)-1

#30 training data points and 50 test points for LiH

#30 training data points and 50 test points for H4 (linear)

#250 training data points and 1250 test points for H4 (rectangle)



