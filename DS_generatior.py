#Creation of electronic Hamiltonians of LiH, H4 (rectangle) and H4(line)

#Atomic distances:
    #LiH and H4(line) [0.5A, 3.3A]
    #H4(rectangle) [0.5A, 2.0A]x[0.5A, 2.0A]

#Use PySCF and OpenFermion
import numpy as np
from pyscf import gto, scf
from openfermionpyscf import run_pyscf
from openfermion.transforms import get_fermion_operator
from openfermion.hamiltonians import MolecularData

#Perform Hartree-Fock calculation, STO-3G minimal basis
#Construct fermionic sec-quantized Hamiltonian for all molecules and config
#Hamiltonian mapped to sum of Pauli operators by Jordan-Wigner transformation H(r)
#Electric Hamiltonians for all molecules turn into 8 qubit hamiltonians

basis = 'sto-3g'

#LiH 
LiH_ham = [
    for d in np.
]

#H4 lin
#H4 rec

#For noiseless simulation, GS of H(r) is prepared by exact diagonalization

#For noiseless simulation, VQE applied to H(r), approximate ground state obtained as U(theta) 0^n
#U(theta) is VQC (ansatz)
#ADdapt UCCSD as U(theta)

#Compute quantities of excited state properties to be predicted (formula 2)

#Diagonalization of H(r) for both simultations

#Fit values into [-1, 1] range

#30 training data points and 50 test points for LiH

#30 training data points and 50 test points for H4 (linear)

#250 training data points and 1250 test points for H4 (rectangle)



