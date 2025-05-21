#Creation of electronic Hamiltonians 

#Use PySCF and OpenFermion
import numpy as np
from openfermion.chem import MolecularData as md
from openfermionpyscf import run_pyscf
from openfermion.hamiltonians import generate_hamiltonian
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator
from scipy.sparse.linalg import eigsh

#Perform Hartree-Fock calculation, STO-3G minimal basis
#Construct fermionic sec-quantized Hamiltonian for all molecules and config
#Hamiltonian mapped to sum of Pauli operators by Jordan-Wigner transformation H(r)
#Electric Hamiltonians for all molecules turn into 8 qubit hamiltonians

class Data_Generator:
    ham = []
    dis = []
    molecules = []

    def __init__(self, elements, distance, bas):
        self.dis = np.linspace(distance[0], distance[1], 200)
        for d in self.dis:
            molecule = md(
                geometry = [[elements[0], (0, 0, 0)], [elements[1], (0, 0, d)]], 
                basis = bas,
                multiplicity = 1
            )

            mol = run_pyscf(molecule, run_scf=1, run_mp2=0, run_cisd=0, run_ccsd=0, run_fci=0)
            self.molecules.append(mol)

            reduced_one_body = mol.one_body_integrals[:4, :4]
            reduced_two_body = mol.two_body_integrals[:4, :4, :4, :4]
            
            hamiltonian = generate_hamiltonian(
                one_body_integrals=reduced_one_body,
                two_body_integrals=reduced_two_body,
                constant=mol.nuclear_repulsion
            )

            self.ham.append(hamiltonian)

        for i in range(len(self.ham)):
            self.ham[i] = jordan_wigner(self.ham[i])

    def get_data(self):
        """
        Generates the data from the Jordan-Wigner transformed Hamiltonians
    
        Returns:
            An array of pairs of integeres deltaE_1 and deltaE_2, and the ground state from the Hamiltonian
        """
        eigenvalues = []
        eigenvectors = []

        for H in self.ham:
            SH = get_sparse_operator(H)
            evals, evcts = eigsh(SH, k=3, which='SA')  # 3 menores valores propios
            eigenvalues.append(evals)
            eigenvectors.append(evcts[:, 0])

        #Compute quantities of excited state properties to be predicted (formula 2) deltaE1 and deltaE2
        deltas = []
        d1min = 1000000000
        d1max = -1000000000
        d2min = 1000000000
        d2max = -1000000000

        for eivls in eigenvalues:
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
        
        return deltas, eigenvectors





