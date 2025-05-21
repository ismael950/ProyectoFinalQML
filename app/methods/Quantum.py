from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import Statevector, Pauli
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator 
from qiskit_aer import Aer
import numpy as np

class Quantum_Unit:
    x = []
    gs = []
    y = []
    N_qubits = 0
    hamiltonian = 0
    evolution_gate = 0
    qc = 0
    def __init__(self, ground_states, deltas, qubits):
        self.gs = ground_states
        self.y = deltas
        self.N_qubits = qubits
    
    def create_tfim_hamiltonian(self):
        """
        Creates a random TFIM Hamiltonian with Gaussian distributed coefficients
        
        Args:
            N_qubits: Number of qubits in the system
        """
        # Initialize empty list for Pauli terms
        pauli_terms = []
        
        # Add ZZ terms with random J_ij coefficients
        for i in range(self.N_qubits):
            for j in range(i+1, self.N_qubits):
                J_ij = np.random.normal(0.75, 0.1)  # N(0.75, 0.1)
                pauli_str = 'I'*i + 'Z' + 'I'*(j-i-1) + 'Z' + 'I'*(self.N_qubits-j-1)
                pauli_terms.append((pauli_str, J_ij))
        
        # Add X terms with random h_i coefficients
        for i in range(self.N_qubits):
            h_i = np.random.normal(1.0, 0.1)  # N(1.0, 0.1)
            pauli_str = 'I'*i + 'X' + 'I'*(self.N_qubits-i-1)
            pauli_terms.append((pauli_str, h_i))
        
        # Create SparsePauliOp from the terms
        self.hamiltonian = SparsePauliOp.from_list(pauli_terms)
        

    def create_entangler(self, N_qubits, T=10, reps=1):
        """
        Creates the entangler U_ent = e^(-i H_TFIM T)
        
        Args:
            N_qubits: Number of qubits
            T: Evolution time (default 10)
            reps: Number of Trotter steps (default 1)
        """
        # Create the random TFIM Hamiltonian
        self.create_tfim_hamiltonian()
        
        # Create evolution gate
        self.evolution_gate = PauliEvolutionGate(self.hamiltonian, time=T, synthesis=LieTrotter(reps=reps))
        
        # Create circuit and add the gate
        self.qc = QuantumCircuit(self.N_qubits)
        self.qc.append(self.evolution_gate, range(self.N_qubits))
    
    def process_state(self, state_vector):
        """
        Procesa un estado cuántico puro: aplica el entangler y mide ⟨X⟩, ⟨Y⟩, ⟨Z⟩ para cada qubit.
        
        Args:
            state_vector: Estado base como vector de amplitudes (len = 2**n).
            
        Returns:
            result_vector: Lista concatenada con ⟨X⟩, ⟨Y⟩, ⟨Z⟩ para cada qubit. Longitud = 3*n_qubits
        """
        # Validación de dimensiones
        assert len(state_vector) == 2**self.N_qubits, "Tamaño del estado no coincide con número de qubits."

        # Evolucionar el estado con el entangler
        sv = Statevector(state_vector)
        U = Operator(self.qc)  # unitaria del circuito entangler
        evolved_sv = sv.evolve(U)

        # Medir ⟨X⟩, ⟨Y⟩, ⟨Z⟩ para cada qubit
        expectations = {'X': [], 'Y': [], 'Z': []}
        pauli_labels = ['X', 'Y', 'Z']

        for qubit in range(self.N_qubits):
            for pauli in pauli_labels:
                label = 'I' * qubit + pauli + 'I' * (self.N_qubits - qubit - 1)
                op = Pauli(label)
                val = evolved_sv.expectation_value(op)
                expectations[pauli].append(np.real(val))  # tomar solo la parte real por seguridad

        # Concatenar en formato [X0, Y0, Z0, X1, Y1, Z1, ..., Xn, Yn, Zn]
        result_vector = []
        for qubit in range(self.N_qubits):
            result_vector.extend([
                expectations['X'][qubit],
                expectations['Y'][qubit],
                expectations['Z'][qubit]
            ])

        return result_vector
        
    def all_states(self):
        self.x=[]
        for GS in self.gs:
            self.x.append(self.process_state(GS))




        