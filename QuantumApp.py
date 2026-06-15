
import streamlit as st
import pandas as pd
import numpy as np
import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# --- 1. Quantum Helper Functions ---

def create_oracle(target_binary, num_qubits):
    """Creates a quantum oracle that flips the phase of the target binary string."""
    oracle = QuantumCircuit(num_qubits)
    
    # Apply X gates to flip 0s so the target state becomes all 1s
    for i in range(num_qubits):
        if target_binary[num_qubits - 1 - i] == '0':
            oracle.x(i)
            
    # Apply multi-controlled Z gate to flip the phase
    oracle.h(num_qubits - 1)
    if num_qubits > 1:
        oracle.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    oracle.h(num_qubits - 1)
    
    # Undo the X gates
    for i in range(num_qubits):
        if target_binary[num_qubits - 1 - i] == '0':
            oracle.x(i)
            
    return oracle

def create_diffuser(num_qubits):
    """Creates the Grover diffusion operator for amplitude amplification."""
    diffuser = QuantumCircuit(num_qubits)
    
    # Apply H and X gates to all qubits
    diffuser.h(range(num_qubits))
    diffuser.x(range(num_qubits))
    
    # Apply multi-controlled Z gate
    diffuser.h(num_qubits - 1)
    if num_qubits > 1:
        diffuser.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    diffuser.h(num_qubits - 1)
    
    # Undo X and H gates
    diffuser.x(range(num_qubits))
    diffuser.h(range(num_qubits))
    
    return diffuser

# --- 2. Streamlit UI & Logic ---

st.title("Quantum Database Search (Grover's Algorithm)")
st.write("Upload a dataset, select an item, and watch a simulated quantum computer find it using superposition, phase inversion, and amplitude amplification.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    
    # Select column and target
    col = st.selectbox("Select the column to search:", df.columns)
    unique_items = df[col].dropna().unique()
    target_item = st.selectbox("Select the target to find:", unique_items)
    
    if st.button("Run Quantum Search"):
        # Find the index of the target item
        target_index = df[df[col] == target_item].index[0]
        N = len(df)
        
        # Calculate required qubits (N <= 2^n)
        num_qubits = math.ceil(math.log2(N))
        
        # Prevent the simulator from crashing your computer (Limit to 10 qubits / 1024 rows)
        if num_qubits > 10:
            st.error(f"Dataset is too large for classical simulation. Max 1024 rows. (Your data needs {num_qubits} qubits).")
        else:
            st.success(f"Target '{target_item}' found at classical index {target_index}.")
            
            # 1. Encode into binary
            target_binary = format(target_index, f'0{num_qubits}b')
            st.write(f"**Step 1:** Mapping index {target_index} to quantum state `|{target_binary}⟩` using {num_qubits} qubits.")
            
            # Calculate iterations: roughly (pi/4) * sqrt(N)
            iterations = math.floor(math.pi / 4 * math.sqrt(2**num_qubits))
            if iterations == 0: iterations = 1
            st.write(f"**Step 2 & 3:** Applying Superposition and Diffuser for {iterations} iterations.")
            
            # Build the circuit
            qc = QuantumCircuit(num_qubits, num_qubits)
            
            # Step 2: Initialize in Superposition
            qc.h(range(num_qubits))
            
            # Step 3: Apply Grover iterations
            oracle = create_oracle(target_binary, num_qubits)
            diffuser = create_diffuser(num_qubits)
            
            for _ in range(iterations):
                qc.append(oracle.to_gate(label="Oracle"), range(num_qubits))
                qc.append(diffuser.to_gate(label="Diffuser"), range(num_qubits))
                
            # Step 4: Measure
            qc.measure(range(num_qubits), range(num_qubits))
            
            # Run Simulation
            st.write("**Step 4:** Simulating Measurement...")
            simulator = Aer.get_backend('qasm_simulator')
            compiled_circuit = transpile(qc, simulator)
            result = simulator.run(compiled_circuit, shots=1024).result()
            counts = result.get_counts()
            
            # Plot Results
            st.subheader("Quantum Measurement Results")
            fig, ax = plt.subplots()
            plot_histogram(counts, ax=ax)
            st.pyplot(fig)
            
            # Verify
            most_probable_state = max(counts, key=counts.get)
            if most_probable_state == target_binary:
                st.success(f"Quantum search successfully collapsed onto the target state `|{most_probable_state}⟩`!")
            else:
                st.warning("Algorithm did not find the target. (Check qubit scaling or iterations).")