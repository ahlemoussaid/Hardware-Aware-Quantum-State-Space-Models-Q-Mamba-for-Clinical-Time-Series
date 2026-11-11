import torch
import torch.nn as nn
import pennylane as qml

class AdvancedQuantumSSMLayer(nn.Module):
    """
    An 8-Qubit Quantum-Enhanced Selective State-Space Model Layer (Q-Mamba).
    Utilizes 8 qubits to natively represent a 256-dimensional hidden state (2^8 = 256).
    Replaces the classical discrete state transition matrix with unitary continuous-time evolution.
    """
    def __init__(self, input_dim, n_qubits=8, q_depth=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.hidden_dim = 256 # 2^8 states
        
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        @qml.qnode(self.dev, interface="torch")
        def quantum_state_transition(inputs, weights):
            # Embed the continuous inputs into the quantum state
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            
            # Parameterized unitary evolution
            for layer in range(q_depth):
                for i in range(self.n_qubits):
                    qml.Rot(*weights[layer, i], wires=i)
                # Entanglement block ensuring global state mixing
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
                qml.CNOT(wires=[self.n_qubits-1, 0])
                
            # Measure probabilities of all 256 basis states to form the new continuous hidden state
            return qml.probs(wires=range(self.n_qubits))
        
        weight_shapes = {"weights": (q_depth, self.n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_state_transition, weight_shapes)
        
        # Projection from combined (input + previous 256D state) down to 8 rotation angles
        self.proj_in = nn.Linear(input_dim + self.hidden_dim, self.n_qubits)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        h_t = torch.zeros(batch_size, self.hidden_dim).to(device)
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            combined = torch.cat([x_t, h_t], dim=-1)
            
            # Map down to 8 angles [-pi, pi] for embedding
            q_in = torch.tanh(self.proj_in(combined)) * torch.pi 
            
            # Unitary evolution natively outputs a 256-dimensional probability distribution
            h_t = self.q_layer(q_in)
            outputs.append(h_t.unsqueeze(1))
            
        return torch.cat(outputs, dim=1)
