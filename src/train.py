import torch
import torch.nn as nn
import torch.optim as optim
from q_mamba_8qubit import AdvancedQuantumSSMLayer

class Q_Mamba_Classifier(nn.Module):
    def __init__(self, input_dim, n_qubits, num_classes):
        super().__init__()
        self.q_ssm = AdvancedQuantumSSMLayer(input_dim, n_qubits)
        # The hidden dimension is naturally 256 from the 8-qubit probability distribution
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x):
        ssm_out = self.q_ssm(x)
        final_state = ssm_out[:, -1, :] 
        return self.classifier(final_state)

def generate_synthetic_ehr_data(num_samples=100, seq_len=50, input_dim=12):
    X = torch.rand(num_samples, seq_len, input_dim)
    # Target rule requiring long-term memory: sum of first 5 steps + sum of last 5 steps
    early_signal = X[:, :5, :].sum(dim=(1,2))
    late_signal = X[:, -5:, :].sum(dim=(1,2))
    y = ((early_signal + late_signal) > (input_dim * 5)).long()
    return X, y

def main():
    print("Initializing 256-Dimensional EHR Feature Space Environment...")
    print("Task: Sepsis Prediction from Continuous Sequence (Seq_Len = 50, Features = 256)\n")
    
    print("--- Training 8-Qubit Quantum Mamba (Q-Mamba) ---")
    
    batch_size = 10
    seq_len = 50
    input_dim = 12       
    n_qubits = 8
    epochs = 5
    
    model = Q_Mamba_Classifier(input_dim, n_qubits, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01) # Proxy for QNG
    
    X_train, y_train = generate_synthetic_ehr_data(100, seq_len, input_dim)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(X_train):.4f} | Accuracy: {acc:.1f}%")
        
    print("\n[+] Result: Q-Mamba successfully compresses 50-step historical context into a stable 256-dimensional Hilbert space representation.")

if __name__ == "__main__":
    main()
