import torch
import torch.nn as nn

class ClassicalSSMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Classical discrete transition matrix A and input matrix B
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.B = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.activation = nn.Tanh()

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        device = x.device
        h_t = torch.zeros(batch_size, self.hidden_dim).to(device)
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            # Discrete classical update
            h_t = self.activation(torch.matmul(h_t, self.A) + torch.matmul(x_t, self.B.T))
            outputs.append(h_t.unsqueeze(1))
            
        return torch.cat(outputs, dim=1)
