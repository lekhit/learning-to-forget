import torch
import torch.nn as nn

class MLPPredictor(nn.Module):
    """
    Candidate A: 2-Layer Multi-Layer Perceptron (Baseline non-linear model).
    Input: x_i in R^d_x
    Architecture: Linear(d_x -> d_h) -> ReLU -> Linear(d_h -> 1) -> Sigmoid
    """
    def __init__(self, d_x=260, d_h=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_x, d_h),
            nn.ReLU(),
            nn.Linear(d_h, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class CNN1DPredictor(nn.Module):
    """
    Candidate B: 1D Temporal Convolutional Network.
    Suited to capture temporal correlation and locality of attention score bursts.
    Input: x_i in R^d_x
    """
    def __init__(self, d_x=260):
        super().__init__()
        # We split the input features:
        # First (d_x - 4) dimensions are raw key/value embeddings
        # Last 4 dimensions are scalar features (position, cumulative attention, etc.)
        self.d_x = d_x
        self.kv_dim = d_x - 4
        self.kv_linear = nn.Linear(self.kv_dim, 32)
        # 1D Convolution over temporal scalars (simulating a history of size 4)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 + 12, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Extract KV representation
        kv_feats = x[:, :self.kv_dim]
        kv_proj = self.kv_linear(kv_feats) # [batch, 32]
        
        # Extract scalar temporal features and add channels dimension
        scalars = x[:, self.kv_dim:self.d_x].unsqueeze(1) # [batch, 1, 4]
        conv_feats = self.conv(scalars) # [batch, 12]
        
        # Concatenate and project
        merged = torch.cat([kv_proj, conv_feats], dim=-1) # [batch, 44]
        return self.fc(merged)

class GRUPredictor(nn.Module):
    """
    Candidate C: Recurrent Network (Tiny GRU).
    Maintains a 16-dimensional hidden state vector h_i for each cached token to model sequence dynamics.
    """
    def __init__(self, d_x=260, d_h=16):
        super().__init__()
        self.d_x = d_x
        self.kv_dim = d_x - 4
        self.d_h = d_h
        # We use a GRUCell for sequential execution inside the attention block
        self.gru_cell = nn.GRUCell(input_size=4, hidden_size=d_h)
        self.fc = nn.Sequential(
            nn.Linear(self.kv_dim + d_h, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, hidden_state=None):
        batch_size = x.shape[0]
        kv_feats = x[:, :self.kv_dim]
        scalars = x[:, self.kv_dim:self.d_x] # [batch, 4]
        
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.d_h, device=x.device, dtype=x.dtype)
            
        # Update hidden state recurrently
        new_hidden = self.gru_cell(scalars, hidden_state) # [batch, d_h]
        
        # Merge with structural KV features
        merged = torch.cat([kv_feats, new_hidden], dim=-1) # [batch, kv_dim + d_h]
        out = self.fc(merged)
        
        return out, new_hidden

class PerceptronBranchPredictor(nn.Module):
    """
    Candidate D: Register-Level Perceptron Branch Predictor (Inspired by hardware architectures).
    Extremely lightweight single-layer perceptron mapping 16 key scalar markers at register level.
    Input: Sub-slice of x_i in R^16
    Architecture: Linear(16 -> 1) -> Sigmoid
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        # Hardware branch prediction initialization: initialize to "strongly keep" (high bias)
        # This prevents premature eviction before the perceptron has finished learning the context!
        nn.init.constant_(self.fc[0].bias, 1.5)
        nn.init.normal_(self.fc[0].weight, std=0.01)
    def forward(self, x):
        # Slice the last dimension (feature dimension) using ellipsis to support 2D or 3D inputs
        sub_x = x[..., :16]
        return self.fc(sub_x)
