import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim, num_heads, dropout):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout),
            num_layers
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Reshape to (seq_len, batch_size, hidden_dim)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Reshape back to (batch_size, seq_len, hidden_dim)
        x = self.fc(x[:, -1, :])  # Take the last hidden state and pass it through a linear layer
        return x
