import torch
import torch.nn as nn
from .base import BaseTorchModel

class RNNClassifier(BaseTorchModel):
    def __init__(self, input_size, hidden_dim, num_classes, **kwargs):
        super().__init__(**kwargs)
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, batch_first=True)
        # Define fully connected output layer
        output_size = 1 if num_classes == 2 else num_classes
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, L] → for LSTM expecting batch_first
        _, (h_n, _) = self.lstm(x)  # h_n: [1, B, H]
        return self.fc(h_n.squeeze(0))  # [B, H] → [B, num_classes]