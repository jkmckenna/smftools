import torch
import torch.nn as nn
from .base import BaseTorchModel

class MLPClassifier(BaseTorchModel):
    def __init__(self, input_dim, num_classes, hidden_sizes=(128, 64), **kwargs):
        super().__init__(**kwargs)
        output_size = 1 if num_classes == 2 else num_classes
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout_rate)])
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)