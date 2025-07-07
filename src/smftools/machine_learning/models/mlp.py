import torch
import torch.nn as nn
from .base import BaseTorchModel
    
class MLPClassifier(BaseTorchModel):
    def __init__(self, input_dim, num_classes=2, hidden_dims=[64, 64], dropout=0.2, use_batchnorm=True, **kwargs):
        super().__init__(**kwargs)
        layers = []
        in_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        output_size = 1 if num_classes == 2 else num_classes

        layers.append(nn.Linear(in_dim, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)