import torch
import torch.nn as nn
from .base import BaseTorchModel

class CNNClassifier(BaseTorchModel):
    def __init__(
        self,
        input_size,
        num_classes=2,
        conv_channels=[16, 32],
        kernel_sizes=[3, 3],
        fc_dims=[64],
        use_batchnorm=False,
        use_pooling=False,
        dropout=0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.name = "CNNClassifier"

        # Normalize input
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(conv_channels)
        assert len(conv_channels) == len(kernel_sizes)

        layers = []
        in_channels = 1

        # Build conv layers
        for out_channels, ksize in zip(conv_channels, kernel_sizes):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=ksize, padding=ksize // 2))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            if use_pooling:
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)

        # Determine flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_size)
            conv_out = self.conv(dummy)
            flattened_size = conv_out.view(1, -1).shape[1]

        # Build FC layers
        fc_layers = []
        in_dim = flattened_size
        for dim in fc_dims:
            fc_layers.append(nn.Linear(in_dim, dim))
            fc_layers.append(nn.ReLU())
            if dropout > 0:
                fc_layers.append(nn.Dropout(dropout))
            in_dim = dim

        output_size = 1 if num_classes == 2 else num_classes
        fc_layers.append(nn.Linear(in_dim, output_size))

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, L]
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)