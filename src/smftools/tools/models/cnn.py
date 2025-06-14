import torch
import torch.nn as nn
from .base import BaseTorchModel

class CNNClassifier(BaseTorchModel):
    def __init__(self, input_size, num_classes, **kwargs):
        super().__init__(**kwargs)
        # Define convolutional layers
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        # Define activation function
        self.relu = nn.ReLU()

        # Determine the flattened size dynamically
        dummy_input = torch.zeros(1, 1, input_size)
        with torch.no_grad():
            dummy_output = self._forward_conv(dummy_input)
        flattened_size = dummy_output.view(1, -1).shape[1]

        # Define fully connected layers
        self.fc1 = nn.Linear(flattened_size, 64)
        output_size = 1 if num_classes == 2 else num_classes
        self.fc2 = nn.Linear(64, output_size)

    def _forward_conv(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, L]
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        return self.fc2(x)
    
class DeeperCNNClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),  # RF = 7
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=3, dilation=4, padding=4),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # output shape [B, 128, 1]
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes if num_classes > 2 else 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, L)
        x = self.conv_block(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x