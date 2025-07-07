import torch
import torch.nn as nn

class ScaledModel(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x):
        mean = self.mean.to(x.device)
        std = torch.clamp(self.std.to(x.device), min=1e-8)
        if x.dim() == 2:
            x = (x - mean) / std
        elif x.dim() == 3:
            x = (x - mean[None, None, :]) / std[None, None, :]
        else:
            raise ValueError(f"Unsupported input shape {x.shape}")
        return self.model(x)