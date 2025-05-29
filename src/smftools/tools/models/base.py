import torch.nn as nn
from ..utils.device import detect_device

class BaseTorchModel(nn.Module):
    """
    Minimal base class for torch models that:
    - Stores device
    - Moves model to detected device on init
    """
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.device = detect_device() # detects available devices
        self.dropout_rate = dropout_rate # default dropout rate to be used in regularization.
        self.to(self.device)  # move model to device