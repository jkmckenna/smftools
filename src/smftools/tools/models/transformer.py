import torch
import torch.nn as nn
from models.base import BaseTorchModel
from models.positional import PositionalEncoding 

class BaseTransformer(BaseTorchModel):
    def __init__(self, input_dim, model_dim, num_heads=4, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.input_fc = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def encode(self, x, mask=None):
        """
        x: (batch, seq_len, input_dim)
        mask: (batch, seq_len) optional
        """
        x = self.input_fc(x) # -> (B, S, D)
        x = self.pos_encoder(x) # -> (B, S, D)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        x = x.permute(1, 0, 2)  # -> (S, B, D)
        encoded = self.transformer(x) # -> (S, B, D)
        return encoded.permute(1, 0, 2)  # -> (B, S, D)
    
class TransformerClassifier(BaseTransformer):
    def __init__(self, input_dim, model_dim, num_classes, num_heads=4, num_layers=2, **kwargs):
        super().__init__(input_dim, model_dim, num_heads, num_layers, **kwargs)
        self.cls_head = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        encoded = self.encode(x) # -> (B, S, D)
        pooled = encoded.mean(dim=1) # -> (B, D)
        return self.cls_head(pooled) # -> (B, C)

class MaskedTransformerPretrainer(BaseTransformer):
    def __init__(self, input_dim, model_dim, num_heads=4, num_layers=2, **kwargs):
        super().__init__(input_dim, model_dim, num_heads, num_layers, **kwargs)
        self.decoder = nn.Linear(model_dim, input_dim)

    def forward(self, x, mask):
        """
        x: (batch, seq_len, input_dim)
        mask: (batch, seq_len) optional
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        encoded = self.encode(x, mask=mask) # -> (B, S, D)
        return self.decoder(encoded) # -> (B, D)