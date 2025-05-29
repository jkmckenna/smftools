import torch
import torch.nn as nn
from base import BaseTorchModel
from positional import PositionalEncoding 
from .utils.grl import grad_reverse

    
class BaseTransformer(BaseTorchModel):
    def __init__(self, input_dim, model_dim, num_heads=4, num_layers=2, seq_len=None, use_learnable_pos=False, **kwargs):
        super().__init__(**kwargs)
        self.input_fc = nn.Linear(input_dim, model_dim)

        if use_learnable_pos:
            assert seq_len is not None, "Must provide seq_len if use_learnable_pos=True"
            self.pos_embed = nn.Parameter(torch.randn(seq_len, model_dim))  # (S, D)
            self.pos_encoder = None
        else:
            self.pos_encoder = PositionalEncoding(model_dim)
            self.pos_embed = None

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def encode(self, x, mask=None):
        """
        x: (B, S, input_dim)
        mask: (B, S) optional
        """
        x = self.input_fc(x)  # (B, S, D)
        if self.pos_embed is not None:
            x = x + self.pos_embed.unsqueeze(0).to(x.device)  # (B, S, D)
        elif self.pos_encoder is not None:
            x = self.pos_encoder(x)  # (B, S, D)
        if mask is not None:
            x = x * mask.unsqueeze(-1)  # (B, S, D)
        x = x.permute(1, 0, 2)  # (S, B, D)
        encoded = self.transformer(x)  # (S, B, D)
        return encoded.permute(1, 0, 2)  # (B, S, D)
    
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
    
class DANNTransformer(BaseTransformer):
    """
    n_batches here is the number of domain batches, not the number of batch dimensions in the tensor
    """
    def __init__(self, seq_len, model_dim=64, n_heads=4, n_layers=2, n_batches=1):
        super().__init__(
            input_dim=1,  # 1D scalar input per token
            model_dim=model_dim,
            num_heads=n_heads,
            num_layers=n_layers,
            seq_len=seq_len,
            use_learnable_pos=True  # enables learnable pos_embed in base
        )

        # Reconstruction head
        self.recon_head = nn.Linear(model_dim, 1)

        # Domain classification head
        self.domain_classifier = nn.Sequential(
            nn.Linear(model_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_batches)
        )

    def forward(self, x, alpha=1.0):
        """
        x: Tensor of shape (B, S) or (B, S, 1)
        alpha: GRL coefficient (float)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, S, 1)

        # Encode sequence
        h = self.encode(x)  # (B, S, D)

        # Head 1: Reconstruction
        recon = self.recon_head(h).squeeze(-1)  # (B, S)

        # Head 2: Domain classification via GRL
        pooled = h.mean(dim=1)  # (B, D)
        rev = grad_reverse(pooled, alpha)
        domain_logits = self.domain_classifier(rev)  # (B, n_batches)

        return recon, domain_logits