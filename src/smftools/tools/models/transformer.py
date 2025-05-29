import torch
import torch.nn as nn
from .base import BaseTorchModel
from .positional import PositionalEncoding 
from ..utils.grl import grad_reverse

    
class BaseTransformer(BaseTorchModel):
    def __init__(self, input_dim, model_dim, num_heads=4, num_layers=2, seq_len=None, use_learnable_pos=False, **kwargs):
        super().__init__(**kwargs)
        # Input FC layer to map D_input to D_model
        self.input_fc = nn.Linear(input_dim, model_dim)

        if use_learnable_pos:
            assert seq_len is not None, "Must provide seq_len if use_learnable_pos=True"
            self.pos_embed = nn.Parameter(torch.randn(seq_len, model_dim))  # (S, D)
            self.pos_encoder = None
        else:
            self.pos_encoder = PositionalEncoding(model_dim)
            self.pos_embed = None

        # Specify the transformer encoder structure
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=False)
        # Stack the transformer encoder layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def encode(self, x, mask=None):
        """
        x: (B, S, D_input)
        mask: (B, S) optional
        """
        x = self.input_fc(x)  # (B, S, D_model)
        if self.pos_embed is not None:
            x = x + self.pos_embed.unsqueeze(0).to(x.device)  # (B, S, D_model)
        elif self.pos_encoder is not None:
            x = self.pos_encoder(x)  # (B, S, D_model)
        if mask is not None:
            x = x * mask.unsqueeze(-1)  # (B, S, D_model)
        x = x.permute(1, 0, 2)  # (S, B, D_model)
        encoded = self.transformer(x)  # (S, B, D_model)
        return encoded.permute(1, 0, 2)  # (B, S, D_model)
    
class TransformerClassifier(BaseTransformer):
    def __init__(self, input_dim, model_dim, num_classes, num_heads=4, num_layers=2, **kwargs):
        super().__init__(input_dim, model_dim, num_heads, num_layers, **kwargs)
        # Classification head
        self.cls_head = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        encoded = self.encode(x) # -> (B, S, D_model)
        pooled = encoded.mean(dim=1) # -> (B, D_model)
        return self.cls_head(pooled) # -> (B, C)
    
class DANNTransformerClassifier(TransformerClassifier):
    def __init__(self, input_dim, model_dim, num_classes, n_domains, **kwargs):
        super().__init__(input_dim, model_dim, num_classes, **kwargs)
        self.domain_classifier = nn.Sequential(
            nn.Linear(model_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_domains)
        )

    def forward(self, x, alpha=1.0):
        encoded = self.encode(x)  # (B, S, D_model)
        pooled = encoded.mean(dim=1)  # (B, D_model)

        class_logits = self.cls_head(pooled)
        domain_logits = self.domain_classifier(grad_reverse(pooled, alpha))

        return class_logits, domain_logits

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
        encoded = self.encode(x, mask=mask) # -> (B, S, D_model)
        return self.decoder(encoded) # -> (B, D_input)
    
class DANNTransformer(BaseTransformer):
    """
    """
    def __init__(self, seq_len, model_dim, n_heads, n_layers, n_domains):
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
            nn.Linear(128, n_domains)
        )

    def forward(self, x, alpha=1.0):
        """
        x: Tensor of shape (B, S) or (B, S, 1)
        alpha: GRL coefficient (float)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, S, 1)

        # Encode sequence
        h = self.encode(x)  # (B, S, D_model)

        # Head 1: Reconstruction
        recon = self.recon_head(h).squeeze(-1)  # (B, S)

        # Head 2: Domain classification via GRL
        pooled = h.mean(dim=1)  # (B, D_model)
        rev = grad_reverse(pooled, alpha)
        domain_logits = self.domain_classifier(rev)  # (B, n_batches)

        return recon, domain_logits
    
