import torch
import torch.nn as nn
from .base import BaseTorchModel
from .positional import PositionalEncoding 
from ..utils.grl import grad_reverse
import numpy as np

class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, src, src_mask=None, is_causal=False, src_key_padding_mask=None):
        self_attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,  # preserve [B, num_heads, S, S]
            is_causal=is_causal
        )
        src = src + self.dropout1(self_attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # Save attention weights to module
        self.attn_weights = attn_weights  # Save to layer
        return src
    
class BaseTransformer(BaseTorchModel):
    def __init__(self, 
                 input_dim=1, 
                 model_dim=64, 
                 num_heads=4, 
                 num_layers=2, 
                 dropout=0.2,
                 seq_len=None, 
                 use_learnable_pos=False, 
                 use_cls_token=True,
                 **kwargs):
        super().__init__(**kwargs)
        # Input FC layer to map D_input to D_model
        self.model_dim = model_dim
        self.input_fc = nn.Linear(input_dim, model_dim)
        self.ff_dim = model_dim * 4
        self.dropout = dropout
        self.use_cls_token = use_cls_token

        self.attn_weights = []
        self.attn_grads = []

        if use_learnable_pos:
            assert seq_len is not None, "Must provide seq_len if use_learnable_pos=True"
            self.pos_embed = nn.Parameter(torch.randn(seq_len + (1 if use_cls_token else 0), model_dim))
            self.pos_encoder = None
        else:
            self.pos_encoder = PositionalEncoding(model_dim)
            self.pos_embed = None

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))  # (1, 1, D)

        # Specify the transformer encoder structure
        encoder_layer = TransformerEncoderLayerWithAttn(d_model=model_dim, nhead=num_heads, batch_first=True, dim_feedforward=self.ff_dim, dropout=self.dropout)
        # Stack the transformer encoder layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Register hooks
        for layer in self.transformer.layers:
            layer.self_attn.register_forward_hook(self._save_attn_weights)
            layer.self_attn.register_full_backward_hook(self._save_attn_grads)

    def _save_attn_weights(self, module, input, output):
        self.attn_weights.append(output[1].detach())

    def _save_attn_grads(self, module, grad_input, grad_output):
        self.attn_grads.append(grad_output[0].detach())

    def encode(self, x, mask=None):
        if x.dim() == 2:  # (B, S)
            x = x.unsqueeze(-1)
        elif x.dim() == 1:  # (S,)
            x = x.unsqueeze(0).unsqueeze(-1)
        elif x.dim() == 3:
            pass
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        x = self.input_fc(x)  # (B, S, D)

        B, S, D = x.shape
        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
            x = torch.cat([cls, x], dim=1)  # (B, S+1, D)

        if self.pos_embed is not None:
            x = x + self.pos_embed.unsqueeze(0)[:, :x.shape[1], :]
        elif self.pos_encoder is not None:
            x = self.pos_encoder(x)

        if mask is not None:
            pad = torch.ones(B, 1, device=mask.device) if self.use_cls_token else 0
            mask = torch.cat([pad, mask], dim=1) if self.use_cls_token else mask
            x = x * mask.unsqueeze(-1)

        encoded = self.transformer(x)
        return encoded
    
    def compute_attn_grad(self, reduction='mean'):
        """
        Computes attention × gradient scores across layers.
        Returns: [B, S] tensor of importance scores
        """
        scores = []
        for attn, grad in zip(self.attn_weights, self.attn_grads):
            # attn: [B, H, S, S]
            # grad: [B, S, D]
            attn = attn.mean(dim=1)            # [B, S, S]
            grad_norm = grad.norm(dim=-1)      # [B, S]
            attn_grad_score = (attn * grad_norm.unsqueeze(1)).sum(dim=-1)  # [B, S]
            scores.append(attn_grad_score)

        # Combine across layers
        stacked = torch.stack(scores, dim=0)  # [L, B, S]
        if reduction == "mean":
            return stacked.mean(dim=0)        # [B, S]
        elif reduction == "sum":
            return stacked.sum(dim=0)         # [B, S]
        else:
            return stacked                    # [L, B, S]

    def compute_rollout(self):
        """
        Computes attention rollout: [B, S, S] final attention influence map
        """
        device = self.attn_weights[0].device
        B, S = self.attn_weights[0].shape[0], self.attn_weights[0].shape[-1]
        rollout = torch.eye(S, device=device).unsqueeze(0).repeat(B, 1, 1)  # [B, S, S]

        for attn in self.attn_weights:
            attn_heads = attn.mean(dim=1)  # [B, S, S]
            attn_heads = attn_heads + torch.eye(S, device=device).unsqueeze(0)  # add residual
            attn_heads = attn_heads / attn_heads.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            rollout = torch.bmm(attn_heads, rollout)  # [B, S, S]
        
        return rollout  # [B, S, S]
    
    def reset_attn_buffers(self):
        self.attn_weights = []
        self.attn_grads = []

    def get_attn_layer(self, layer_idx=0, head_idx=None):
        """
        Returns attention map from a specific layer (and optionally head).
        """
        attn = self.attn_weights[layer_idx]  # [B, H, S, S]
        if head_idx is not None:
            attn = attn[:, head_idx]  # [B, S, S]
        return attn
    
    def apply_attn_interpretations_to_adata(self, dataloader, adata, 
                                            obsm_key_grad="attn_grad", 
                                            obsm_key_rollout="attn_rollout",
                                            device="cpu"):
        self.to(device)
        self.eval()
        grad_maps = []
        rollout_maps = []

        for batch in dataloader:
            x = batch[0].to(device)
            x.requires_grad_()

            self.reset_attn_buffers()
            logits = self(x)

            if logits.shape[1] == 1:
                target_score = logits.squeeze()
            else:
                target_score = logits.max(dim=1).values

            target_score.sum().backward()

            grad = self.compute_attn_grad()  # [B, S+1]
            if self.use_cls_token:
                grad = grad[:, 1:]  # ignore CLS token
            grad_maps.append(grad.detach().cpu().numpy())

        grad_concat = np.concatenate(grad_maps, axis=0)
        adata.obsm[obsm_key_grad] = grad_concat

        # add per-row normalized version
        grad_normed = grad_concat / (np.max(grad_concat, axis=1, keepdims=True) + 1e-8)
        adata.obsm[f"{obsm_key_grad}_normalized"] = grad_normed
    
class TransformerClassifier(BaseTransformer):
    def __init__(self, 
                 input_dim, 
                 num_classes, 
                 **kwargs):
        super().__init__(input_dim, **kwargs)
        # Classification head
        output_size = 1 if num_classes == 2 else num_classes
        self.cls_head = nn.Linear(self.model_dim, output_size)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        self.reset_attn_buffers()
        if x.dim() == 2:  # shape (B, S)
            x = x.unsqueeze(-1)  # → (B, S, 1)
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(-1)  # just in case (S,) → (1, S, 1)
        else:
            pass
        encoded = self.encode(x) # -> (B, S, D_model)
        if self.use_cls_token:
            pooled = encoded[:, 0]  # (B, D)
        else:
            pooled = encoded.mean(dim=1)  # (B, D)        out = self.cls_head(pooled) # -> (B, C)

        out = self.cls_head(pooled)  # (B, C)
        return out
    
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
    
