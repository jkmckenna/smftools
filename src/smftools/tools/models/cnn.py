import torch
import torch.nn as nn
from .base import BaseTorchModel
import numpy as np

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
        gradcam_layer_idx=-1,
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

        # Build gradcam hooks
        self.gradcam_layer_idx = gradcam_layer_idx
        self.gradcam_activations = None
        self.gradcam_gradients = None
        if not hasattr(self, "_hooks_registered"):
            self._register_gradcam_hooks()
            self._hooks_registered = True

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, L]
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
    def _register_gradcam_hooks(self):
        def forward_hook(module, input, output):
            self.gradcam_activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradcam_gradients = grad_output[0].detach()

        target_layer = list(self.conv.children())[self.gradcam_layer_idx]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def compute_gradcam(self, x, class_idx=None):
        self.zero_grad()

        x = x.detach().clone().requires_grad_().to(self.device)

        was_training = self.training
        self.eval()  # disable dropout etc.

        output = self.forward(x)  # shape (B, C) or (B, 1)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        if output.shape[1] == 1:
            target = output.view(-1)  # shape (B,)
        else:
            target = output[torch.arange(output.shape[0]), class_idx]
        
        target.sum().backward(retain_graph=True)

        # restore training mode
        if was_training:
            self.train()

        # get activations and gradients (set these via forward hook!)
        activations = self.gradcam_activations  # (B, C, L)
        gradients = self.gradcam_gradients      # (B, C, L)

        weights = gradients.mean(dim=2, keepdim=True)  # (B, C, 1)
        cam = (weights * activations).sum(dim=1)       # (B, L)

        cam = torch.relu(cam)
        cam = cam / (cam.max(dim=1, keepdim=True).values + 1e-6)

        return cam
    
    def apply_gradcam_to_adata(self, dataloader, adata, obsm_key="gradcam", device="cpu"):
        self.to(device)
        self.eval()
        cams = []

        for batch in dataloader:
            x = batch[0].to(device)
            cam_batch = self.compute_gradcam(x)
            cams.append(cam_batch.cpu().numpy())

        cams = np.concatenate(cams, axis=0)  # shape: [n_obs, input_len]
        adata.obsm[obsm_key] = cams