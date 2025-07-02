import torch
import torch.nn as nn
import numpy as np
from ..utils.device import detect_device

class BaseTorchModel(nn.Module):
    """
    Minimal base class for torch models that:
    - Stores device and dropout regularization
    """
    def __init__(self, dropout_rate=0.0):
        super().__init__()
        self.device = detect_device() # detects available devices
        self.dropout_rate = dropout_rate # default dropout rate to be used in regularization.

    def compute_saliency(
        self,
        x,
        target_class=None,
        reduction="sum",
        smoothgrad=False,
        smooth_samples=25,
        smooth_noise=0.1
    ):
        """
        Compute vanilla saliency or SmoothGrad saliency.
        
        Arguments:
        ----------
        x : torch.Tensor
            Input tensor [B, S, D].
        target_class : int or list, optional
            If None, uses model predicted class.
        reduction : str
            'sum' or 'mean' across channels
        smoothgrad : bool
            Whether to apply SmoothGrad.
        smooth_samples : int
            Number of noisy samples for SmoothGrad
        smooth_noise : float
            Standard deviation of noise added to input
        """
        self.eval()
        x = x.clone().detach().requires_grad_(True)
        
        if smoothgrad:
            saliency_accum = torch.zeros_like(x)
            for i in range(smooth_samples):
                noise = torch.normal(mean=0.0, std=smooth_noise, size=x.shape).to(x.device)
                x_noisy = x + noise
                x_noisy.requires_grad_(True)
                x_noisy.retain_grad()  # <<< fixes the issue
                logits = self.forward(x_noisy)
                if target_class is None:
                    target = logits.max(dim=1)[1]
                else:
                    target = torch.tensor(target_class).to(x.device)
                scores = logits[torch.arange(x.shape[0]), target]
                scores.sum().backward()
                saliency_accum += x_noisy.grad.detach()
            saliency = saliency_accum / smooth_samples
        else:
            logits = self.forward(x)
            if target_class is None:
                target = logits.max(dim=1)[1]
            else:
                target = torch.tensor(target_class).to(x.device)
            scores = logits[torch.arange(x.shape[0]), target]
            scores.sum().backward()
            saliency = x.grad.detach()
        
        # reduce across channels
        if reduction == "sum" and x.ndim == 3:
            return saliency.abs().sum(dim=-1)
        elif reduction == "mean" and x.ndim == 3:
            return saliency.abs().mean(dim=-1)
        else:
            return saliency.abs()
        
    def compute_gradient_x_input(self, x, target_class=None):
        """
        Computes gradient × input attribution.
        """
        x = x.clone().detach().requires_grad_(True)
        logits = self.forward(x)
        if target_class is None:
            target_class = logits.argmax(dim=1)
        scores = logits[torch.arange(logits.size(0)), target_class]
        scores.sum().backward()
        grads = x.grad
        return grads * x

    def compute_integrated_gradients(self, x, target=None, steps=50):
        """
        Compute Integrated Gradients for a batch of x.
        If target=None, uses the predicted class.
        Returns: [B, seq_len, channels] attribution tensor
        """
        from captum.attr import IntegratedGradients

        ig = IntegratedGradients(self)
        self.eval()
        x = x.requires_grad_(True)

        with torch.no_grad():
            logits = self(x)
            if target is None:
                target = logits.argmax(dim=1)

        attributions, delta = ig.attribute(
            x,
            baselines=torch.zeros_like(x),
            target=target,
            n_steps=steps,
            return_convergence_delta=True
        )
        return attributions, delta

    def compute_deeplift(
        self,
        x,
        baseline=None,
        target_class=None,
        reduction="sum"
    ):
        """
        Compute DeepLIFT scores using captum.

        baseline:
            reference input for DeepLIFT.
        """
        from captum.attr import DeepLift

        self.eval()
        deeplift = DeepLift(self)
        if target_class is None:
            target_class = self.forward(x).max(dim=1)[1]
        attr = deeplift.attribute(x, target=target_class, baselines=baseline)

        if reduction == "sum" and x.ndim == 3:
            return attr.abs().sum(dim=-1)
        elif reduction == "mean" and x.ndim == 3:
            return attr.abs().mean(dim=-1)
        else:
            return attr.abs()
        
    def compute_occlusion(self, x, target_class=None, window_size=5):
        """
        Computes per-sample occlusion attribution.
        Returns [B, S]
        """
        B, S, D = x.shape
        x = x.detach().cpu().numpy()
        occlusion_scores = np.zeros((B, S))
        baseline = np.mean(x, axis=1, keepdims=True)  # [B,1,D]

        for i in range(S):
            x_occluded = x.copy()
            left = max(0, i - window_size // 2)
            right = min(S, i + window_size // 2)
            x_occluded[:, left:right, :] = baseline[:, left:right, :]
            x_tensor = torch.tensor(x_occluded, device=self.device, dtype=torch.float32)
            logits = self.forward(x_tensor)
            if target_class is None:
                target_class = logits.argmax(dim=1)
            scores = logits[torch.arange(B), target_class]
            occlusion_scores[:, i] = scores.detach().cpu().numpy()
        return occlusion_scores

    def apply_attributions_to_adata(
        model,
        dataloader,
        adata,
        method="saliency",  # saliency, smoothgrad, IG, deeplift, gradxinput, occlusion
        adata_key="attributions",
        baseline=None,
        device="cpu",
        target_class=None,
        normalize=True
    ):
        """
        Apply a chosen attribution method to a dataloader and store results in adata.
        """

        results = []
        model.to(device)
        model.eval()

        for batch in dataloader:
            x = batch[0].to(device)
            if method == "saliency":
                attr = model.compute_saliency(x, target_class=target_class)
            elif method == "smoothgrad":
                attr = model.compute_saliency(x, smoothgrad=True, target_class=target_class)
            elif method == "IG":
                attributions, delta = model.compute_integrated_gradients(x, target=target_class)
                attr = attributions
            elif method == "deeplift":
                attr = model.compute_deeplift(x, baseline=baseline, target_class=target_class)
            elif method == "gradxinput":
                attr = model.compute_gradient_x_input(x, target_class=target_class)
            elif method == "occlusion":
                # note: occlusion returns 1D scores across positions
                attr = model.compute_occlusion(x, target_class=target_class)
            else:
                raise ValueError(f"Unknown method {method}")
            
            # make sure it's np
            attr = attr.detach().cpu().numpy() if torch.is_tensor(attr) else attr
            results.append(attr)

        results_stacked = np.concatenate(results, axis=0)
        adata.obsm[adata_key] = results_stacked

        if normalize:
            row_max = np.max(np.abs(results_stacked), axis=1, keepdims=True)
            row_max[row_max == 0] = 1  # avoid divide by zero
            results_normalized = results_stacked / row_max
            adata.obsm[adata_key + "_normalized"] = results_normalized