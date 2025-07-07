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
        smooth_noise=0.1,
        signed=True
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
                target_class = self._resolve_target_class(logits, target_class)
                if logits.shape[1] == 1:
                    scores = logits.squeeze(1)
                else:
                    scores = logits[torch.arange(x.shape[0]), target_class]                
                scores.sum().backward()
                saliency_accum += x_noisy.grad.detach()
            saliency = saliency_accum / smooth_samples
        else:
            logits = self.forward(x)
            target_class = self._resolve_target_class(logits, target_class)
            if logits.shape[1] == 1:
                scores = logits.squeeze(1)
            else:
                scores = logits[torch.arange(x.shape[0]), target_class]
            scores.sum().backward()
            saliency = x.grad.detach()
        
        if not signed:
            saliency = saliency.abs()
    
        if reduction == "sum" and x.ndim == 3:
            return saliency.sum(dim=-1)
        elif reduction == "mean" and x.ndim == 3:
            return saliency.mean(dim=-1)
        else:
            return saliency
        
    def compute_gradient_x_input(self, x, target_class=None):
        """
        Computes gradient × input attribution.
        """
        x = x.clone().detach().requires_grad_(True)
        logits = self.forward(x)
        target_class = self._resolve_target_class(logits, target_class)
        if logits.shape[1] == 1:
            scores = logits.squeeze(1)
        else:
            scores = logits[torch.arange(x.shape[0]), target_class]
        scores.sum().backward()
        grads = x.grad
        return grads * x

    def compute_integrated_gradients(self, x, target_class=None, steps=50, baseline=None):
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
            logits = self.forward(x)
            if logits.shape[1] == 1:
                target_class = 0  # only one column exists, representing class 1 logit
            else:
                target_class = self._resolve_target_class(logits, target_class)

        if baseline is None:
            baseline = torch.zeros_like(x)

        attributions, delta = ig.attribute(
            x,
            baselines=baseline,
            target=target_class,
            n_steps=steps,
            return_convergence_delta=True
        )
        return attributions, delta

    def compute_deeplift(
        self,
        x,
        baseline=None,
        target_class=None,
        reduction="sum",
        signed=True
    ):
        """
        Compute DeepLIFT scores using captum.

        baseline:
            reference input for DeepLIFT.
        """
        from captum.attr import DeepLift

        self.eval()
        deeplift = DeepLift(self)

        logits = self.forward(x)
        if logits.shape[1] == 1:
            target_class = 0  # only one column exists, representing class 1 logit
        else:
            target_class = self._resolve_target_class(logits, target_class)

        if baseline is None:
            baseline = torch.zeros_like(x)

        attr = deeplift.attribute(x, target=target_class, baselines=baseline)

        if not signed:
            attr = attr.abs()
        
        if reduction == "sum" and x.ndim == 3:
            return attr.sum(dim=-1)
        elif reduction == "mean" and x.ndim == 3:
            return attr.mean(dim=-1)
        else:
            return attr
        
    def compute_occlusion(
        self,
        x,
        target_class=None,
        window_size=5,
        baseline=None
    ):
        """
        Computes per-sample occlusion attribution.
        Supports 2D [B, S] or 3D [B, S, D] inputs.
        Returns: [B, S] occlusion scores
        """
        self.eval()

        x_np = x.detach().cpu().numpy()
        ndim = x_np.ndim
        if ndim == 2:
            B, S = x_np.shape
            D = 1
        elif ndim == 3:
            B, S, D = x_np.shape
        else:
            raise ValueError(f"Unsupported input shape {x_np.shape}")

        # if no baseline provided, fallback to mean
        if baseline is None:
            baseline = np.mean(x_np, axis=0)

        occlusion_scores = np.zeros((B, S))

        for b in range(B):
            for i in range(S):
                x_occluded = x_np[b].copy()
                left = max(0, i - window_size // 2)
                right = min(S, i + window_size // 2)

                if ndim == 2:
                    x_occluded[left:right] = baseline[left:right]
                else:
                    x_occluded[left:right, :] = baseline[left:right, :]

                x_tensor = torch.tensor(
                    x_occluded,
                    device=self.device,
                    dtype=torch.float32
                ).unsqueeze(0)

                logits = self.forward(x_tensor)
                target_class = self._resolve_target_class(logits, target_class)

                if logits.shape[1] == 1:
                    scores = logits.squeeze(1)
                else:
                    scores = logits[torch.arange(x.shape[0]), target_class]

                occlusion_scores[b, i] = scores.mean().item()

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
        normalize=True,
        signed=True
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
                attr = model.compute_saliency(x, target_class=target_class, signed=signed)

            elif method == "smoothgrad":
                attr = model.compute_saliency(x, smoothgrad=True, target_class=target_class, signed=signed)

            elif method == "IG":
                attributions, delta = model.compute_integrated_gradients(
                    x, target_class=target_class, baseline=baseline
                )
                attr = attributions

            elif method == "deeplift":
                attr = model.compute_deeplift(x, baseline=baseline, target_class=target_class, signed=signed)

            elif method == "gradxinput":
                attr = model.compute_gradient_x_input(x, target_class=target_class)

            elif method == "occlusion":
                attr = model.compute_occlusion(
                    x, target_class=target_class, baseline=baseline
                )

            else:
                raise ValueError(f"Unknown method {method}")

            # ensure numpy
            attr = attr.detach().cpu().numpy() if torch.is_tensor(attr) else attr
            results.append(attr)

        results_stacked = np.concatenate(results, axis=0)
        adata.obsm[adata_key] = results_stacked

        if normalize:
            row_max = np.max(np.abs(results_stacked), axis=1, keepdims=True)
            row_max[row_max == 0] = 1  # avoid divide by zero
            results_normalized = results_stacked / row_max
            adata.obsm[adata_key + "_normalized"] = results_normalized

    def _resolve_target_class(self, logits, target_class):
        if target_class is not None:
            return target_class
        if logits.shape[1] == 1:
            return (logits > 0).long().squeeze(1)
        return logits.argmax(dim=1)