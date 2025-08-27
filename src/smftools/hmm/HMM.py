import math
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

def _logsumexp(vec: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    return torch.logsumexp(vec, dim=dim, keepdim=keepdim)

class HMM(nn.Module):
    """
    Vectorized HMM (Bernoulli emissions) implemented in PyTorch.

    Methods:
      - fit(data, ...) -> trains params in-place
      - predict(data, ...) -> list of (L, K) posterior marginals (gamma) numpy arrays
      - viterbi(seq, ...) -> (path_list, score)
      - batch_viterbi(data, ...) -> list of (path_list, score)
      - score(seq_or_list, ...) -> float or list of floats

    Notes:
      - data: list of sequences (each sequence is iterable of {0,1,np.nan}).
      - impute_strategy: "ignore" (NaN treated as missing), "random" (fill NaNs randomly with 0/1).
    """

    def __init__(
        self,
        n_states: int = 2,
        init_start: Optional[List[float]] = None,
        init_trans: Optional[List[List[float]]] = None,
        init_emission: Optional[List[float]] = None,
        dtype: torch.dtype = torch.float64,
        eps: float = 1e-8,
        smf_modality: Optional[str] = None,
    ):
        super().__init__()
        if n_states < 2:
            raise ValueError("n_states must be >= 2")
        self.n_states = n_states
        self.eps = float(eps)
        self.dtype = dtype
        self.smf_modality = smf_modality

        # initialize params (probabilities)
        if init_start is None:
            start = np.full((n_states,), 1.0 / n_states, dtype=float)
        else:
            start = np.asarray(init_start, dtype=float)
        if init_trans is None:
            trans = np.full((n_states, n_states), 1.0 / n_states, dtype=float)
        else:
            trans = np.asarray(init_trans, dtype=float)
        # --- sanitize init_emission so it's a 1-D list of P(obs==1 | state) ---
        if init_emission is None:
            emission = np.full((n_states,), 0.5, dtype=float)
        else:
            em_arr = np.asarray(init_emission, dtype=float)
            # case:  (K,2) -> pick P(obs==1) from second column
            if em_arr.ndim == 2 and em_arr.shape[1] == 2 and em_arr.shape[0] == n_states:
                emission = em_arr[:, 1].astype(float)
            # case: maybe shape (1,K,2) etc. -> try to collapse trailing axis of length 2
            elif em_arr.ndim >= 2 and em_arr.shape[-1] == 2:
                emission = em_arr.reshape(-1, 2)[:n_states, 1].astype(float)
            else:
                emission = em_arr.reshape(-1)[:n_states].astype(float)

        # store as parameters (not trainable via grad; EM updates .data in-place)
        self.start = nn.Parameter(torch.tensor(start, dtype=self.dtype), requires_grad=False)
        self.trans = nn.Parameter(torch.tensor(trans, dtype=self.dtype), requires_grad=False)
        self.emission = nn.Parameter(torch.tensor(emission, dtype=self.dtype), requires_grad=False)

        self._normalize_params()

    def _normalize_params(self):
        with torch.no_grad():
            # coerce shapes
            K = self.n_states
            self.start.data = self.start.data.squeeze()
            if self.start.data.numel() != K:
                self.start.data = torch.full((K,), 1.0 / K, dtype=self.dtype)

            self.trans.data = self.trans.data.squeeze()
            if not (self.trans.data.ndim == 2 and self.trans.data.shape == (K, K)):
                if K == 2:
                    self.trans.data = torch.tensor([[0.9,0.1],[0.1,0.9]], dtype=self.dtype)
                else:
                    self.trans.data = torch.full((K, K), 1.0 / K, dtype=self.dtype)

            self.emission.data = self.emission.data.squeeze()
            if self.emission.data.numel() != K:
                self.emission.data = torch.full((K,), 0.5, dtype=self.dtype)

            # now perform smoothing/normalization
            self.start.data = (self.start.data + self.eps)
            self.start.data = self.start.data / self.start.data.sum()

            self.trans.data = (self.trans.data + self.eps)
            row_sums = self.trans.data.sum(dim=1, keepdim=True)
            row_sums[row_sums == 0.0] = 1.0
            self.trans.data = self.trans.data / row_sums

            self.emission.data = self.emission.data.clamp(min=self.eps, max=1.0 - self.eps)


    def _ensure_device_dtype(self, device: Optional[torch.device]):
        if device is None:
            device = next(self.parameters()).device
        self.start.data = self.start.data.to(device=device, dtype=self.dtype)
        self.trans.data = self.trans.data.to(device=device, dtype=self.dtype)
        self.emission.data = self.emission.data.to(device=device, dtype=self.dtype)
        return device

    @staticmethod
    def _pad_and_mask(
        data: List[List],
        device: torch.device,
        dtype: torch.dtype,
        impute_strategy: str = "ignore",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pads sequences to shape (B, L). Returns (obs, mask, lengths)
        - Accepts: list-of-seqs, or 2D ndarray (B, L).
        - If a sequence element is itself an array (per-timestep feature vector),
          collapse the last axis by mean (warns once).
        """
        import warnings

        # If somebody passed a 2-D ndarray directly, convert to list-of-rows
        if isinstance(data, np.ndarray) and data.ndim == 2:
            # convert rows -> python lists (scalars per timestep)
            data = data.tolist()

        B = len(data)
        lengths = torch.tensor([len(s) for s in data], dtype=torch.long, device=device)
        L = int(lengths.max().item()) if B > 0 else 0
        obs = torch.zeros((B, L), dtype=dtype, device=device)
        mask = torch.zeros((B, L), dtype=torch.bool, device=device)

        warned_collapse = False

        for i, seq in enumerate(data):
            # seq may be list/ndarray of scalars OR list/ndarray of per-timestep arrays
            arr = np.asarray(seq, dtype=float)

            # If arr is shape (L,1,1,...) squeeze trailing singletons
            while arr.ndim > 1 and arr.shape[-1] == 1:
                arr = np.squeeze(arr, axis=-1)

            # If arr is still >1D (e.g., (L, F)), collapse the last axis by mean
            if arr.ndim > 1:
                if not warned_collapse:
                    warnings.warn(
                        "HMM._pad_and_mask: collapsing per-timestep feature axis by mean "
                        "(arr had shape {}). If you prefer a different reduction, "
                        "preprocess your data.".format(arr.shape),
                        stacklevel=2,
                    )
                    warned_collapse = True
                # collapse features -> scalar per timestep
                arr = np.asarray(arr, dtype=float).mean(axis=-1)

            # now arr should be 1D (T,)
            if arr.ndim == 0:
                # single scalar: treat as length-1 sequence
                arr = np.atleast_1d(arr)

            nan_mask = np.isnan(arr)
            if impute_strategy == "random" and nan_mask.any():
                arr[nan_mask] = np.random.choice([0, 1], size=nan_mask.sum())
                local_mask = np.ones_like(arr, dtype=bool)
            else:
                local_mask = ~nan_mask
                arr = np.where(local_mask, arr, 0.0)

            L_i = arr.shape[0]
            obs[i, :L_i] = torch.tensor(arr, dtype=dtype, device=device)
            mask[i, :L_i] = torch.tensor(local_mask, dtype=torch.bool, device=device)

        return obs, mask, lengths

    def _log_emission(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, L)
        mask: (B, L) bool
        returns logB (B, L, K)
        """
        B, L = obs.shape
        p = self.emission  # (K,)
        logp = torch.log(p + self.eps)
        log1mp = torch.log1p(-p + self.eps)
        obs_expand = obs.unsqueeze(-1)  # (B, L, 1)
        logB = obs_expand * logp.unsqueeze(0).unsqueeze(0) + (1.0 - obs_expand) * log1mp.unsqueeze(0).unsqueeze(0)
        logB = torch.where(mask.unsqueeze(-1), logB, torch.zeros_like(logB))
        return logB

    def fit(
        self,
        data: List[List],
        max_iter: int = 100,
        tol: float = 1e-4,
        impute_strategy: str = "ignore",
        verbose: bool = True,
        return_history: bool = False,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Vectorized Baum-Welch EM across a batch of sequences (padded).
        """
        if device is None:
            device = next(self.parameters()).device
        elif isinstance(device, str):
            device = torch.device(device)
        device = self._ensure_device_dtype(device)

        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                # rows are sequences: convert to list of 1D arrays
                data = data.tolist()
            elif data.ndim == 1:
                # single sequence
                data = [data.tolist()]
            else:
                raise ValueError(f"Expected data to be 1D or 2D ndarray; got array with ndim={data.ndim}")

        obs, mask, lengths = self._pad_and_mask(data, device=device, dtype=self.dtype, impute_strategy=impute_strategy)
        B, L = obs.shape
        K = self.n_states
        eps = float(self.eps)

        if verbose:
            print(f"[HMM.fit] device={device}, batch={B}, max_len={L}, states={K}")

        loglik_history = []

        for it in range(1, max_iter + 1):
            if verbose:
                print(f"[HMM.fit] EM iter {it}")

            # compute batched emission logs
            logB = self._log_emission(obs, mask)  # (B, L, K)

            # logs for start and transition
            logA = torch.log(self.trans + eps)  # (K, K)
            logstart = torch.log(self.start + eps)  # (K,)

            # Forward (batched)
            alpha = torch.empty((B, L, K), dtype=self.dtype, device=device)
            alpha[:, 0, :] = logstart.unsqueeze(0) + logB[:, 0, :]  # (B,K)
            for t in range(1, L):
                # prev: (B, i, 1) + (1, i, j) broadcast => (B, i, j)
                prev = alpha[:, t - 1, :].unsqueeze(2) + logA.unsqueeze(0)  # (B, K, K)
                alpha[:, t, :] = _logsumexp(prev, dim=1) + logB[:, t, :]

            # Backward (batched)
            beta = torch.empty((B, L, K), dtype=self.dtype, device=device)
            beta[:, L - 1, :] = torch.zeros((K,), dtype=self.dtype, device=device).unsqueeze(0).expand(B, K)
            for t in range(L - 2, -1, -1):
                # temp (B, i, j) = logA[i,j] + logB[:,t+1,j] + beta[:,t+1,j]
                temp = logA.unsqueeze(0) + (logB[:, t + 1, :].unsqueeze(1) + beta[:, t + 1, :].unsqueeze(1))
                beta[:, t, :] = _logsumexp(temp, dim=2)

            # sequence log-likelihoods (use last real index)
            last_idx = (lengths - 1).clamp(min=0)
            idx_range = torch.arange(B, device=device)
            final_alpha = alpha[idx_range, last_idx, :]  # (B, K)
            seq_loglikes = _logsumexp(final_alpha, dim=1)  # (B,)
            total_loglike = float(seq_loglikes.sum().item())

            # posterior gamma (B, L, K)
            log_gamma = alpha + beta  # (B, L, K)
            logZ_time = _logsumexp(log_gamma, dim=2, keepdim=True)  # (B, L, 1)
            gamma = (log_gamma - logZ_time).exp()  # (B, L, K)

            # accumulators: starts, transitions, emissions
            gamma_start_accum = gamma[:, 0, :].sum(dim=0)  # (K,)

            # emission accumulators: sum over observed positions only
            mask_f = mask.unsqueeze(-1)  # (B, L, 1)
            emit_num = (gamma * obs.unsqueeze(-1) * mask_f).sum(dim=(0, 1))  # (K,)
            emit_den = (gamma * mask_f).sum(dim=(0, 1))  # (K,)

            # transitions: accumulate xi across t for valid positions
            trans_accum = torch.zeros((K, K), dtype=self.dtype, device=device)
            if L >= 2:
                time_idx = torch.arange(L - 1, device=device).unsqueeze(0).expand(B, L - 1)  # (B, L-1)
                valid = time_idx < (lengths.unsqueeze(1) - 1)  # (B, L-1) bool
                for t in range(L - 1):
                    a_t = alpha[:, t, :].unsqueeze(2)  # (B, i, 1)
                    b_next = (logB[:, t + 1, :] + beta[:, t + 1, :]).unsqueeze(1)  # (B, 1, j)
                    log_xi_unnorm = a_t + logA.unsqueeze(0) + b_next  # (B, i, j)
                    log_xi_flat = log_xi_unnorm.view(B, -1)  # (B, i*j)
                    log_norm = _logsumexp(log_xi_flat, dim=1).unsqueeze(1).unsqueeze(2)  # (B,1,1)
                    xi = (log_xi_unnorm - log_norm).exp()  # (B,i,j)
                    valid_t = valid[:, t].float().unsqueeze(1).unsqueeze(2)  # (B,1,1)
                    xi_masked = xi * valid_t
                    trans_accum += xi_masked.sum(dim=0)  # (i,j)

            # M-step: update parameters with smoothing
            with torch.no_grad():
                new_start = gamma_start_accum + eps
                new_start = new_start / new_start.sum()

                new_trans = trans_accum + eps
                row_sums = new_trans.sum(dim=1, keepdim=True)
                row_sums[row_sums == 0.0] = 1.0
                new_trans = new_trans / row_sums

                new_emission = (emit_num + eps) / (emit_den + 2.0 * eps)
                new_emission = new_emission.clamp(min=eps, max=1.0 - eps)

                self.start.data = new_start
                self.trans.data = new_trans
                self.emission.data = new_emission

            loglik_history.append(total_loglike)
            if verbose:
                print(f"  total loglik = {total_loglike:.6f}")

            if len(loglik_history) > 1 and abs(loglik_history[-1] - loglik_history[-2]) < tol:
                if verbose:
                    print(f"[HMM.fit] converged (Δll < {tol}) at iter {it}")
                break

        return loglik_history if return_history else None
    
    def get_params(self) -> dict:
        """
        Return model parameters as numpy arrays on CPU.
        """
        with torch.no_grad():
            return {
                "n_states": int(self.n_states),
                "start": self.start.detach().cpu().numpy().astype(float).reshape(-1),
                "trans": self.trans.detach().cpu().numpy().astype(float),
                "emission": self.emission.detach().cpu().numpy().astype(float).reshape(-1),
            }

    def print_params(self, decimals: int = 4):
        """
        Nicely print start, transition, and emission probabilities.
        """
        params = self.get_params()
        K = params["n_states"]
        fmt = f"{{:.{decimals}f}}"
        print(f"HMM params (K={K} states):")
        print(" start probs:")
        print("  [" + ", ".join(fmt.format(v) for v in params["start"]) + "]")
        print(" transition matrix (rows = from-state, cols = to-state):")
        for i, row in enumerate(params["trans"]):
            print("  s{:d}: [".format(i) + ", ".join(fmt.format(v) for v in row) + "]")
        print(" emission P(obs==1 | state):")
        for i, v in enumerate(params["emission"]):
            print(f"  s{i}: {fmt.format(v)}")

    def to_dataframes(self) -> dict:
        """
        Return pandas DataFrames for start (Series), trans (DataFrame), emission (Series).
        """
        p = self.get_params()
        K = p["n_states"]
        state_names = [f"state_{i}" for i in range(K)]
        start_s = pd.Series(p["start"], index=state_names, name="start_prob")
        trans_df = pd.DataFrame(p["trans"], index=state_names, columns=state_names)
        emission_s = pd.Series(p["emission"], index=state_names, name="p_obs1")
        return {"start": start_s, "trans": trans_df, "emission": emission_s}

    def predict(self, data: List[List], impute_strategy: str = "ignore", device: Optional[Union[torch.device, str]] = None) -> List[np.ndarray]:
        """
        Return posterior marginals gamma_t(k) for each sequence as list of (L, K) numpy arrays.
        """
        if device is None:
            device = next(self.parameters()).device
        elif isinstance(device, str):
            device = torch.device(device)
        device = self._ensure_device_dtype(device)

        obs, mask, lengths = self._pad_and_mask(data, device=device, dtype=self.dtype, impute_strategy=impute_strategy)
        B, L = obs.shape
        K = self.n_states
        eps = float(self.eps)

        logB = self._log_emission(obs, mask)  # (B, L, K)
        logA = torch.log(self.trans + eps)
        logstart = torch.log(self.start + eps)

        # Forward
        alpha = torch.empty((B, L, K), dtype=self.dtype, device=device)
        alpha[:, 0, :] = logstart.unsqueeze(0) + logB[:, 0, :]
        for t in range(1, L):
            prev = alpha[:, t - 1, :].unsqueeze(2) + logA.unsqueeze(0)
            alpha[:, t, :] = _logsumexp(prev, dim=1) + logB[:, t, :]

        # Backward
        beta = torch.empty((B, L, K), dtype=self.dtype, device=device)
        beta[:, L - 1, :] = torch.zeros((K,), dtype=self.dtype, device=device).unsqueeze(0).expand(B, K)
        for t in range(L - 2, -1, -1):
            temp = logA.unsqueeze(0) + (logB[:, t + 1, :].unsqueeze(1) + beta[:, t + 1, :].unsqueeze(1))
            beta[:, t, :] = _logsumexp(temp, dim=2)

        # gamma
        log_gamma = alpha + beta
        logZ_time = _logsumexp(log_gamma, dim=2, keepdim=True)
        gamma = (log_gamma - logZ_time).exp()  # (B, L, K)

        results = []
        for i in range(B):
            L_i = int(lengths[i].item())
            results.append(gamma[i, :L_i, :].detach().cpu().numpy())
        return results

    def score(self, seq_or_list: Union[List[float], List[List[float]]], impute_strategy: str = "ignore", device: Optional[Union[torch.device, str]] = None) -> Union[float, List[float]]:
        """
        Compute log-likelihood of a single sequence or list of sequences under current params.
        Returns float (single) or list of floats (batch).
        """
        single = False
        if not isinstance(seq_or_list[0], (list, tuple, np.ndarray)):
            seqs = [seq_or_list]
            single = True
        else:
            seqs = seq_or_list

        if device is None:
            device = next(self.parameters()).device
        elif isinstance(device, str):
            device = torch.device(device)
        device = self._ensure_device_dtype(device)

        obs, mask, lengths = self._pad_and_mask(seqs, device=device, dtype=self.dtype, impute_strategy=impute_strategy)
        B, L = obs.shape
        K = self.n_states
        eps = float(self.eps)

        logB = self._log_emission(obs, mask)
        logA = torch.log(self.trans + eps)
        logstart = torch.log(self.start + eps)

        alpha = torch.empty((B, L, K), dtype=self.dtype, device=device)
        alpha[:, 0, :] = logstart.unsqueeze(0) + logB[:, 0, :]
        for t in range(1, L):
            prev = alpha[:, t - 1, :].unsqueeze(2) + logA.unsqueeze(0)
            alpha[:, t, :] = _logsumexp(prev, dim=1) + logB[:, t, :]

        last_idx = (lengths - 1).clamp(min=0)
        idx_range = torch.arange(B, device=device)
        final_alpha = alpha[idx_range, last_idx, :]  # (B, K)
        seq_loglikes = _logsumexp(final_alpha, dim=1)  # (B,)
        seq_loglikes = seq_loglikes.detach().cpu().numpy().tolist()
        return seq_loglikes[0] if single else seq_loglikes

    def viterbi(self, seq: List[float], impute_strategy: str = "ignore", device: Optional[Union[torch.device, str]] = None) -> Tuple[List[int], float]:
        """
        Viterbi decode a single sequence. Returns (state_path, log_probability_of_path).
        """
        paths, scores = self.batch_viterbi([seq], impute_strategy=impute_strategy, device=device)
        return paths[0], scores[0]

    def batch_viterbi(self, data: List[List[float]], impute_strategy: str = "ignore", device: Optional[Union[torch.device, str]] = None) -> Tuple[List[List[int]], List[float]]:
        """
        Batched Viterbi decoding on padded sequences. Returns (list_of_paths, list_of_scores).
        Each path is the length of the original sequence.
        """
        if device is None:
            device = next(self.parameters()).device
        elif isinstance(device, str):
            device = torch.device(device)
        device = self._ensure_device_dtype(device)

        obs, mask, lengths = self._pad_and_mask(data, device=device, dtype=self.dtype, impute_strategy=impute_strategy)
        B, L = obs.shape
        K = self.n_states
        eps = float(self.eps)

        p = self.emission
        logp = torch.log(p + eps)
        log1mp = torch.log1p(-p + eps)
        logB = obs.unsqueeze(-1) * logp.unsqueeze(0).unsqueeze(0) + (1.0 - obs.unsqueeze(-1)) * log1mp.unsqueeze(0).unsqueeze(0)
        logB = torch.where(mask.unsqueeze(-1), logB, torch.zeros_like(logB))

        logstart = torch.log(self.start + eps)
        logA = torch.log(self.trans + eps)

        # delta (score) and psi (argmax pointers)
        delta = torch.empty((B, L, K), dtype=self.dtype, device=device)
        psi = torch.zeros((B, L, K), dtype=torch.long, device=device)

        delta[:, 0, :] = logstart.unsqueeze(0) + logB[:, 0, :]
        psi[:, 0, :] = -1  # sentinel

        for t in range(1, L):
            # cand shape (B, i, j)
            cand = delta[:, t - 1, :].unsqueeze(2) + logA.unsqueeze(0)  # (B, K, K)
            best_val, best_idx = cand.max(dim=1)  # best over previous i: results (B, j)
            delta[:, t, :] = best_val + logB[:, t, :]
            psi[:, t, :] = best_idx  # best previous state index for each (B, j)

        # backtrack
        last_idx = (lengths - 1).clamp(min=0)
        idx_range = torch.arange(B, device=device)
        final_delta = delta[idx_range, last_idx, :]  # (B, K)
        best_last_val, best_last_state = final_delta.max(dim=1)  # (B,), (B,)
        paths = []
        scores = []
        for b in range(B):
            Lb = int(lengths[b].item())
            if Lb == 0:
                paths.append([])
                scores.append(float("-inf"))
                continue
            s = int(best_last_state[b].item())
            path = [s]
            for t in range(Lb - 1, 0, -1):
                s = int(psi[b, t, s].item())
                path.append(s)
            path.reverse()
            paths.append(path)
            scores.append(float(best_last_val[b].item()))
        return paths, scores
    
    def save(self, path: str) -> None:
        """
        Save HMM to `path` using torch.save. Stores:
        - n_states, eps, dtype (string)
        - start, trans, emission (CPU tensors)
        """
        payload = {
            "n_states": int(self.n_states),
            "eps": float(self.eps),
            # store dtype as a string like "torch.float64" (portable)
            "dtype": str(self.dtype),
            "start": self.start.detach().cpu(),
            "trans": self.trans.detach().cpu(),
            "emission": self.emission.detach().cpu(),
        }
        torch.save(payload, path)


    @classmethod
    def load(cls, path: str, device: Optional[Union[torch.device, str]] = None) -> "HMM":
        """
        Load model from `path`. If `device` is provided (str or torch.device),
        parameters will be moved to that device; otherwise they remain on CPU.
        Example: model = HMM.load('hmm.pt', device='cuda')
        """
        payload = torch.load(path, map_location="cpu")

        n_states = int(payload.get("n_states"))
        eps = float(payload.get("eps", 1e-8))
        dtype_entry = payload.get("dtype", "torch.float64")

        # Resolve dtype string robustly:
        # Accept "torch.float64" or "float64" or actual torch.dtype (older payloads)
        if isinstance(dtype_entry, torch.dtype):
            torch_dtype = dtype_entry
        else:
            # dtype_entry expected to be a string
            dtype_str = str(dtype_entry)
            # take last part after dot if present: "torch.float64" -> "float64"
            name = dtype_str.split(".")[-1]
            # map to torch dtype if available, else fallback mapping
            if hasattr(torch, name):
                torch_dtype = getattr(torch, name)
            else:
                fallback = {"float64": torch.float64, "float32": torch.float32, "float16": torch.float16}
                torch_dtype = fallback.get(name, torch.float64)

        # Build instance (use resolved dtype)
        model = cls(n_states=n_states, dtype=torch_dtype, eps=eps)

        # Determine target device
        if device is None:
            device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        # Load params (they were saved on CPU) and cast to model dtype/device
        with torch.no_grad():
            model.start.data = payload["start"].to(device=device, dtype=model.dtype)
            model.trans.data = payload["trans"].to(device=device, dtype=model.dtype)
            model.emission.data = payload["emission"].to(device=device, dtype=model.dtype)

        # Normalize / coerce shapes just in case
        model._normalize_params()
        return model
        
    def annotate_adata(
        self,
        adata,
        obs_column: str,
        layer: Optional[str] = None,
        footprints: bool = True,
        accessible_patches: bool = False,
        cpg: bool = False,
        methbases: Optional[List[str]] = None,
        threshold: float = 0.7,
        device: Optional[Union[str, torch.device]] = None,
        batch_size: int = 1024,
        use_viterbi: bool = False,
        in_place: bool = True,
        verbose: bool = True,
        uns_key: str = "hmm_appended_layers",
    ):
        """
        Annotate an AnnData with HMM-derived features (in adata.obs and adata.layers).

        Parameters are similar to the standalone helper:
        - adata: AnnData object
        - obs_column: categorical obs column grouping reads (e.g. 'Reference_strand')
        - layer: optional layer name to read sequence matrix from; else use .X
        - footprints, accessible_patches, cpg: which feature groups to compute
        - methbases: list of methbases to test (defaults ["GpC","CpG","A","C"])
        - threshold: minimum probability to accept a region
        - device: device string or torch.device (defaults to model's device)
        - batch_size: number of reads to predict per chunk
        - use_viterbi: if True use Viterbi paths (batch_viterbi), otherwise argmax on posteriors
        - in_place: if False returns a copy, else modifies adata in-place
        - verbose: show progress bars
        """
        import numpy as _np
        import torch as _torch
        from tqdm import trange, tqdm as _tqdm

        if methbases is None:
            methbases = ["GpC", "CpG", "A", "C"]

        # Feature definitions
        feature_sets = {}
        if footprints:
            feature_sets["footprint"] = {
                "features": {
                    "small_bound_stretch": [0, 20],
                    "medium_bound_stretch": [20, 50],
                    "putative_nucleosome": [50, 200],
                    "large_bound_stretch": [200, _np.inf],
                },
                "state": "Non-Modified",
            }
        if accessible_patches:
            feature_sets["accessible"] = {
                "features": {
                    "small_accessible_patch": [0, 20],
                    "mid_accessible_patch": [20, 80],
                    "large_accessible_patch": [80, _np.inf],
                },
                "state": "Modified",
            }
        if cpg:
            feature_sets["cpg"] = {"features": {"cpg_patch": [0, _np.inf]}, "state": "Modified"}

        # copy vs in-place
        if not in_place:
            adata = adata.copy()

        # prepare column names
        all_features = []
        combined_prefix = "Combined"
        for key, fs in feature_sets.items():
            if key == "cpg":
                all_features += [f"CpG_{f}" for f in fs["features"]]
                all_features.append(f"CpG_all_{key}_features")
            else:
                for methbase in methbases:
                    all_features += [f"{methbase}_{f}" for f in fs["features"]]
                    all_features.append(f"{methbase}_all_{key}_features")
                if len(methbases) > 1:
                    all_features += [f"{combined_prefix}_{f}" for f in fs["features"]]
                    all_features.append(f"{combined_prefix}_all_{key}_features")

        # initialize obs columns (unique lists per row)
        n_rows = adata.shape[0]
        for feature in all_features:
            adata.obs[feature] = [[] for _ in range(n_rows)]
            adata.obs[f"{feature}_distances"] = [None] * n_rows
            adata.obs[f"n_{feature}"] = -1

        # keep track of which layers we actually append (so we can write to .uns)
        appended_layers: List[str] = []

        # device management
        if device is None:
            device = next(self.parameters()).device
        elif isinstance(device, str):
            device = _torch.device(device)
        self.to(device)

        # helpers ---------------------------------------------------------------
        def _ensure_2d_array_like(matrix):
            arr = _np.asarray(matrix)
            if arr.ndim == 1:
                arr = arr[_np.newaxis, :]
            elif arr.ndim > 2:
                # squeeze trailing singletons
                while arr.ndim > 2 and arr.shape[-1] == 1:
                    arr = _np.squeeze(arr, axis=-1)
                if arr.ndim != 2:
                    raise ValueError(f"Expected 2D sequence matrix; got array with shape {arr.shape}")
            return arr

        def calculate_batch_distances(intervals_list, threshold_local=0.9):
            results_local = []
            for intervals in intervals_list:
                if not isinstance(intervals, list) or len(intervals) == 0:
                    results_local.append([])
                    continue
                valid = [iv for iv in intervals if iv[2] > threshold_local]
                if len(valid) <= 1:
                    results_local.append([])
                    continue
                valid = sorted(valid, key=lambda x: x[0])
                dists = [(valid[i + 1][0] - (valid[i][0] + valid[i][1])) for i in range(len(valid) - 1)]
                results_local.append(dists)
            return results_local

        def classify_batch_local(predicted_states_batch, probabilities_batch, coordinates, classification_mapping, target_state="Modified"):
            # Accept numpy arrays or torch tensors
            if isinstance(predicted_states_batch, _torch.Tensor):
                pred_np = predicted_states_batch.detach().cpu().numpy()
            else:
                pred_np = _np.asarray(predicted_states_batch)
            if isinstance(probabilities_batch, _torch.Tensor):
                probs_np = probabilities_batch.detach().cpu().numpy()
            else:
                probs_np = _np.asarray(probabilities_batch)

            batch_size, L = pred_np.shape
            all_classifications_local = []
            state_labels = ["Non-Modified", "Modified"]
            try:
                target_idx = state_labels.index(target_state)
            except ValueError:
                target_idx = 1  # fallback

            for b in range(batch_size):
                predicted_states = pred_np[b]
                probabilities = probs_np[b]
                regions = []
                current_start, current_length, current_probs = None, 0, []
                for i, state_index in enumerate(predicted_states):
                    state_prob = float(probabilities[i][state_index])
                    if state_index == target_idx:
                        if current_start is None:
                            current_start = i
                        current_length += 1
                        current_probs.append(state_prob)
                    elif current_start is not None:
                        regions.append((current_start, current_length, float(_np.mean(current_probs))))
                        current_start, current_length, current_probs = None, 0, []
                if current_start is not None:
                    regions.append((current_start, current_length, float(_np.mean(current_probs))))

                final = []
                for start, length, prob in regions:
                    # compute genomic length try/catch
                    try:
                        # compute genomic length in same coordinate system as coordinates[]
                        feature_length = int(coordinates[start + length - 1]) - int(coordinates[start]) + 1
                    except Exception:
                        feature_length = int(length)

                    label = next((ftype for ftype, rng in classification_mapping.items() if rng[0] <= feature_length < rng[1]),
                                list(classification_mapping.keys())[0])

                    # Store the reported start coordinate in the same coordinate system as `coordinates`.
                    # DON'T add a +1 offset here; finalization expects coordinates in the same base.
                    try:
                        genomic_start = int(coordinates[start])
                    except Exception:
                        # If coordinates could not be interpreted to ints, fall back to storing index
                        genomic_start = int(start)
                    final.append((genomic_start, feature_length, label, prob))
                all_classifications_local.append(final)
            return all_classifications_local

        # -----------------------------------------------------------------------

        references = adata.obs[obs_column].cat.categories
        ref_iter = references if not verbose else _tqdm(references, desc="Processing References")
        for ref in ref_iter:
            # subset reads with this obs_column value
            ref_mask = adata.obs[obs_column] == ref
            ref_subset = adata[ref_mask].copy()

            combined_mask = None
            # per-methbase processing
            for methbase in methbases:
                key_lower = methbase.lower()
                pos_mask = {
                    "a": ref_subset.var[f"{ref}_strand_FASTA_base"] == "A",
                    "c": ref_subset.var[f"{ref}_any_C_site"] == True,
                    "gpc": ref_subset.var[f"{ref}_GpC_site"] == True,
                    "cpg": ref_subset.var[f"{ref}_CpG_site"] == True,
                }.get(key_lower, None)

                if pos_mask is None:
                    continue
                combined_mask = pos_mask if combined_mask is None else (combined_mask | pos_mask)

                if pos_mask.sum() == 0:
                    continue

                sub = ref_subset[:, pos_mask]
                # choose matrix
                matrix = sub.layers[layer] if (layer and layer in sub.layers) else sub.X
                matrix = _ensure_2d_array_like(matrix)
                n_reads = matrix.shape[0]

                # coordinates for this sub (try to convert to ints, else fallback to indices)
                try:
                    coords = _np.asarray(sub.var_names, dtype=int)
                except Exception:
                    coords = _np.arange(sub.shape[1], dtype=int)

                # chunked processing
                chunk_iter = range(0, n_reads, batch_size)
                if verbose:
                    chunk_iter = _tqdm(list(chunk_iter), desc=f"{ref}:{methbase} chunks")
                for start_idx in chunk_iter:
                    stop_idx = min(n_reads, start_idx + batch_size)
                    chunk = matrix[start_idx:stop_idx]
                    seqs = chunk.tolist()
                    # posterior marginals
                    gammas = self.predict(seqs, impute_strategy="ignore", device=device)
                    if len(gammas) == 0:
                        continue
                    probs_batch = _np.stack(gammas, axis=0)  # (B, L, K)
                    if use_viterbi:
                        paths, _scores = self.batch_viterbi(seqs, impute_strategy="ignore", device=device)
                        pred_states = _np.asarray(paths)
                    else:
                        pred_states = _np.argmax(probs_batch, axis=2)

                    classifications = classify_batch_local(pred_states, probs_batch, coords, feature_sets.get("footprint", {}).get("features", {}) if False else {}, target_state="Modified")  # placeholder not used

                    # For each feature group, classify separately and write back
                    for key, fs in feature_sets.items():
                        if key == "cpg":
                            continue
                        state_target = fs["state"]
                        feature_map = fs["features"]
                        classifications = classify_batch_local(pred_states, probs_batch, coords, feature_map, target_state=state_target)

                        # write results to adata.obs rows (use original index names)
                        row_indices = list(sub.obs.index[start_idx:stop_idx])
                        for i_local, idx in enumerate(row_indices):
                            for start, length, label, prob in classifications[i_local]:
                                adata.obs.at[idx, f"{methbase}_{label}"].append([start, length, prob])
                                adata.obs.at[idx, f"{methbase}_all_{key}_features"].append([start, length, prob])

            # Combined subset (if multiple methbases)
            if len(methbases) > 1 and (combined_mask is not None) and (combined_mask.sum() > 0):
                comb = ref_subset[:, combined_mask]
                if comb.shape[1] > 0:
                    matrix = comb.layers[layer] if (layer and layer in comb.layers) else comb.X
                    matrix = _ensure_2d_array_like(matrix)
                    n_reads_comb = matrix.shape[0]
                    try:
                        coords_comb = _np.asarray(comb.var_names, dtype=int)
                    except Exception:
                        coords_comb = _np.arange(comb.shape[1], dtype=int)

                    chunk_iter = range(0, n_reads_comb, batch_size)
                    if verbose:
                        chunk_iter = _tqdm(list(chunk_iter), desc=f"{ref}:Combined chunks")
                    for start_idx in chunk_iter:
                        stop_idx = min(n_reads_comb, start_idx + batch_size)
                        chunk = matrix[start_idx:stop_idx]
                        seqs = chunk.tolist()
                        gammas = self.predict(seqs, impute_strategy="ignore", device=device)
                        if len(gammas) == 0:
                            continue
                        probs_batch = _np.stack(gammas, axis=0)
                        if use_viterbi:
                            paths, _scores = self.batch_viterbi(seqs, impute_strategy="ignore", device=device)
                            pred_states = _np.asarray(paths)
                        else:
                            pred_states = _np.argmax(probs_batch, axis=2)

                        for key, fs in feature_sets.items():
                            if key == "cpg":
                                continue
                            state_target = fs["state"]
                            feature_map = fs["features"]
                            classifications = classify_batch_local(pred_states, probs_batch, coords_comb, feature_map, target_state=state_target)
                            row_indices = list(comb.obs.index[start_idx:stop_idx])
                            for i_local, idx in enumerate(row_indices):
                                for start, length, label, prob in classifications[i_local]:
                                    adata.obs.at[idx, f"{combined_prefix}_{label}"].append([start, length, prob])
                                    adata.obs.at[idx, f"{combined_prefix}_all_{key}_features"].append([start, length, prob])

        # CpG special handling
        if cpg:
            cpg_iter = references if not verbose else _tqdm(references, desc="Processing CpG")
            for ref in cpg_iter:
                ref_mask = adata.obs[obs_column] == ref
                ref_subset = adata[ref_mask].copy()
                pos_mask = ref_subset.var[f"{ref}_CpG_site"] == True
                if pos_mask.sum() == 0:
                    continue
                cpg_sub = ref_subset[:, pos_mask]
                matrix = cpg_sub.layers[layer] if (layer and layer in cpg_sub.layers) else cpg_sub.X
                matrix = _ensure_2d_array_like(matrix)
                n_reads = matrix.shape[0]
                try:
                    coords_cpg = _np.asarray(cpg_sub.var_names, dtype=int)
                except Exception:
                    coords_cpg = _np.arange(cpg_sub.shape[1], dtype=int)

                chunk_iter = range(0, n_reads, batch_size)
                if verbose:
                    chunk_iter = _tqdm(list(chunk_iter), desc=f"{ref}:CpG chunks")
                for start_idx in chunk_iter:
                    stop_idx = min(n_reads, start_idx + batch_size)
                    chunk = matrix[start_idx:stop_idx]
                    seqs = chunk.tolist()
                    gammas = self.predict(seqs, impute_strategy="ignore", device=device)
                    if len(gammas) == 0:
                        continue
                    probs_batch = _np.stack(gammas, axis=0)
                    if use_viterbi:
                        paths, _scores = self.batch_viterbi(seqs, impute_strategy="ignore", device=device)
                        pred_states = _np.asarray(paths)
                    else:
                        pred_states = _np.argmax(probs_batch, axis=2)

                    fs = feature_sets["cpg"]
                    state_target = fs["state"]
                    feature_map = fs["features"]
                    classifications = classify_batch_local(pred_states, probs_batch, coords_cpg, feature_map, target_state=state_target)
                    row_indices = list(cpg_sub.obs.index[start_idx:stop_idx])
                    for i_local, idx in enumerate(row_indices):
                        for start, length, label, prob in classifications[i_local]:
                            adata.obs.at[idx, f"CpG_{label}"].append([start, length, prob])
                            adata.obs.at[idx, f"CpG_all_cpg_features"].append([start, length, prob])

        try:
            # try to interpret var_names as integers (genomic coordinates)
            coordinates = _np.asarray(adata.var_names, dtype=int)
            coords_are_ints = True
        except Exception:
            # fallback: treat var positions as simple indices 0..n_vars-1
            coordinates = _np.arange(adata.shape[1], dtype=int)
            coords_are_ints = False

        features_iter = all_features if not verbose else _tqdm(all_features, desc="Finalizing Layers")
        for feature in features_iter:
            bin_matrix = _np.zeros((adata.shape[0], adata.shape[1]), dtype=int)
            counts = _np.zeros(adata.shape[0], dtype=int)
            for row_idx, intervals in enumerate(adata.obs[feature]):
                if not isinstance(intervals, list):
                    intervals = []
                for start, length, prob in intervals:
                    if prob > threshold:
                        # If coordinates are ints, we treat stored 'start' as a genomic coordinate.
                        # Otherwise assume stored 'start' is an index (0-based).
                        if coords_are_ints:
                            # start is genomic coordinate (e.g. bp). Find matching / nearest indices.
                            start_idx = _np.searchsorted(coordinates, int(start), side="left")
                            end_idx = _np.searchsorted(coordinates, int(start) + int(length) - 1, side="right")
                        else:
                            # fallback: interpret 'start' as an index
                            start_idx = int(start)
                            end_idx = start_idx + int(length)

                        # bounds check
                        start_idx = max(0, min(start_idx, adata.shape[1]))
                        end_idx = max(0, min(end_idx, adata.shape[1]))

                        if start_idx < end_idx:
                            bin_matrix[row_idx, start_idx:end_idx] = 1
                            counts[row_idx] += 1
                        else:
                            # optional: debug/diagnostic hook
                            # warnings.warn(f"Feature {feature} row {row_idx} interval ({start},{length}) mapped to empty slice [{start_idx},{end_idx})")
                            pass
            # write layer and track name
            adata.layers[feature] = bin_matrix
            appended_layers.append(feature)

            adata.obs[f"n_{feature}"] = counts
            adata.obs[f"{feature}_distances"] = calculate_batch_distances(adata.obs[feature].tolist(), threshold)

        # Merge appended_layers into adata.uns[uns_key] (preserve pre-existing and avoid duplicates)
        existing = list(adata.uns.get(uns_key, [])) if adata.uns.get(uns_key) is not None else []
        # preserve insertion order, avoid duplicates
        new_list = existing + [l for l in appended_layers if l not in existing]
        adata.uns[uns_key] = new_list

        # return or in-place behavior (unchanged)
        return None if in_place else adata

    def _ensure_final_layer_and_assign(self, final_adata, layer_name: str, subset_idx_mask: np.ndarray, sub_data):
        """
        Ensure final_adata.layers[layer_name] exists and assign rows corresponding to subset_idx_mask
        sub_data has shape (n_subset_rows, n_vars).
        subset_idx_mask: boolean array of length final_adata.n_obs with True where rows belong to subset.
        """
        from scipy.sparse import issparse, csr_matrix
        import warnings

        n_final_obs, n_vars = final_adata.shape
        n_sub_rows = int(subset_idx_mask.sum())

        # prepare row indices in final_adata
        final_row_indices = np.nonzero(subset_idx_mask)[0]

        # if sub_data is sparse, work with sparse
        if issparse(sub_data):
            sub_csr = sub_data.tocsr()
            # if final layer not present, create sparse CSR with zero rows and same n_vars
            if layer_name not in final_adata.layers:
                # create an empty CSR of shape (n_final_obs, n_vars)
                final_adata.layers[layer_name] = csr_matrix((n_final_obs, n_vars), dtype=sub_csr.dtype)
            final_csr = final_adata.layers[layer_name]
            if not issparse(final_csr):
                # convert dense final to sparse first
                final_csr = csr_matrix(final_csr)
            # replace the block of rows: easiest is to build a new csr by stacking pieces
            # (efficient for moderate sizes; for huge data you might want an in-place approach)
            # Build list of blocks: rows before, the subset rows (from final where mask False -> zeros), rows after
            # We'll convert final to LIL for row assignment (mutable), then back to CSR.
            final_lil = final_csr.tolil()
            for i_local, r in enumerate(final_row_indices):
                final_lil.rows[r] = sub_csr.getrow(i_local).indices.tolist()
                final_lil.data[r] = sub_csr.getrow(i_local).data.tolist()
            final_csr = final_lil.tocsr()
            final_adata.layers[layer_name] = final_csr
        else:
            # dense numpy array
            sub_arr = np.asarray(sub_data)
            if sub_arr.shape[0] != n_sub_rows:
                raise ValueError(f"Sub data rows ({sub_arr.shape[0]}) != mask selected rows ({n_sub_rows})")
            if layer_name not in final_adata.layers:
                # create zero array with small dtype
                final_adata.layers[layer_name] = np.zeros((n_final_obs, n_vars), dtype=sub_arr.dtype)
            final_arr = final_adata.layers[layer_name]
            if issparse(final_arr):
                # convert sparse final to dense (or convert sub to sparse); we'll convert final to dense here
                final_arr = final_arr.toarray()
            # assign
            final_arr[final_row_indices, :] = sub_arr
            final_adata.layers[layer_name] = final_arr