import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score, confusion_matrix, roc_curve
)

class SklearnModelWrapper:
    """
    Unified sklearn wrapper matching TorchClassifierWrapper interface.
    """
    def __init__(
        self, 
        model, 
        label_col: str,
        num_classes: int, 
        class_names=None, 
        focus_class: int=1,
        enforce_eval_balance: bool=False,
        target_eval_freq: float=0.3,
        max_eval_positive=None
    ):
        self.model = model
        self.label_col = label_col
        self.num_classes = num_classes
        self.class_names = class_names
        self.focus_class = self._resolve_focus_class(focus_class)
        self.focus_class_name = focus_class
        self.enforce_eval_balance = enforce_eval_balance
        self.target_eval_freq = target_eval_freq
        self.max_eval_positive = max_eval_positive
        self.metrics = {}

    def _resolve_focus_class(self, focus_class):
        if isinstance(focus_class, int):
            return focus_class
        elif isinstance(focus_class, str):
            if self.class_names is None:
                raise ValueError("class_names must be provided if focus_class is a string.")
            if focus_class not in self.class_names:
                raise ValueError(f"focus_class '{focus_class}' not found in class_names {self.class_names}.")
            return self.class_names.index(focus_class)
        else:
            raise ValueError(f"focus_class must be int or str, got {type(focus_class)}")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def _subsample_for_fixed_positive_frequency(self, y_true):
        pos_idx = np.where(y_true == self.focus_class)[0]
        neg_idx = np.where(y_true != self.focus_class)[0]

        max_neg = len(neg_idx)
        max_pos = len(pos_idx)
        max_possible_freq = max_pos / (max_pos + max_neg)

        target_freq = min(self.target_eval_freq, max_possible_freq)
        num_pos_target = min(int(target_freq * max_neg / (1 - target_freq)), max_pos)
        num_neg_target = int(num_pos_target * (1 - target_freq) / target_freq)
        num_neg_target = min(num_neg_target, max_neg)

        if self.max_eval_positive is not None:
            num_pos_target = min(num_pos_target, self.max_eval_positive)

        pos_sampled = np.random.choice(pos_idx, size=num_pos_target, replace=False)
        neg_sampled = np.random.choice(neg_idx, size=num_neg_target, replace=False)

        sampled_idx = np.concatenate([pos_sampled, neg_sampled])
        np.random.shuffle(sampled_idx)

        return sampled_idx

    def evaluate(self, X, y, prefix="test"):
        y_true = y
        y_prob = self.predict_proba(X)
        y_pred = self.predict(X)

        if self.enforce_eval_balance:
            sampled_idx = self._subsample_for_fixed_positive_frequency(y_true)
            y_true = y_true[sampled_idx]
            y_prob = y_prob[sampled_idx]
            y_pred = y_pred[sampled_idx]

        binary_focus = (y_true == self.focus_class).astype(int)
        num_pos = binary_focus.sum()

        is_binary = self.num_classes == 2

        if is_binary:
            if self.focus_class == 1:
                focus_probs = y_prob[:, 1]
            else:
                focus_probs = y_prob[:, 0]
            preds_focus = (y_pred == self.focus_class).astype(int)
        else:
            focus_probs = y_prob[:, self.focus_class]
            preds_focus = (y_pred == self.focus_class).astype(int)

        f1 = f1_score(binary_focus, preds_focus)
        roc_auc = roc_auc_score(binary_focus, focus_probs)
        pr, rc, _ = precision_recall_curve(binary_focus, focus_probs)
        pr_auc = auc(rc, pr)
        pos_freq = binary_focus.mean()
        pr_auc_norm = pr_auc / pos_freq if pos_freq > 0 else np.nan
        fpr, tpr, _ = roc_curve(binary_focus, focus_probs)
        cm = confusion_matrix(y_true, y_pred)
        acc = np.mean(y_pred == y_true)

        # store metrics as attributes for plotting later
        setattr(self, f"{prefix}_f1", f1)
        setattr(self, f"{prefix}_roc_curve", (fpr, tpr))
        setattr(self, f"{prefix}_pr_curve", (rc, pr))
        setattr(self, f"{prefix}_roc_auc", roc_auc)
        setattr(self, f"{prefix}_pr_auc", pr_auc)
        setattr(self, f"{prefix}_pos_freq", pos_freq)
        setattr(self, f"{prefix}_num_pos", num_pos)
        setattr(self, f"{prefix}_confusion_matrix", cm)
        setattr(self, f"{prefix}_acc", acc)

        # also store a metrics dict
        self.metrics = {
            f"{prefix}_acc": acc,
            f"{prefix}_f1": f1,
            f"{prefix}_auc": roc_auc,
            f"{prefix}_pr_auc": pr_auc,
            f"{prefix}_pr_auc_norm": pr_auc_norm,
            f"{prefix}_pos_freq": pos_freq,
            f"{prefix}_num_pos": num_pos
        }

        return self.metrics

    def plot_roc_pr_curves(self, prefix="test"):
        plt.figure(figsize=(12, 5))

        fpr, tpr = getattr(self, f"{prefix}_roc_curve")
        roc_auc = getattr(self, f"{prefix}_roc_auc")
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.ylim(0, 1.05)
        plt.title(f"ROC Curve - {getattr(self, f'{prefix}_num_pos')} positives")
        plt.legend()

        rc, pr = getattr(self, f"{prefix}_pr_curve")
        pr_auc = getattr(self, f"{prefix}_pr_auc")
        pos_freq = getattr(self, f"{prefix}_pos_freq")
        plt.subplot(1, 2, 2)
        plt.plot(rc, pr, label=f"PR AUC={pr_auc:.3f}")
        plt.axhline(pos_freq, linestyle="--", color="gray")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim(0, 1.05)
        plt.title(f"PR Curve - {getattr(self, f'{prefix}_num_pos')} positives")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def fit_from_datamodule(self, datamodule):
        datamodule.setup()
        X_tensor, y_tensor = datamodule.train_set.dataset.X_tensor, datamodule.train_set.dataset.y_tensor
        indices = datamodule.train_set.indices
        X_train = X_tensor[indices].numpy()
        y_train = y_tensor[indices].numpy()
        self.fit(X_train, y_train)
        self.train_obs_names = datamodule.adata.obs_names[datamodule.train_set.indices].tolist()
        self.val_obs_names = datamodule.adata.obs_names[datamodule.val_set.indices].tolist()
        self.test_obs_names = datamodule.adata.obs_names[datamodule.test_set.indices].tolist()

    def evaluate_from_datamodule(self, datamodule, split="test"):
        datamodule.setup()
        if split == "val":
            subset = datamodule.val_set
        elif split == "test":
            subset = datamodule.test_set
        else:
            raise ValueError(f"Invalid split '{split}'")

        X_tensor, y_tensor = subset.dataset.X_tensor, subset.dataset.y_tensor
        indices = subset.indices
        X_eval = X_tensor[indices].numpy()
        y_eval = y_tensor[indices].numpy()

        return self.evaluate(X_eval, y_eval, prefix=split)
    
    def compute_shap(self, X, background=None, nsamples=100, target_class=None):
        """
        Compute SHAP values on input X, optionally for a specified target class.
        
        Parameters
        ----------
        X : array-like
            Input features
        background : array-like
            SHAP background
        nsamples : int
            Number of samples for kernel approximation
        target_class : int, optional
            If None, uses model predicted class
        """
        import shap

        # choose explainer
        if hasattr(self.model, "tree_") or hasattr(self.model, "estimators_"):
            explainer = shap.TreeExplainer(self.model, data=background)
        else:
            if background is None:
                background = shap.kmeans(X, 10)
            explainer = shap.KernelExplainer(self.model.predict_proba, background)

        # determine class
        if target_class is None:
            preds = self.model.predict(X)
            target_class = preds

        if isinstance(explainer, shap.TreeExplainer):
            shap_values = explainer.shap_values(X)
        else:
            shap_values = explainer.shap_values(X, nsamples=nsamples)
        
        if isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                if isinstance(target_class, int):
                    return shap_values[:, :, target_class]
                elif isinstance(target_class, np.ndarray):
                    # target_class is per-sample
                    if np.any(target_class >= shap_values.shape[2]):
                        raise ValueError(f"target_class values exceed {shap_values.shape[2]}")
                    selected = np.array([
                        shap_values[i, :, c]
                        for i, c in enumerate(target_class)
                    ])
                    return selected
                else:
                    # fallback to class 0
                    return shap_values[:, :, 0]
            else:
                # 2D shape (samples, features), no class dimension
                return shap_values

    def apply_shap_to_adata(self, dataloader, adata, background=None, adata_key="shap_values", target_class=None, normalize=True):
        """
        Compute SHAP from a DataLoader and store in AnnData if provided.
        """
        X_batches = []

        for batch in dataloader:
            X = batch[0].detach().cpu().numpy()
            X_batches.append(X)

        X_full = np.concatenate(X_batches, axis=0)

        shap_values = self.compute_shap(X_full, background=background, target_class=target_class)

        if adata is not None:
            adata.obsm[adata_key] = shap_values

        if normalize:
            arr = shap_values
            # row-wise normalization
            row_max = np.max(np.abs(arr), axis=1, keepdims=True)
            row_max[row_max == 0] = 1  # avoid divide by zero
            normalized = arr / row_max

            adata.obsm[f"{adata_key}_normalized"] = normalized