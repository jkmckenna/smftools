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
