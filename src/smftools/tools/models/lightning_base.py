import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score, confusion_matrix, roc_curve
)
import numpy as np

class TorchClassifierWrapper(pl.LightningModule):
    """
    A Pytorch Lightning wrapper for PyTorch classifiers.
    - Takes a PyTorch model as input.
    - Number of classes should be passed.
    - Optimizer is set as default to AdamW without any keyword arguments.
    - Loss criterion is automatically detected based on if it's a binary of multi-class classifier.
    - Can pass the index of the class label to use as the focus class when calculating precision/recall.
    - Contains a prediction step to run inference with.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        label_col: str,
        num_classes: int,
        class_names: list=None,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs=None,
        criterion_kwargs=None,
        lr: float = 1e-3,
        focus_class: int = 1,  # used for binary or multiclass precision-recall
        class_weights=None,
        enforce_eval_balance: bool=False,
        target_eval_freq: float=0.3,
        max_eval_positive: int=None
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=['model'])  # logs all except actual model instance
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {"weight_decay": 1e-4}
        self.criterion = None
        self.lr = lr
        self.label_col = label_col
        self.num_classes = num_classes
        self.class_names = class_names
        self.focus_class = self._resolve_focus_class(focus_class)
        self.focus_class_name = focus_class
        self.enforce_eval_balance = enforce_eval_balance
        self.target_eval_freq = target_eval_freq
        self.max_eval_positive = max_eval_positive

        # Handle class weights
        self.criterion_kwargs = criterion_kwargs or {}

        if class_weights is not None:
            if num_classes == 2:
                # BCEWithLogits uses pos_weight, expects a scalar or tensor
                if torch.is_tensor(class_weights[self.focus_class]):
                    self.criterion_kwargs["pos_weight"] = class_weights[self.focus_class]
                else:
                    self.criterion_kwargs["pos_weight"] = torch.tensor(class_weights[self.focus_class], dtype=torch.float32, device=self.device)
            else:
                # CrossEntropyLoss expects weight tensor of size C
                if torch.is_tensor(class_weights):
                    self.criterion_kwargs["weight"] = class_weights
                else:
                    self.criterion_kwargs["weight"] = torch.tensor(class_weights, dtype=torch.float32)


        self._val_outputs = []
        self._test_outputs = []

    def setup(self, stage=None):
        """
        Sets the loss criterion.
        """
        if self.criterion is None and self.num_classes is not None:
            self._init_criterion()

    def _init_criterion(self):
        if self.num_classes == 2:
            if "pos_weight" in self.criterion_kwargs and not torch.is_tensor(self.criterion_kwargs["pos_weight"]):
                self.criterion_kwargs["pos_weight"] = torch.tensor(self.criterion_kwargs["pos_weight"], dtype=torch.float32, device=self.device)
            self.criterion = torch.nn.BCEWithLogitsLoss(**self.criterion_kwargs)
        else:
            if "weight" in self.criterion_kwargs and not torch.is_tensor(self.criterion_kwargs["weight"]):
                self.criterion_kwargs["weight"] = torch.tensor(self.criterion_kwargs["weight"], dtype=torch.float32, device=self.device)
            self.criterion = torch.nn.CrossEntropyLoss(**self.criterion_kwargs)

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
        
    def set_training_indices(self, datamodule):
        """
        Store obs_names for train/val/test subsets used during training.
        """
        self.train_obs_names = datamodule.adata.obs_names[datamodule.train_set.indices].tolist()
        self.val_obs_names = datamodule.adata.obs_names[datamodule.val_set.indices].tolist()
        self.test_obs_names = datamodule.adata.obs_names[datamodule.test_set.indices].tolist()

    def configure_optimizers(self):
        return self.optimizer_cls(self.parameters(), lr=self.lr, **self.optimizer_kwargs)

    def forward(self, x):
        """
        Forward pass through the model.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step for a batch through the Lightning Trainer.
        """
        x, y = batch
        if self.num_classes is None:
            self.num_classes = int(torch.max(y).item()) + 1
            self._init_criterion()
        logits = self(x)
        loss = self._compute_loss(logits, y)
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for a batch through the Lightning Trainer.
        """
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        preds = self._get_preds(logits)
        acc = (preds == y).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=False)
        self._val_outputs.append((logits.detach(), y.detach()))
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        Test step for a batch through the Lightning Trainer.
        """
        x, y = batch
        logits = self(x)
        self._test_outputs.append((logits.detach(), y.detach()))

    def predict_step(self, batch, batch_idx):
        """
        Gets predictions and prediction probabilities for the batch using the trained Lightning model.
        """
        x = batch[0]
        logits = self(x)
        probs = self._get_probs(logits)
        preds = self._get_preds(logits)
        return preds, probs

    def on_validation_epoch_end(self):
        """
        Final logging of all validation steps
        """
        if not self._val_outputs:
            return
        logits, targets = zip(*self._val_outputs)
        self._val_outputs.clear()
        self._log_classification_metrics(logits, targets, prefix="val")

    def on_test_epoch_end(self):
        """
        Final logging of all testing steps
        """
        if not self._test_outputs:
            return
        logits, targets = zip(*self._test_outputs)
        self._test_outputs.clear()
        self._log_classification_metrics(logits, targets, prefix="test")
        self._plot_roc_pr_curves(logits, targets)

    def _compute_loss(self, logits, y):
        """
        A helper function for computing loss for binary vs multiclass classifications.
        """
        if self.num_classes == 2:
            y = y.float().view(-1, 1)  # shape [B, 1]
            return self.criterion(logits.view(-1, 1), y)
        else:
            return self.criterion(logits, y)
        
    def _get_probs(self, logits):
        """
        A helper function for getting class probabilities for binary vs multiclass classifications.
        """
        if self.num_classes == 2:
            return torch.sigmoid(logits.view(-1))
        else:
            return torch.softmax(logits, dim=1)

    def _get_preds(self, logits):
        """
        A helper function for getting class predictions for binary vs multiclass classifications.
        """
        if self.num_classes == 2:
            return (torch.sigmoid(logits.view(-1)) >= 0.5).long()
        else:
            return logits.argmax(dim=1)
        
    def _subsample_for_fixed_positive_frequency(self, y_true, probs, target_freq=0.3, max_positive=None):
        pos_idx = np.where(y_true == self.focus_class)[0]
        neg_idx = np.where(y_true != self.focus_class)[0]

        max_negatives_possible = len(neg_idx)
        max_positives_possible = len(pos_idx)

        # maximum achievable positive class frequency
        max_possible_freq = max_positives_possible / (max_positives_possible + max_negatives_possible)

        if target_freq > max_possible_freq:
            target_freq = max_possible_freq  # clip if you ask for impossible freq

        # now calculate positive count
        num_pos_target = min(int(target_freq * max_negatives_possible / (1 - target_freq)), max_positives_possible)
        num_neg_target = int(num_pos_target * (1 - target_freq) / target_freq)
        num_neg_target = min(num_neg_target, max_negatives_possible)
        
        pos_sampled = np.random.choice(pos_idx, size=num_pos_target, replace=False)
        neg_sampled = np.random.choice(neg_idx, size=num_neg_target, replace=False)

        sampled_idx = np.concatenate([pos_sampled, neg_sampled])
        np.random.shuffle(sampled_idx)

        actual_freq = len(pos_sampled) / len(sampled_idx)

        return sampled_idx
        
    def _log_classification_metrics(self, logits, targets, prefix="val"):
        """
        A helper function for logging validation and testing split model evaluations.
        """
        logits = torch.cat(logits).cpu()
        y_true = torch.cat(targets).cpu().numpy()

        probs = self._get_probs(logits).numpy()
        preds = self._get_preds(logits).cpu().numpy()

        # remap binary focus class correctly:
        binary_focus = (y_true == self.focus_class).astype(int)

        num_pos = binary_focus.sum()

        # Subsample if you want to enforce a fixed proportion of the positive class
        if prefix == 'test' and self.enforce_eval_balance:
            sampled_idx = self._subsample_for_fixed_positive_frequency(
                y_true, probs, target_freq=self.target_eval_freq, max_positive=self.max_eval_positive
            )
            y_true = y_true[sampled_idx]
            probs = probs[sampled_idx]
            preds = preds[sampled_idx]
            binary_focus = (y_true == self.focus_class).astype(int)
            num_pos = binary_focus.sum()

        # Accuracy
        acc = np.mean(preds == y_true)

        # F1 & ROC-AUC
        if self.num_classes == 2:
            if self.focus_class == 1:
                focus_probs = probs
            else:
                focus_probs = 1 - probs
            f1 = f1_score(y_true, preds)
            fpr, tpr, _ = roc_curve((y_true == self.focus_class).astype(int), focus_probs)
            roc_auc = roc_auc_score((y_true == self.focus_class).astype(int), focus_probs)
        else:
            f1 = f1_score(y_true, preds, average="macro")
            roc_auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
            focus_probs = probs[:, self.focus_class]
            fpr, tpr, _ = roc_curve((y_true == self.focus_class).astype(int), focus_probs)

        # PR AUC for focus class
        pr, rc, _ = precision_recall_curve(binary_focus, focus_probs)
        pr_auc = auc(rc, pr)
        pos_freq = binary_focus.mean()
        pr_auc_norm = pr_auc / pos_freq if pos_freq > 0 else np.nan

        cm = confusion_matrix(y_true, preds)

        # Save attributes for later plotting
        if prefix == 'test':
            self.test_roc_curve = (fpr, tpr)
            self.test_pr_curve = (rc, pr)
            self.test_roc_auc = roc_auc
            self.test_pr_auc = pr_auc
            self.test_pos_freq = pos_freq
            self.test_num_pos = num_pos
            self.test_acc = acc
            self.test_f1 = f1
        elif prefix == 'val':
            pass

        # Logging
        self.log_dict({
            f"{prefix}_acc": acc,
            f"{prefix}_f1": f1,
            f"{prefix}_auc": roc_auc,
            f"{prefix}_pr_auc": pr_auc,
            f"{prefix}_pr_auc_norm": pr_auc_norm,
            f"{prefix}_pos_freq": pos_freq,
            f"{prefix}_num_pos": num_pos
        })
        setattr(self, f"{prefix}_confusion_matrix", cm)

    def _plot_roc_pr_curves(self, logits, targets):
        plt.figure(figsize=(12, 5))

        # ROC Curve
        fpr, tpr = self.test_roc_curve
        roc_auc = self.test_roc_auc
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.ylim(0, 1.05)
        plt.title(f"Test ROC Curve - {self.test_num_pos} positive class instances")
        plt.legend()

        # PR Curve
        rc, pr = self.test_pr_curve
        pr_auc = self.test_pr_auc
        pos_freq = self.test_pos_freq
        plt.subplot(1, 2, 2)
        plt.plot(rc, pr, label=f"PR AUC={pr_auc:.3f}")
        plt.axhline(pos_freq, linestyle='--', color="gray")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim(0, 1.05)
        plt.title(f"Test Precision-Recall Curve - {self.test_num_pos} positive class instances")
        plt.legend()

        plt.tight_layout()
        plt.show()
