import torch
import pytorch_lightning as pl

from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score, confusion_matrix
)
import numpy as np

class TorchClassifierWrapper(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs=None,
        criterion_kwargs=None,
        lr: float = 1e-3,
        focus_class: int = 1,  # used for binary or multiclass precision-recall
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=['model'])  # logs all except actual model instance
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.criterion = None
        self.criterion_kwargs = criterion_kwargs or {}
        self.lr = lr
        self.focus_class = focus_class
        self.num_classes = num_classes

        self._val_outputs = []
        self._test_outputs = []

    def setup(self, stage=None):
        if self.criterion is None and self.num_classes is not None:
            self._init_criterion()

    def _init_criterion(self):
        if self.num_classes == 2:
            self.criterion = torch.nn.BCEWithLogitsLoss(**self.criterion_kwargs)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(**self.criterion_kwargs)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.num_classes is None:
            self.num_classes = int(torch.max(y).item()) + 1
            self._init_criterion()
        logits = self(x)
        loss = self._compute_loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        preds = self._get_preds(logits)
        acc = (preds == y).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)
        self._val_outputs.append((logits.detach(), y.detach()))
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self._test_outputs.append((logits.detach(), y.detach()))

    def on_validation_epoch_end(self):
        if not self._val_outputs:
            return
        logits, targets = zip(*self._val_outputs)
        self._val_outputs.clear()
        self._log_classification_metrics(logits, targets, prefix="val")

    def on_test_epoch_end(self):
        if not self._test_outputs:
            return
        logits, targets = zip(*self._test_outputs)
        self._test_outputs.clear()
        self._log_classification_metrics(logits, targets, prefix="test")

    def _compute_loss(self, logits, y):
        if self.num_classes == 2:
            y = y.float().view(-1, 1)  # shape [B, 1]
            return self.criterion(logits.view(-1, 1), y)
        else:
            return self.criterion(logits, y)
        
    def _get_probs(self, logits):
        if self.num_classes == 2:
            return torch.sigmoid(logits.view(-1))
        else:
            return torch.softmax(logits, dim=1)

    def _get_preds(self, logits):
        if self.num_classes == 2:
            return (torch.sigmoid(logits.view(-1)) >= 0.5).long()
        else:
            return logits.argmax(dim=1)
        
    def _log_classification_metrics(self, logits, targets, prefix="val"):
        logits = torch.cat(logits).cpu()
        y_true = torch.cat(targets).cpu().numpy()

        probs = self._get_probs(logits).numpy()
        preds = self._get_preds(logits).cpu().numpy()

        # Accuracy
        acc = np.mean(preds == y_true)

        # F1 & ROC-AUC
        if self.num_classes == 2:
            f1 = f1_score(y_true, preds)
            roc_auc = roc_auc_score(y_true, probs)
            focus_probs = probs
        else:
            f1 = f1_score(y_true, preds, average="macro")
            roc_auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
            focus_probs = probs[:, self.focus_class]

        # PR AUC for focus class
        binary_focus = (y_true == self.focus_class).astype(int)
        pr, rc, _ = precision_recall_curve(binary_focus, focus_probs)
        pr_auc = auc(rc, pr)
        pos_freq = binary_focus.mean()
        pr_auc_norm = pr_auc / pos_freq if pos_freq > 0 else np.nan

        cm = confusion_matrix(y_true, preds)

        # Logging
        self.log_dict({
            f"{prefix}_acc": acc,
            f"{prefix}_f1": f1,
            f"{prefix}_auc": roc_auc,
            f"{prefix}_pr_auc": pr_auc,
            f"{prefix}_pr_auc_norm": pr_auc_norm,
        })
        setattr(self, f"{prefix}_confusion_matrix", cm)

    def configure_optimizers(self):
        return self.optimizer_cls(self.parameters(), lr=self.lr, **self.optimizer_kwargs)
