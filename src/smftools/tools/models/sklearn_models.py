from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight

import numpy as np

class SklearnModelWrapper:
    def __init__(self, model, name=None):
        self.model = model
        self.name = name or model.__class__.__name__

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError(f"{self.name} does not support predict_proba.")
        
    def evaluate(self, X, y, focus_class=None):
        probs = self.predict_proba(X)
        preds = self.predict(X)

        y_true = y
        y_pred = preds
        y_prob = probs

        unique_classes = np.unique(y_true)
        n_classes = len(unique_classes)
        is_binary = n_classes == 2

        # Handle focus_class (str or int)
        class_labels = unique_classes.tolist()
        if focus_class is not None:
            if isinstance(focus_class, str):
                if hasattr(focus_class, "cat"):
                    focus_class_idx = focus_class.cat.categories.get_loc(focus_class)
                else:
                    raise ValueError("String focus_class requires categorical-encoded labels.")
            else:
                focus_class_idx = int(focus_class)
        else:
            focus_class_idx = 1 if is_binary else None

        metrics = {}

        if is_binary:
            y_score = y_prob[:, focus_class_idx]
            y_binary = (y_true == focus_class_idx).astype(int)
            y_pred_bin = (y_pred == focus_class_idx).astype(int)

            precision, recall, _ = precision_recall_curve(y_binary, y_score)
            pr_auc = average_precision_score(y_binary, y_score)
            f1 = f1_score(y_binary, y_pred_bin)
            roc_auc = roc_auc_score(y_binary, y_score)
            pos_freq = np.mean(y_binary)
            pr_auc_norm = pr_auc / pos_freq if pos_freq > 0 else np.nan
            cm = confusion_matrix(y_true, y_pred)

            metrics.update({
                "focus_class": focus_class_idx,
                "f1": f1,
                "auc": roc_auc,
                "pr_auc": pr_auc,
                "pr_auc_norm": pr_auc_norm,
                "confusion_matrix": cm,
            })

        else:
            f1 = f1_score(y_true, y_pred, average="macro")
            roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            pr_auc = average_precision_score(y_true, y_prob, average="macro")
            pr_auc_norm = pr_auc / np.mean(np.bincount(y_true) / len(y_true))
            cm = confusion_matrix(y_true, y_pred)

            metrics.update({
                "f1_macro": f1,
                "auc_macro": roc_auc,
                "pr_auc_macro": pr_auc,
                "pr_auc_norm_macro": pr_auc_norm,
                "confusion_matrix": cm,
            })

            if focus_class_idx is not None:
                y_binary = (y_true == focus_class_idx).astype(int)
                y_score = y_prob[:, focus_class_idx]
                y_pred_bin = (y_pred == focus_class_idx).astype(int)

                precision, recall, _ = precision_recall_curve(y_binary, y_score)
                pr_auc = average_precision_score(y_binary, y_score)
                f1 = f1_score(y_binary, y_pred_bin)
                roc_auc = roc_auc_score(y_binary, y_score)
                pos_freq = np.mean(y_binary)
                pr_auc_norm = pr_auc / pos_freq if pos_freq > 0 else np.nan

                metrics.update({
                    f"class_{focus_class_idx}_f1": f1,
                    f"class_{focus_class_idx}_auc": roc_auc,
                    f"class_{focus_class_idx}_pr_auc": pr_auc,
                    f"class_{focus_class_idx}_pr_auc_norm": pr_auc_norm,
                })

        return metrics

    def fit_from_datamodule(self, datamodule):
        datamodule.setup()
        X_train, y_train = datamodule.train_set.tensors
        self.fit(X_train.cpu().numpy(), y_train.cpu().numpy())

    def evaluate_from_datamodule(self, datamodule, split="test"):
        datamodule.setup()

        if split == "val":
            X, y = datamodule.val_set.tensors
        elif split == "test":
            X, y = datamodule.test_set.tensors
        else:
            raise ValueError(f"Invalid split '{split}'; must be 'val' or 'test'")

        return self.evaluate(X.cpu().numpy(), y.cpu().numpy())
