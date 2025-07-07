import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score, confusion_matrix, roc_curve
)

class ModelEvaluator:
    """
    A model evaluator for consolidating Sklearn and Lightning model evaluation metrics on testing data
    """
    def __init__(self):
        self.results = []
        self.pos_freq = None
        self.num_pos = None

    def add_model(self, name, model, is_torch=True):
        """
        Add a trained model with its evaluation metrics.
        """
        if is_torch:
            entry = {
                'name': name,
                'f1': model.test_f1,
                'auc': model.test_roc_auc,
                'pr_auc': model.test_pr_auc,
                'pr_auc_norm': model.test_pr_auc / model.test_pos_freq if model.test_pos_freq > 0 else np.nan,
                'pr_curve': model.test_pr_curve,
                'roc_curve': model.test_roc_curve,
                'num_pos': model.test_num_pos,
                'pos_freq': model.test_pos_freq
            }
        else:
            entry = {
                'name': name,
                'f1': model.test_f1,
                'auc': model.test_roc_auc,
                'pr_auc': model.test_pr_auc,
                'pr_auc_norm': model.test_pr_auc / model.test_pos_freq if model.test_pos_freq > 0 else np.nan,
                'pr_curve': model.test_pr_curve,
                'roc_curve': model.test_roc_curve,
                'num_pos': model.test_num_pos,
                'pos_freq': model.test_pos_freq
            }
        
        self.results.append(entry)

        if not self.pos_freq:
            self.pos_freq = entry['pos_freq']
            self.num_pos = entry['num_pos']

    def get_metrics_dataframe(self):
        """
        Return all metrics as pandas DataFrame.
        """
        df = pd.DataFrame(self.results)
        return df[['name', 'f1', 'auc', 'pr_auc', 'pr_auc_norm', 'num_pos', 'pos_freq']]

    def plot_all_curves(self):
        """
        Plot unified ROC and PR curves across all models.
        """
        plt.figure(figsize=(12, 5))

        # ROC
        plt.subplot(1, 2, 1)
        for res in self.results:
            fpr, tpr = res['roc_curve']
            plt.plot(fpr, tpr, label=f"{res['name']} (AUC={res['auc']:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.ylim(0,1.05)
        plt.title(f"ROC Curves - {self.num_pos} positive instances")
        plt.legend()

        # PR
        plt.subplot(1, 2, 2)
        for res in self.results:
            rc, pr = res['pr_curve']
            plt.plot(rc, pr, label=f"{res['name']} (AUPRC={res['pr_auc']:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim(0,1.05)
        plt.axhline(self.pos_freq, linestyle='--', color='grey')
        plt.title(f"Precision-Recall Curves - {self.num_pos} positive instances")
        plt.legend()

        plt.tight_layout()
        plt.show()

class PostInferenceModelEvaluator:
    def __init__(self, adata, models, target_eval_freq=None, max_eval_positive=None):
        """
        Initialize evaluator.

        Parameters:
        -----------
        adata : AnnData
            The annotated dataset where predictions are stored in obs/obsm.
        models : dict
            Dictionary of models: {model_name: model_instance}.
            Supports TorchClassifierWrapper and SklearnModelWrapper.
        """
        self.adata = adata
        self.models = models
        self.target_eval_freq = target_eval_freq
        self.max_eval_positive = max_eval_positive
        self.results = {}

    def evaluate_all(self):
        """
        Evaluate all models and store results.
        """
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            label_col = model.label_col
            full_prefix = f"{name}_{label_col}"
            self.results[full_prefix] = self._evaluate_model(name, model)

    def _evaluate_model(self, model_name, model):
        """
        Evaluate one model and return metrics.
        """
        label_col = model.label_col
        num_classes = model.num_classes
        class_names = model.class_names
        focus_class = model.focus_class

        full_prefix = f"{model_name}_{label_col}"

        # Extract ground truth + predictions
        y_true = self.adata.obs[label_col].cat.codes.to_numpy()
        y_pred = self.adata.obs[f"{full_prefix}_pred"].to_numpy()
        probs_all = self.adata.obsm[f"{full_prefix}_pred_prob_all"]

        binary_focus = (y_true == focus_class).astype(int)

        # OPTIONAL SUBSAMPLING
        if self.target_eval_freq is not None:
            indices = self._subsample_for_fixed_positive_frequency(
                binary_focus, target_freq=self.target_eval_freq, max_positive=self.max_eval_positive
            )
            y_true = y_true[indices]
            y_pred = y_pred[indices]
            probs_all = probs_all[indices]
            binary_focus = (y_true == focus_class).astype(int)

        acc = np.mean(y_true == y_pred)

        if num_classes == 2:
            focus_probs = probs_all[:, focus_class]
            f1 = f1_score(binary_focus, (y_pred == focus_class).astype(int))
            roc_auc = roc_auc_score(binary_focus, focus_probs)
            pr, rc, _ = precision_recall_curve(binary_focus, focus_probs)
            fpr, tpr, _ = roc_curve(binary_focus, focus_probs)
            pr_auc = auc(rc, pr)
            pos_freq = binary_focus.mean()
            pr_auc_norm = pr_auc / pos_freq if pos_freq > 0 else np.nan
        else:
            f1 = f1_score(y_true, y_pred, average="macro")
            roc_auc = roc_auc_score(y_true, probs_all, multi_class="ovr", average="macro")
            focus_probs = probs_all[:, focus_class]
            pr, rc, _ = precision_recall_curve(binary_focus, focus_probs)
            fpr, tpr, _ = roc_curve(binary_focus, focus_probs)
            pr_auc = auc(rc, pr)
            pos_freq = binary_focus.mean()
            pr_auc_norm = pr_auc / pos_freq if pos_freq > 0 else np.nan

        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            "accuracy": acc,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "pr_auc_norm": pr_auc_norm,
            "pos_freq": pos_freq,
            "confusion_matrix": cm,
            "pr_rc_curve": (pr, rc),
            "roc_curve": (tpr, fpr)
        }

        return metrics
    
    def _subsample_for_fixed_positive_frequency(self, binary_labels, target_freq=0.3, max_positive=None):
        pos_idx = np.where(binary_labels == 1)[0]
        neg_idx = np.where(binary_labels == 0)[0]

        max_pos = len(pos_idx)
        max_neg = len(neg_idx)

        max_possible_freq = max_pos / (max_pos + max_neg)
        if target_freq > max_possible_freq:
            target_freq = max_possible_freq

        num_pos_target = int(target_freq * max_neg / (1 - target_freq))
        num_pos_target = min(num_pos_target, max_pos)
        if max_positive is not None:
            num_pos_target = min(num_pos_target, max_positive)

        num_neg_target = int(num_pos_target * (1 - target_freq) / target_freq)
        num_neg_target = min(num_neg_target, max_neg)

        pos_sampled = np.random.choice(pos_idx, size=num_pos_target, replace=False)
        neg_sampled = np.random.choice(neg_idx, size=num_neg_target, replace=False)
        sampled_idx = np.concatenate([pos_sampled, neg_sampled])
        np.random.shuffle(sampled_idx)
        return sampled_idx

    def to_dataframe(self):
        """
        Convert results to pandas DataFrame (excluding confusion matrices).
        """
        records = []
        for model_name, metrics in self.results.items():
            row = {"model": model_name}
            for k, v in metrics.items():
                if k not in ["confusion_matrix", "pr_rc_curve", "roc_curve"]:
                    row[k] = v
            records.append(row)
        return pd.DataFrame(records)
