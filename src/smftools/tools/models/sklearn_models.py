from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, f1_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

import numpy as np

class SklearnModelWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        probs = self.predict_proba(X_test)[:, 1]
        preds = self.predict(X_test)

        fpr, tpr, _ = roc_curve(y_test, probs)
        precision, recall, _ = precision_recall_curve(y_test, probs)
        f1 = f1_score(y_test, preds)
        auc_score = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        cm = confusion_matrix(y_test, preds)
        pos_freq = np.mean(y_test == 1)
        pr_auc_norm = pr_auc / pos_freq

        return {
            "fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall,
            "f1": f1, "auc": auc_score, "pr_auc": pr_auc,
            "pr_auc_norm": pr_auc_norm, "confusion_matrix": cm
        }