import pandas as pd
import numpy as np
from .inference_utils import annotate_split_column


def run_sklearn_inference(
    adata,
    model,
    datamodule, 
    prefix="model"
):
    """
    Run inference on AnnData using SklearnModelWrapper.
    """

    label_col = model.label_col
    num_classes = model.num_classes
    class_labels = model.class_names
    focus_class_name = model.focus_class_name

    annotate_split_column(adata, model, split_col=f"{prefix}_training_split")

    datamodule.setup()

    X_infer = datamodule.to_numpy()

    # Run predictions
    preds = model.predict(X_infer)
    probs = model.predict_proba(X_infer)

    # Handle binary vs multiclass formats
    if num_classes == 2:
        # probs shape: (N, 2) from predict_proba
        pred_class_idx = preds
        probs_all = probs
        pred_class_probs = probs[np.arange(len(probs)), pred_class_idx]
    else:
        pred_class_idx = preds
        probs_all = probs
        pred_class_probs = probs[np.arange(len(probs)), pred_class_idx]

    pred_class_labels = [class_labels[i] for i in pred_class_idx]

    full_prefix = f"{prefix}_{label_col}"

    adata.obs[f"{full_prefix}_pred"] = pred_class_idx
    adata.obs[f"{full_prefix}_pred_label"] = pd.Categorical(pred_class_labels, categories=class_labels)
    adata.obs[f"{full_prefix}_pred_prob"] = pred_class_probs

    for i, class_name in enumerate(class_labels):
        adata.obs[f"{full_prefix}_prob_{class_name}"] = probs_all[:, i]

    adata.obsm[f"{full_prefix}_pred_prob_all"] = probs_all

    print(f"Inference complete: stored under prefix '{full_prefix}'")
