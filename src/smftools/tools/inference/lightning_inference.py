import torch
import pandas as pd
import numpy as np
from pytorch_lightning import Trainer

def run_lightning_inference(
    adata,
    model,
    datamodule,
    label_col="labels",
    prefix="model"
):

    # Get class labels
    if label_col in adata.obs and pd.api.types.is_categorical_dtype(adata.obs[label_col]):
        class_labels = adata.obs[label_col].cat.categories.tolist()
    else:
        raise ValueError("label_col must be a categorical column in adata.obs")

    # Run predictions
    trainer = Trainer(accelerator="auto", devices=1, logger=False, enable_checkpointing=False)
    preds = trainer.predict(model, datamodule=datamodule)
    probs = torch.cat(preds, dim=0).cpu().numpy()  # (N, C)
    pred_class_idx = probs.argmax(axis=1)
    pred_class_labels = [class_labels[i] for i in pred_class_idx]
    pred_class_probs = probs[np.arange(len(probs)), pred_class_idx]

    # Construct full prefix with label_col
    full_prefix = f"{prefix}_{label_col}"

    # Store predictions in obs
    adata.obs[f"{full_prefix}_pred"] = pred_class_idx
    adata.obs[f"{full_prefix}_pred_label"] = pd.Categorical(pred_class_labels, categories=class_labels)
    adata.obs[f"{full_prefix}_pred_prob"] = pred_class_probs

    # Per-class probabilities
    for i, class_name in enumerate(class_labels):
        adata.obs[f"{full_prefix}_prob_{class_name}"] = probs[:, i]

    # Full probability matrix in obsm
    adata.obsm[f"{full_prefix}_pred_prob_all"] = probs
