import torch
import pandas as pd
import numpy as np
from pytorch_lightning import Trainer
from .inference_utils import annotate_split_column

def run_lightning_inference(
    adata,
    model,
    datamodule,
    trainer,
    prefix="model",
    devices=1
):
    """
    Run inference on AnnData using TorchClassifierWrapper + AnnDataModule (in inference mode).
    """

    # Device logic
    if torch.cuda.is_available():
        accelerator = "gpu"
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
    else:
        accelerator = "cpu"
        devices = 1

    label_col = model.label_col
    num_classes = model.num_classes
    class_labels = model.class_names
    focus_class = model.focus_class
    focus_class_name = model.focus_class_name

    annotate_split_column(adata, model, split_col=f"{prefix}_training_split")

    # Run predictions
    outputs = trainer.predict(model, datamodule=datamodule)

    preds_list, probs_list = zip(*outputs)
    preds = torch.cat(preds_list, dim=0).cpu().numpy()
    probs = torch.cat(probs_list, dim=0).cpu().numpy()

    # Handle binary vs multiclass formats
    if num_classes == 2:
        # probs shape: (N,) from sigmoid
        pred_class_idx = (probs >= 0.5).astype(int)
        probs_all = np.vstack([1 - probs, probs]).T  # shape (N, 2)
        pred_class_probs = probs_all[np.arange(len(probs_all)), pred_class_idx]
    else:
        pred_class_idx = probs.argmax(axis=1)
        probs_all = probs
        pred_class_probs = probs_all[np.arange(len(probs_all)), pred_class_idx]

    pred_class_labels = [class_labels[i] for i in pred_class_idx]

    full_prefix = f"{prefix}_{label_col}"

    adata.obs[f"{full_prefix}_pred"] = pred_class_idx
    adata.obs[f"{full_prefix}_pred_label"] = pd.Categorical(pred_class_labels, categories=class_labels)
    adata.obs[f"{full_prefix}_pred_prob"] = pred_class_probs

    for i, class_name in enumerate(class_labels):
        adata.obs[f"{full_prefix}_prob_{class_name}"] = probs_all[:, i]

    adata.obsm[f"{full_prefix}_pred_prob_all"] = probs_all

    print(f"Inference complete: stored under prefix '{full_prefix}'")