import pandas as pd

def annotate_split_column(adata, model, split_col="split"):
    """
    Annotate adata.obs with train/val/test/new labels based on model's stored obs_names.
    """
    # Get sets for fast lookup
    train_set = set(model.train_obs_names)
    val_set = set(model.val_obs_names)
    test_set = set(model.test_obs_names)
    
    # Create array for split labels
    split_labels = []
    for obs in adata.obs_names:
        if obs in train_set:
            split_labels.append("training")
        elif obs in val_set:
            split_labels.append("validation")
        elif obs in test_set:
            split_labels.append("testing")
        else:
            split_labels.append("new")
    
    # Store in AnnData.obs
    adata.obs[split_col] = pd.Categorical(split_labels, categories=["training", "validation", "testing", "new"])
    
    print(f"Annotated {split_col} column with training/validation/testing/new status.")
