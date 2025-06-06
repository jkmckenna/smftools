import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
import numpy as np
import pandas as pd

class AnnDataModule(pl.LightningDataModule):
    def __init__(self, adata, tensor_source="X", tensor_key=None, label_col="labels", 
                 batch_size=64, train_frac=0.7, random_seed=42, split_col='train_val_split', split_save_path=None, load_existing_split=False,
                 inference_mode=False):
        super().__init__()
        self.adata = adata # The adata object
        self.tensor_source = tensor_source # X, layers, obsm
        self.tensor_key = tensor_key  # name of the layer or obsm key
        self.label_col = label_col # name of the label column in obs
        self.batch_size = batch_size
        self.train_frac = train_frac
        self.random_seed = random_seed
        self.split_col = split_col  # Name of obs column to store "train"/"val"
        self.split_save_path = split_save_path # Where to save the obs_names and train/test split logging
        self.load_existing_split = load_existing_split # Whether to load from an existing split
        self.inference_mode = inference_mode # Whether to load the AnnDataModule in inference mode.

    def setup(self, stage=None):
        # Load feature matrix
        if self.tensor_source == "X":
            X = self.adata.X
        elif self.tensor_source == "layers":
            assert self.tensor_key in self.adata.layers, f"Layer '{self.tensor_key}' not found."
            X = self.adata.layers[self.tensor_key]
        elif self.tensor_source == "obsm":
            assert self.tensor_key in self.adata.obsm, f"obsm key '{self.tensor_key}' not found."
            X = self.adata.obsm[self.tensor_key]
        else:
            raise ValueError(f"Invalid tensor_source: {self.tensor_source}")

        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)

        if self.inference_mode:
            self.infer_dataset = TensorDataset(X_tensor)

        else:
            # Load and encode labels
            y = self.adata.obs[self.label_col]
            if y.dtype.name == 'category':
                y = y.cat.codes
            y_tensor = torch.tensor(y.values, dtype=torch.long)

            # Use existing split
            if self.load_existing_split:
                split_df = pd.read_csv(self.split_save_path, index_col=0)
                assert self.split_col in split_df.columns, f"'{self.split_col}' column missing in split file."
                self.adata.obs[self.split_col] = split_df.loc[self.adata.obs_names][self.split_col].values

            # If no split exists, create one
            if self.split_col not in self.adata.obs:
                full_dataset = TensorDataset(X_tensor, y_tensor)
                n_train = int(self.train_frac * len(full_dataset))
                n_val = len(full_dataset) - n_train
                self.train_set, self.val_set = random_split(
                    full_dataset, [n_train, n_val],
                    generator=torch.Generator().manual_seed(self.random_seed)
                )
                # Assign split labels
                split_array = np.full(len(self.adata), "val", dtype=object)
                train_idx = self.train_set.indices if hasattr(self.train_set, "indices") else self.train_set._indices
                split_array[train_idx] = "train"
                self.adata.obs[self.split_col] = split_array

                # Save to disk
                if self.split_save_path:
                    self.adata.obs[[self.split_col]].to_csv(self.split_save_path)
            else:
                split_labels = self.adata.obs[self.split_col].values
                train_mask = split_labels == "train"
                val_mask = split_labels == "val"
                self.train_set = TensorDataset(X_tensor[train_mask], y_tensor[train_mask])
                self.val_set = TensorDataset(X_tensor[val_mask], y_tensor[val_mask])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        if not self.inference_mode:
            raise RuntimeError("predict_dataloader only available in inference mode.")
        return DataLoader(self.infer_dataset, batch_size=self.batch_size)