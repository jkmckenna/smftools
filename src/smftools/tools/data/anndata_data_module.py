import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
import numpy as np
import pandas as pd

class AnnDataModule(pl.LightningDataModule):
    def __init__(self, adata, tensor_source="X", tensor_key=None, label_col="labels", 
                 batch_size=64, train_frac=0.6, val_frac=0.1, test_frac=0.3, random_seed=42, 
                 split_col='train_val_test_split', split_save_path=None, load_existing_split=False,
                 inference_mode=False):
        super().__init__()
        self.adata = adata # The adata object
        self.tensor_source = tensor_source # X, layers, obsm
        self.tensor_key = tensor_key  # name of the layer or obsm key
        self.label_col = label_col # name of the label column in obs
        self.batch_size = batch_size
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.random_seed = random_seed
        self.split_col = split_col  # Name of obs column to store "train"/"val"
        self.split_save_path = split_save_path # Where to save the obs_names and train/test split logging
        self.load_existing_split = load_existing_split # Whether to load from an existing split
        self.inference_mode = inference_mode # Whether to load the AnnDataModule in inference mode.
        self.var_names = adata.var_names.copy() # Save the var_names of the AnnData to the AnnDataModule
        self.obs_names = adata.obs_names.copy() # Save the obs_names of the AnnData to the AnnDataModule

    def setup(self, stage=None):

        total = self.train_frac + self.val_frac + self.test_frac
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split fractions must sum to 1.0 (got {total})")
        
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
            return

        else:
            # Load and encode labels
            y = self.adata.obs[self.label_col]
            if y.dtype.name == 'category':
                y = y.cat.codes
            y_tensor = torch.tensor(y.values, dtype=torch.long)

        # Load or create split
        if self.load_existing_split:
            split_df = pd.read_csv(self.split_save_path, index_col=0)
            self.adata.obs[self.split_col] = split_df.loc[self.adata.obs_names][self.split_col].values

        if self.split_col not in self.adata.obs:
            # Create new split
            n_total = len(self.adata)
            indices = np.arange(n_total)
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)

            n_train = int(self.train_frac * n_total)
            n_val = int(self.val_frac * n_total)
            n_test = n_total - n_train - n_val

            split_array = np.full(n_total, "test", dtype=object)
            split_array[indices[:n_train]] = "train"
            split_array[indices[n_train:n_train+n_val]] = "val"
            self.adata.obs[self.split_col] = split_array

            if self.split_save_path:
                self.adata.obs[[self.split_col]].to_csv(self.split_save_path)

        # Build datasets
        split_labels = self.adata.obs[self.split_col].values
        train_mask = split_labels == "train"
        val_mask = split_labels == "val"
        test_mask = split_labels == "test"

        self.train_set = TensorDataset(X_tensor[train_mask], y_tensor[train_mask])
        self.val_set = TensorDataset(X_tensor[val_mask], y_tensor[val_mask])
        self.test_set = TensorDataset(X_tensor[test_mask], y_tensor[test_mask])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        if not self.inference_mode:
            raise RuntimeError("predict_dataloader only available in inference mode.")
        return DataLoader(self.infer_dataset, batch_size=self.batch_size)