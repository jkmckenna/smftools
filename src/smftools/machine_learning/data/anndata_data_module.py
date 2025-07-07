import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset, Subset
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from .preprocessing import random_fill_nans
from sklearn.utils.class_weight import compute_class_weight

    
class AnnDataDataset(Dataset):
    """
    Generic PyTorch Dataset from AnnData.
    """
    def __init__(self, adata, tensor_source="X", tensor_key=None, label_col=None, window_start=None, window_size=None):
        self.adata = adata
        self.tensor_source = tensor_source
        self.tensor_key = tensor_key
        self.label_col = label_col
        self.window_start = window_start
        self.window_size = window_size
        
        if tensor_source == "X":
            X = adata.X
        elif tensor_source == "layers":
            assert tensor_key in adata.layers
            X = adata.layers[tensor_key]
        elif tensor_source == "obsm":
            assert tensor_key in adata.obsm
            X = adata.obsm[tensor_key]
        else:
            raise ValueError(f"Invalid tensor_source: {tensor_source}")
        
        if self.window_start is not None and self.window_size is not None:
            X = X[:, self.window_start : self.window_start + self.window_size]
        
        X = random_fill_nans(X)

        self.X_tensor = torch.tensor(X, dtype=torch.float32)

        if label_col is not None:
            y = adata.obs[label_col]
            if y.dtype.name == 'category':
                y = y.cat.codes
            self.y_tensor = torch.tensor(y.values, dtype=torch.long)
        else:
            self.y_tensor = None

    def numpy(self, indices):
        return self.X_tensor[indices].numpy(), self.y_tensor[indices].numpy()
    
    def __len__(self):
        return len(self.X_tensor)

    def __getitem__(self, idx):
        x = self.X_tensor[idx]
        if self.y_tensor is not None:
            y = self.y_tensor[idx]
            return x, y
        else:
            return (x,)


def split_dataset(adata, dataset, train_frac=0.6, val_frac=0.1, test_frac=0.3, 
                                 random_seed=42, split_col="train_val_test_split", 
                                 load_existing_split=False, split_save_path=None):
    """
    Perform split and record assignment into adata.obs[split_col].
    """
    total_len = len(dataset)

    if load_existing_split:
        if split_col in adata.obs:
            pass  # use existing
        elif split_save_path:
            split_df = pd.read_csv(split_save_path, index_col=0)
            adata.obs[split_col] = split_df.loc[adata.obs_names][split_col].values
        else:
            raise ValueError("No existing split column found and no file provided.")
    else:
        indices = np.arange(total_len)
        np.random.seed(random_seed)
        np.random.shuffle(indices)

        n_train = int(train_frac * total_len)
        n_val = int(val_frac * total_len)
        n_test = total_len - n_train - n_val

        split_array = np.full(total_len, "test", dtype=object)
        split_array[indices[:n_train]] = "train"
        split_array[indices[n_train:n_train + n_val]] = "val"
        adata.obs[split_col] = split_array

        if split_save_path:
            adata.obs[[split_col]].to_csv(split_save_path)

    split_labels = adata.obs[split_col].values
    train_indices = np.where(split_labels == "train")[0]
    val_indices = np.where(split_labels == "val")[0]
    test_indices = np.where(split_labels == "test")[0]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    return train_set, val_set, test_set

class AnnDataModule(pl.LightningDataModule):
    """
    Unified LightningDataModule version of AnnDataDataset + splitting with adata.obs recording.
    """
    def __init__(self, adata, tensor_source="X", tensor_key=None, label_col="labels",
                 batch_size=64, train_frac=0.6, val_frac=0.1, test_frac=0.3, random_seed=42,
                 inference_mode=False, split_col="train_val_test_split", split_save_path=None,
                 load_existing_split=False, window_start=None, window_size=None, num_workers=None, persistent_workers=False):
        super().__init__()
        self.adata = adata
        self.tensor_source = tensor_source
        self.tensor_key = tensor_key
        self.label_col = label_col
        self.batch_size = batch_size
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.random_seed = random_seed
        self.inference_mode = inference_mode
        self.split_col = split_col
        self.split_save_path = split_save_path
        self.load_existing_split = load_existing_split
        self.var_names = adata.var_names.copy()
        self.window_start = window_start
        self.window_size = window_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    def setup(self, stage=None):
        dataset = AnnDataDataset(self.adata, self.tensor_source, self.tensor_key, 
                                  None if self.inference_mode else self.label_col,
                                    window_start=self.window_start, window_size=self.window_size)

        if self.inference_mode:
            self.infer_dataset = dataset
            return

        self.train_set, self.val_set, self.test_set = split_dataset(
            self.adata, dataset, train_frac=self.train_frac, val_frac=self.val_frac, 
            test_frac=self.test_frac, random_seed=self.random_seed, 
            split_col=self.split_col, split_save_path=self.split_save_path, 
            load_existing_split=self.load_existing_split
        )

    def train_dataloader(self):
        if self.num_workers:
            return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
        else:
            return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        if self.num_workers:
            return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
        else:
            return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False)
        
    def test_dataloader(self):
        if self.num_workers:
            return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
        else:
            return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False)
        
    def predict_dataloader(self):
        if not self.inference_mode:
            raise RuntimeError("Only valid in inference mode")
        return DataLoader(self.infer_dataset, batch_size=self.batch_size)
    
    def compute_class_weights(self):
        train_indices = self.train_set.indices # get the indices of the training set
        y_all = self.train_set.dataset.y_tensor # get labels for the entire dataset (We are pulling from a Subset object, so this syntax can be confusing)
        y_train = y_all[train_indices].cpu().numpy() # get the labels for the training set and move to a numpy array
        
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        return torch.tensor(class_weights, dtype=torch.float32)
    
    def inference_numpy(self):
        """
        Return inference data as numpy for use in sklearn inference.
        """
        if not self.inference_mode:
            raise RuntimeError("Must be in inference_mode=True to use inference_numpy()")
        X_np = self.infer_dataset.X_tensor.numpy()
        return X_np
    
    def to_numpy(self):
        """
        Move the AnnDataModule tensors into numpy arrays
        """
        if not self.inference_mode:
            train_X, train_y = self.train_set.dataset.numpy(self.train_set.indices)
            val_X, val_y = self.val_set.dataset.numpy(self.val_set.indices)
            test_X, test_Y = self.test_set.dataset.numpy(self.test_set.indices)
            return train_X, train_y, val_X, val_y, test_X, test_Y
        else:
            return self.inference_numpy()


def build_anndata_loader(
    adata, tensor_source="X", tensor_key=None, label_col=None, train_frac=0.6, val_frac=0.1, 
    test_frac=0.3, random_seed=42, batch_size=64, lightning=True, inference_mode=False, 
    split_col="train_val_test_split", split_save_path=None, load_existing_split=False
):
    """
    Unified pipeline for both Lightning and raw PyTorch.
    The lightning loader works for both Lightning and the Sklearn wrapper.
    Set lightning to False if you want to make data loaders for base PyTorch or base sklearn models
    """
    if lightning:
        return AnnDataModule(
            adata, tensor_source=tensor_source, tensor_key=tensor_key, label_col=label_col,
            batch_size=batch_size, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac,
            random_seed=random_seed, inference_mode=inference_mode,
            split_col=split_col, split_save_path=split_save_path, load_existing_split=load_existing_split
        )
    else:
        var_names = adata.var_names.copy()
        dataset = AnnDataDataset(adata, tensor_source, tensor_key, None if inference_mode else label_col)
        if inference_mode:
            return DataLoader(dataset, batch_size=batch_size)
        else:
            train_set, val_set, test_set = split_dataset(
                adata, dataset, train_frac, val_frac, test_frac, random_seed, 
                split_col, split_save_path, load_existing_split
            )
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=batch_size)
            test_loader = DataLoader(test_set, batch_size=batch_size)
            return train_loader, val_loader, test_loader