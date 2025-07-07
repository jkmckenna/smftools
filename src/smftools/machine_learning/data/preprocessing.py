import numpy as np

def random_fill_nans(X):
    nan_mask = np.isnan(X)
    X[nan_mask] = np.random.rand(*X[nan_mask].shape)
    return X