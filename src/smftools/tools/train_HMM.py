# train_HMM

def train_HMM(adata, model_name='trained_HMM', save_hmm=False):
    """

    Parameters:
        adata (AnnData): Input AnnData object
        model_name (str): Name of the model
        save_hmm (bool): Whether to save the model
    
    """
    import numpy as np
    import anndata as ad
    from pomegranate.distributions import Categorical
    from pomegranate.hmm import DenseHMM

    bound = Categorical([[0.95, 0.05]])
    unbound = Categorical([[0.05, 0.95]])

    edges = [[0.9, 0.1], [0.1, 0.9]]
    starts = [0.5, 0.5]
    ends = [0.5, 0.5]

    model = DenseHMM([bound, unbound], edges=edges, starts=starts, ends=ends, max_iter=5, verbose=True)

    # define training sets and labels
    # Determine the number of reads to sample
    n_sample = round(0.7 * adata.X.shape[0])
    # Generate random indices
    np.random.seed(0)
    random_indices = np.random.choice(adata.shape[0], size=n_sample, replace=False)
    # Subset the AnnData object using the random indices
    training_adata_subsampled = adata[random_indices, :]
    training_sequences = training_adata_subsampled.X

    # Train the HMM without labeled data
    model.fit(training_sequences, algorithm='baum-welch')

    if save_hmm:
        # Save the model to a file
        model_json = model.to_json()
        with open(f'{model_name}.json', 'w') as f:
                f.write(model_json)