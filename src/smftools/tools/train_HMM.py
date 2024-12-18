# train_HMM

def train_HMM(adata, model_name='Methylation_HMM', save_hmm=False):
    """

    Parameters:
        adata (AnnData): Input AnnData object
        model_name (str): Name of the model
        save_hmm (bool): Whether to save the model

    Returns:
        model
    
    """
    import numpy as np
    import anndata as ad
    from pomegranate import HiddenMarkovModel, State, DiscreteDistribution


    # define training sets and labels
    # Determine the number of reads to sample
    n_sample = round(0.7 * adata.X.shape[0])
    # Generate random indices
    np.random.seed(0)
    random_indices = np.random.choice(adata.shape[0], size=n_sample, replace=False)
    # Subset the AnnData object using the random indices
    training_adata_subsampled = adata[random_indices, :].copy()
    training_sequences = training_adata_subsampled.X

    # Initialize a pomegranate HMM
    model = HiddenMarkovModel(name=model_name)

    # States
    non_methylated = State(DiscreteDistribution({0: 0.98, 1: 0.02}), name="Non-Methylated")
    methylated = State(DiscreteDistribution({0: 0.02, 1: 0.98}), name="Methylated")

    # Add states to the model
    model.add_states(non_methylated, methylated)

    # Transitions
    model.add_transition(model.start, non_methylated, 0.5)
    model.add_transition(model.start, methylated, 0.5)
    model.add_transition(non_methylated, non_methylated, 0.98)
    model.add_transition(non_methylated, methylated, 0.02)
    model.add_transition(methylated, non_methylated, 0.02)
    model.add_transition(methylated, methylated, 0.98)
    model.add_transition(non_methylated, model.end, 0.5)
    model.add_transition(methylated, model.end, 0.5)

    # Build graph
    model.bake()

    # Train the HMM without labeled data
    model.fit(training_sequences, algorithm='baum-welch')

    if save_hmm:
        # Save the model to a file
        model_json = model.to_json()
        with open(f'{model_name}.json', 'w') as f:
                f.write(model_json)

    # Print the transition matrix and emission probabilities
    print("Transition matrix:")
    print(model.dense_transition_matrix())
    print("\nEmission probabilities:")
    for state in model.states:
        if isinstance(state.distribution, DiscreteDistribution):
            print(state.name, state.distribution.parameters)

    return model