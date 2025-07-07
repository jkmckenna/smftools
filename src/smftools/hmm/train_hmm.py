def train_hmm(
    data,
    emission_probs=[[0.8, 0.2], [0.2, 0.8]],
    transitions=[[0.9, 0.1], [0.1, 0.9]],
    start_probs=[0.5, 0.5],
    end_probs=[0.5, 0.5],
    device=None,
    max_iter=50,
    verbose=True,
    tol=50,
    pad_value=0,
):
    """
    Trains a 2-state DenseHMM model on binary methylation data.

    Parameters:
        data (list or np.ndarray): List of sequences (lists) with 0, 1, or NaN.
        emission_probs (list): List of emission probabilities for two states.
        transitions (list): Transition matrix between states.
        start_probs (list): Initial state probabilities.
        end_probs (list): End state probabilities.
        device (str or torch.device): "cpu", "mps", "cuda", or None (auto).
        max_iter (int): Maximum EM iterations.
        verbose (bool): Verbose output from pomegranate.
        tol (float): Convergence tolerance.
        pad_value (int): Value used to pad shorter sequences.

    Returns:
        hmm: Trained DenseHMM model
    """
    import torch
    from pomegranate.hmm import DenseHMM
    from pomegranate.distributions import Categorical
    import numpy as np
    from tqdm import tqdm

    # Auto device detection
    if device is None:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    print(f"Using device: {device}")

    # Ensure emission probs on correct device
    dists = [
        Categorical(torch.tensor([p], device=device))
        for p in emission_probs
    ]

    # Create DenseHMM
    hmm = DenseHMM(
        distributions=dists,
        edges=transitions,
        starts=start_probs,
        ends=end_probs,
        verbose=verbose,
        max_iter=max_iter,
        tol=tol,
    ).to(device)

    # Convert data to list if needed
    if isinstance(data, np.ndarray):
        data = data.tolist()

    # Preprocess data (replace NaNs + pad)
    max_length = max(len(seq) for seq in data)
    processed_data = []
    for sequence in tqdm(data, desc="Preprocessing Sequences"):
        cleaned_seq = [int(x) if not np.isnan(x) else np.random.choice([0, 1]) for x in sequence]
        cleaned_seq += [pad_value] * (max_length - len(cleaned_seq))
        processed_data.append(cleaned_seq)

    tensor_data = torch.tensor(processed_data, dtype=torch.long, device=device).unsqueeze(-1)

    # Fit HMM
    hmm.fit(tensor_data)

    return hmm