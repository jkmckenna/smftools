def load_hmm(model_path, device="cpu"):
    """
    Reads in a pretrained HMM.

    Parameters:
        model_path (str): Path to a pretrained HMM
    """
    import torch

    # Load model using PyTorch
    hmm = torch.load(model_path)
    hmm.to(device)
    return hmm


def save_hmm(model, model_path):
    """Save a pretrained HMM to disk.

    Args:
        model: HMM model instance.
        model_path: Output path for the model.
    """
    import torch

    torch.save(model, model_path)
