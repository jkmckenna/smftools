def load_hmm(model_path, device='cpu'):
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
    import torch
    torch.save(model, model_path)