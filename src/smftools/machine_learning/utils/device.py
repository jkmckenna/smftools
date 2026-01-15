from smftools.optional_imports import require

torch = require("torch", extra="ml", purpose="device utilities")


def detect_device():
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Detected device: {device}")
    return device
