from smftools.logging_utils import get_logger
from smftools.optional_imports import require

logger = get_logger(__name__)


def display_hmm(hmm, state_labels=["Non-Methylated", "Methylated"], obs_labels=["0", "1"]):
    """Log a summary of HMM transition and emission parameters.

    Args:
        hmm: HMM object with edges and distributions.
        state_labels: Optional labels for states.
        obs_labels: Optional labels for observations.
    """
    torch = require("torch", extra="ml", purpose="HMM display")

    logger.info("**HMM Model Overview**")
    logger.info("%s", hmm)

    logger.info("**Transition Matrix**")
    transition_matrix = torch.exp(hmm.edges).detach().cpu().numpy()
    for i, row in enumerate(transition_matrix):
        label = state_labels[i] if state_labels else f"State {i}"
        formatted_row = ", ".join(f"{p:.6f}" for p in row)
        logger.info("%s: [%s]", label, formatted_row)

    logger.info("**Emission Probabilities**")
    for i, dist in enumerate(hmm.distributions):
        label = state_labels[i] if state_labels else f"State {i}"
        probs = dist.probs.detach().cpu().numpy()
        formatted_emissions = {obs_labels[j]: probs[j] for j in range(len(probs))}
        logger.info("%s: %s", label, formatted_emissions)
