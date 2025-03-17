def display_hmm(hmm, state_labels=["Non-Methylated", "Methylated"], obs_labels=["0", "1"]):
    import torch
    print("\n🔹 **HMM Model Overview**")
    print(hmm)

    print("\n🔹 **Transition Matrix**")
    transition_matrix = torch.exp(hmm.edges).detach().cpu().numpy()
    for i, row in enumerate(transition_matrix):
        label = state_labels[i] if state_labels else f"State {i}"
        formatted_row = ", ".join(f"{p:.6f}" for p in row)
        print(f"{label}: [{formatted_row}]")

    print("\n🔹 **Emission Probabilities**")
    for i, dist in enumerate(hmm.distributions):
        label = state_labels[i] if state_labels else f"State {i}"
        probs = dist.probs.detach().cpu().numpy()
        formatted_emissions = {obs_labels[j]: probs[j] for j in range(len(probs))}
        print(f"{label}: {formatted_emissions}")