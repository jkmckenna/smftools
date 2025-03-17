# classify_methylated_features

def classify_methylated_features(read, model, coordinates, classification_mapping={}):
    """
    Classifies methylated features (accessible features or CpG methylated features)

    Parameters:
        read (np.ndarray) : An array of binarized SMF data representing a read
        model (): a trained pomegranate HMM
        coordinates (list): a list of postional coordinates corresponding to the positions in the read
        classification_mapping (dict): A dictionary keyed by classification name that points to a 2-element list containing size boundary constraints for that feature.
    Returns:
        final_classifications (list): A list of tuples, where each tuple is an instance of a non-methylated feature in the read. The tuple contains: feature start, feature length, feature classification, and HMM probability.
    """
    import numpy as np

    sequence = list(read)
    # Get the predicted states using the MAP algorithm
    predicted_states = model.predict(sequence, algorithm='map')
    
    # Get the probabilities for each state using the forward-backward algorithm
    probabilities = model.predict_proba(sequence)
    
    # Initialize lists to store the classifications and their probabilities
    classifications = []
    current_start = None
    current_length = 0
    current_probs = []
    
    for i, state_index in enumerate(predicted_states):
        state_name = model.states[state_index].name
        state_prob = probabilities[i][state_index]
        
        if state_name == "Methylated":
            if current_start is None:
                current_start = i
            current_length += 1
            current_probs.append(state_prob)
        else:
            if current_start is not None:
                avg_prob = np.mean(current_probs)
                classifications.append((current_start, current_length, "Methylated", avg_prob))
                current_start = None
                current_length = 0
                current_probs = []
    
    if current_start is not None:
        avg_prob = np.mean(current_probs)
        classifications.append((current_start, current_length, "Methylated", avg_prob))

    final_classifications = []
    for start, length, classification, prob in classifications:
        final_classification = None
        feature_length = int(coordinates[start + length - 1]) - int(coordinates[start]) + 1
        for feature_type, size_range in classification_mapping.items():
            if size_range[0] <= feature_length < size_range[1]:
                final_classification = feature_type
                break
            else:
                pass
        if not final_classification:
            final_classification = classification

        final_classifications.append((int(coordinates[start]) + 1, feature_length, final_classification, prob))
    
    return final_classifications