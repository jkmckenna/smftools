# read_HMM

def read_HMM(model_path):
    """
    Reads in a pretrained HMM.
    
    Parameters:
        model_path (str): Path to a json of the pretrained HMM
    """
    from pomegranate import HiddenMarkovModel, DiscreteDistribution
    with open(model_path, 'r') as f:
        model_json = f.read()

    model = HiddenMarkovModel.from_json(model_json)

    # Print the transition matrix and emission probabilities
    print("Transition matrix:")
    print(model.dense_transition_matrix())
    print("\nEmission probabilities:")
    for state in model.states:
        if isinstance(state.distribution, DiscreteDistribution):
            print(state.name, state.distribution.parameters)

    return model