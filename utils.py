import numpy as np

def extract_probabilities(output, num_queries):
    """
    Extract query probabilities from the terminal output of our pasp experiments.

    Parameters:
    - output (str): Combined stdout and stderr from pasp.
    - num_queries (int): Number of queries expected.

    Returns:
    - probabilities (list): List of extracted probabilities.
    """
    lines = output.strip().split('\n') 
    probabilities = []
    for line in lines:
        if line.startswith('ℙ('):
            # The line we are looking for looks like this: 'ℙ(was_deal) = 0.500000'
            parts = line.split('=')
            if len(parts) == 2:
                prob_str = parts[1].strip()
                try:
                    prob = float(prob_str)
                except ValueError:
                    prob = float('nan')
                probabilities.append(prob)
            else:
                probabilities.append(float('nan'))

    # Ensure we only return the expected number of queries
    # This is needed to guarantee consistency when calculating the plots
    return probabilities[:num_queries]

def generate_nmodels_list(estimated_models):
    """
    Generate a dynamic nmodels_list that increases rapidly at first and slows down towards the end,
    using a multiplicative approach with decreasing steps.

    Returns:
    - nmodels_list (list): List of nmodels values.
    """
    k = np.linspace(0, 1, num=40)
    s = k ** 0.5  # Adjust exponent to control the descent rate (0.5 can be modified)
    logs = s * np.log(estimated_models / 2) # Multiply by the log ratio of B_n and B_0=2
    nmodels = 2 * np.exp(logs)
    nmodels = np.unique(np.round(nmodels).astype(int))
    nmodels[0] = 2  # Ensure the first element is 2
    nmodels[-1] = estimated_models  # Ensure the last element matches estimated_models
    return list(nmodels)