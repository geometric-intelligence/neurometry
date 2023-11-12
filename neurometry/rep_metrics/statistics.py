import numpy as np
import pandas as pd

# Helper function to bootstrap the neural data for a single subject
def bootstrap_neural_data(subject_neural_data, n_images):
    bootstrapped_data = {}
    for roi, data in subject_neural_data.items():
        sampled_columns = np.random.choice(data.columns, size=n_images, replace=True)
        bootstrapped_data[roi] = data[sampled_columns]
    return bootstrapped_data


# Function to bootstrap neural data for all subjects
def bootstrap_all_neural_data(neural_data, n_images):
    bootstrapped_neural_data = {}
    for subject_id, subject_data in neural_data.items():
        bootstrapped_neural_data[subject_id] = bootstrap_neural_data(
            subject_data, n_images
        )
    return bootstrapped_neural_data


def generate_multiple_bootstrap_samples(neural_data, n_images, n_bootstrap_iterations):
    """
    Generate multiple bootstrap samples of neural data.

    Parameters:
    - neural_data: dict, original neural data with subjects as keys and their neural responses as values.
    - n_images: int, number of stimulus images (columns in each DataFrame).
    - n_bootstrap_iterations: int, number of bootstrap samples to create.

    Returns:
    - all_bootstrapped_samples: list, a list where each element is a bootstrapped version of the neural_data.
    """
    all_bootstrapped_samples = []

    for _ in range(n_bootstrap_iterations):
        bootstrapped_sample = bootstrap_all_neural_data(neural_data, n_images)
        all_bootstrapped_samples.append(bootstrapped_sample)

    return all_bootstrapped_samples

