import numpy as np
import pandas as pd
from typing import Union, Tuple

DataFrame = pd.DataFrame

def pocket_complexity(
        epsilon: float,
        epsilon_knot: float,
        delta: float,
        p: int,
        L: float,
        ) -> float:
    """ Compute Gallant's Bound
    Parameters
    ----------
    epsilon: float
        Actual Error of network
    epsilon_knot: floa
        Measured error on training set of E
    delta: float
        Probability
    p: int
        Number of input neurons
    L: float
        Length of the weight vector.
    """

    # Compute slack
    s = (epsilon - epsilon_knot) / epsilon
    # Leftmost fraction 8 / s^2 * epsilon
    c_fract = (8 / (s ** 2 * epsilon))
    # Max operands
    a = np.math.log(8 / delta)
    b = min(2 * (p + 1), 4 * (p + 1) * np.math.log10(np.math.e)) * np.math.log(16 / (s ** 2 * epsilon))
    c = max(a, b)
    # Compute first operand of outmost minimum
    A = c_fract * c

    # Second large fraction [ln (1/delta) + (p+1) * ln(2L+q)] / [s^2 * epsilon]
    d_fract = (np.math.log(1 / delta) + (p + 1) * np.math.log(2 * L + 1)) / (s ** 2 * epsilon)
    # Compute second operand of outmost minimum
    B = d_fract * min(1 / (2 * epsilon), 2)
    # Ceil since we are looking at sample complexity being int
    return np.math.ceil(min(A, B))


def epsilon_pocket_complexity(
        total_data: int,
        total_buckets: int,
        epsilon_knot: float,
        delta: float,
        vc_dim: int,
        L: int,
        resolution: int = 10_000_000
):
    """Compute Gallant accuracy to samples per bucket matrix

    Algorith used for approximating the true error rate given an experiment's average error. The bound used here is the
    gallant bound which can be found in Gallant's 1990 "Perceptron-Based Learning Algorithms" paper.

    Parameters
    ----------
    total_data: int
        Total data used. Used for computing samples per bucket.
    total_buckets: int
        Number of corruption buckets used. Used for finding number of samples relevant.
    epsilon_knot: int
        Experimental error. Will be converted to accuracy in code.
    delta: float
        Probability
    vc_dim: int
        Number of data parameters
    L: int
        Length of weight vector
    resolution: int
        Resolution of epsilon sampling. The larger this value, the smaller the change applied to epsilon, decreasing the
        possibility of "jumping over" samples. See try/except in for loop
    """
    # vc-dim - 1 == p

    # To contain score per # of datapoints
    # By score I mean accuracy in this case
    accuracy_to_corruption = {'accuracy': [], 'corruption': []}

    # Will create dictionary of epsilon given some number of samples
    epsilon_samples = np.linspace(0.05, 0.999999, resolution)  # Used in sampling for sample complexity.
    # Collects the sample lower bound, relating it with the epsilon obtained.
    samples_to_epsilon = {
        pocket_complexity(epsilon=epsilon, epsilon_knot=epsilon_knot, delta=delta, p=vc_dim - 1, L=L): epsilon for
        epsilon in epsilon_samples
    }
    # Ratio of data samples per bucket
    data_per_buckets = total_data // total_buckets
    # Dictionary relating the number of samples given some number of buckets
    buckets_to_samples = {buckets: data_per_buckets * buckets for buckets in range(1, total_buckets + 1)}

    for buckets, n_data in buckets_to_samples.items():
        try:
            # Collect 1 - epsilon (accuracy) w.r.t. number of sample data.
            accuracy_bound = 1 - samples_to_epsilon[n_data]
        except KeyError:
            # There was no epsilon that gave the sample lower boutn n_data.
            # This is probably because the resolution was too low.
            accuracy_bound = np.NaN
        # Store accuracy with respect to buckets
        accuracy_to_corruption[f'accuracy'].append(accuracy_bound)
        accuracy_to_corruption['corruption'].append(100 - buckets)

    return pd.DataFrame(accuracy_to_corruption)


def epsilon_approximation_gallant(T_prime: int,
                                  epsilon_knot: float,
                                  delta: float,
                                  p: int,
                                  L: float,
                                  resolution: int = 100,
                                  method: str = 'binary',
                                  accuracy: bool = False,
                                  ) -> Tuple[float, float]:
    """Approximate epsilon value given experiment results.

    Algorith used for approximating the true error rate given an experiment's average error. The bound used here is the
    gallant bound which can be found in Gallant's 1990 "Perceptron-Based Learning Algorithms" paper.

    Parameters
    ----------
    T_prime: int
        Total dataset used for training. Will be used for comparison when approximating epsilon.
    epsilon_knot: float
        Training error obtained during training.
    delta: float
        Probability for bound.
    p: int
        Number of parameters used for training.
    L: float
        Length of weight vector: ||W||_2
    resolution: int
        Number of values to approximate epsilon.
    method: str
        Method to use for approximating epsilon. If continuous, algorithm will increment 1/resolution starting from a
        pre-set minimum being epsilon_knot up to 1. If binary search, it will use resolution binary searches to find
        epsilon.
    accuracy: bool
        Flag for returning true accuracy instead of true error rate.
    """
    error_and_sample: Tuple[float, float]
    if method == 'continuous':
        pass
    elif method == 'binary':
        # Make first split
        upper_bound = 1
        lower_bound = epsilon_knot
        epsilon_curr = epsilon_knot + (upper_bound - epsilon_knot)/2

        T_curr = pocket_complexity(epsilon_curr, epsilon_knot, delta, p, L)
        # Perform binary search
        for split in range(resolution - 1):
            # Obtain comparison with respect to training T
            T_diff = T_curr - T_prime
            if T_diff == 0:
                # We found exactly what epsilon is! Should we remove ceil from bound?
                break
            elif T_diff > 0:
                # Sample complexity obtained is too large, meaning that the epsilon is too close to
                # 0. Increase epsilon instead.
                lower_bound = epsilon_curr
                epsilon_curr += (upper_bound - epsilon_curr) / 2

            else:
                # Sample complexity obtained is too small, meaning that the epsilon is too close to
                # 1. Decrease epsilon;
                upper_bound = epsilon_curr
                epsilon_curr -= (epsilon_curr - lower_bound) / 2
            # Re-calculate split.
            T_curr = pocket_complexity(epsilon_curr, epsilon_knot, delta, p, L)
        # Convert to accuracy if preffered
        if accuracy:
            epsilon_curr = 1 - epsilon_curr
        # Return best obtained bound.
        error_and_sample = epsilon_curr, T_curr
        return error_and_sample


