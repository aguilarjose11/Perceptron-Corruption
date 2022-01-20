import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def xor_perceptron():
    return {
        "X": np.array([
            [+1, -1, -1],
            [+1, -1, +1],
            [+1, +1, -1],
            [+1, +1, +1]
        ]),
        "y": np.array([
            [-1],
            [+1],
            [+1],
            [-1]
        ]),
        "optimal_w": np.array([
            [1], 
            [-1], 
            [-1]
        ]),
        "optimal_y":np.array([
            [+1],
            [+1],
            [+1],
            [-1]
        ])
    }