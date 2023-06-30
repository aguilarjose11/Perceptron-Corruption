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

@pytest.fixture
def learn_xor_pocket():
    return{
        "X": np.array([
            [+1, +1, +1],
            [+1, +1, +1],
            [+1, -1, +1],
            [+1, +1, -1],
            [+1, +1, +1],
            [+1, -1, +1],
            [+1, +1, -1],
            [+1, -1, -1],
        ]),
        "y": np.array([
            [-1],
            [-1],
            [+1],
            [+1],
            [-1],
            [+1],
            [+1],
            [-1]
        ]),
        "pi": np.array([
            [0],
            [0],
            [0]
        ]),
        "W": np.array([
            [1],
            [-1],
            [-1]
        ]),
        "run_pi": 0,
        "run_W": 1,
    }

@pytest.fixture
def perceptron_comp():
    return {
        "W_1": np.array([
            [4],
            [3],
            [5],
            [6],
            [1],
            [2],
        ]),
        "W_2": np.array([
            [-1],
            [2],
            [-1],
            [1],
            [-1],
            [-1],
        ]),
        "diff": 0.5
    }