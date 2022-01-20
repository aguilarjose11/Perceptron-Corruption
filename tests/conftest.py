import pytest
import pandas as pd

@pytest.fixture
def xor_perceptron():
    return pd.DataFrame({
        "X": [
            [+1, -1, -1],
            [+1, -1, +1],
            [+1, +1, -1],
            [+1, +1, +1]
        ],
        "y": [
            [-1],
            [+1],
            [+1],
            [-1]
        ]
    })