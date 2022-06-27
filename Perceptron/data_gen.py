import numpy as np
import random
from typing import Tuple, Set, List, Callable
from .perceptron import PocketPerceptron


sample = Tuple[float]


class Universe:
    """Data universe creator."""
    
    def __init__(self,
        dist: Tuple[Tuple[float, float], ...],
        dim: Tuple[int, ...]
        ) -> None:
        """Create data universe
        
        Parameters
        ==========

        dist
            - Ranges per dimension.
        
        step
            - Space between data point.
        """
        assert len(dist) == len(dim), f"Dimensions of descriptors not matching: {len(dist)} != {len(dim)}"
        self.dist = dist
        self.n_dim = len(dist)
        self.dim = dim

    def gen(self) -> List:
        universe = np.linspace(*self.dist[0], self.dim[0]).reshape((self.dim[0], 1))
        for dist, dim in zip(self.dist[1:], self.dim[1:]):
            universe = [np.concatenate((x, y)) for x in universe for y in np.linspace(*dist, dim).reshape((dim, 1))]
        return np.array(universe)
            

def separable_regression(
    weights: Tuple[float, ...], 
    dim_dist: Tuple[Tuple[float, float], ...], 
    dim_step: Tuple[int, ...]
    ) -> Tuple[List[float], List[bool]]:
    
    model   = PocketPerceptron(input=len(dim_dist))
    model.W = weights

    X = Universe(dim_dist, dim_step).gen()
    y = model.solve(X)

    return X, y

def data_distribution(
    concept:      Callable[[sample], bool], 
    bounds:       Tuple[Tuple[float, float]],
    distribution: Callable[[None], float],
    bias = True
) -> Tuple[sample, bool]:
    data_point = []
    for a, b in bounds:
        point = distribution() * abs(a - b) + a
        data_point.append(point)
    
    if bias:
            data_point += [1]
    label = concept(data_point)

    return data_point, label

