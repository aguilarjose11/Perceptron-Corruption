import numpy as np
import random
from typing import Tuple, Set

class Universe:
    """Data universe creator."""
    
    def __init__(self,
        dist: Tuple[Tuple[float, float], ...],
        step: float
        ) -> None:
        """Create data universe
        
        Parameters
        ==========

        dist
            - Ranges per dimension.
        
        step
            - Space between data point.
        """
        self.dist = dist
        self.n_dim = len(dist)
        self.step = step

    def gen(self) -> Set:
        universe = Set()
        universe.add()
        for _ in range(self.n_dim):

            """Algorithm:
            
            First add points that belong to first dimension.
            
            Then go through every dimension, grab an item of the already generated list and
            add the new generation list. In this way the list will increase. Use ogrid iff it is faster.
            """