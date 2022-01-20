import numpy as np

from typing import Union, List, Callable

class PocketPerceptron:
    """ Learn using single-cell perceptron
    
    """

    def __init__(self, 
    input: int      =10,
    eta: float      =1,
    max_iter: int   =1000,
    rand_seed: int  =37):
        """Create pocket-trained perceptron
        
        Following Gallant's theory, initialize a perceptron
        that shall be trained with the pocket algorithm. 
        Unlike Gallant's, this algorithm picks random samp-
        les in each iteration without repetition.

        Parameters:
        -----------
        input
        - Number of inputs for perceptron 
        
        eta
        - Learning rate for algorithm
        
        max_iter
        - Maximum number of iterations of pocket algorithm

        rand_seed
        - Random seed for random iterator.
        """
        self.pi         = np.zeros((input, 1))
        self.W          = np.zeros((input, 1))
        self.run_pi     = 0
        self.run_W      = 0
        self.num_ok_pi  = 0
        self.num_ok_W   = 0
        self.eta        = eta
        self.max_iter   = max_iter
        self.rand_seed  = rand_seed
    
    def solve(self, X) -> int:
        """Solve using pocket hypothesis
        
        Using the pocketed weights, the perceptron predicts
        using the stored hypothesis.

        Input
        -----
        X
        - Input data in 1xI shape, where I is the number of
          inputs.
        
        return
        ------
        Returns the linear combination's sign function
        """
        # X's shape is assumed to be 1xI
        return -1 if X @ self.W < 0 else +1
    
    def train(self):
        """"""
        pass