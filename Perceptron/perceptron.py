import numpy as np
import random
from typing import Tuple, Union, List, Callable

array = np.array

class PocketPerceptron:
    """ Learn using single-cell perceptron
    
    Functions:
    ----------

    solve:
    - Calculate perceptron inference.

    train:
    - Main training function implementing pocket algorithm.

    learn:
    - Implement core pocket algorithm.
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
        self.input = input
        self.pi         = np.ones((input, 1))
        self.W          = np.ones((input, 1))
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
        activation = np.sign(X @ self.W)
        activation[activation == 0] = -1
        return activation
    
    def train(self, X, y):
        """Train Perceptron Model"""
        for i in range(self.max_iter):
            index = random.sample(range(len(X)), len(X))
            E = X[index]
            C = y[index]
            if self.learn(E, C):
                return
        print("Maximum iterations reached: No convergence. Ignore if data is non-separable")


    def __num_ok(self,
        X: array,
        y: array
        ):
        """Calculate total number of correct values in entire dataset"""

        self.num_ok_pi  = 0
        self.num_ok_W   = 0
        for E, C in zip(X, y):
            pi_y    = -1 if E @ self.pi < 0 else +1
            W_y     = -1 if E @ self.W < 0 else +1
            self.num_ok_pi += 1 if pi_y == C else 0
            self.num_ok_W += 1 if W_y == C else 0

    def learn(self, 
        X: array, 
        y: array
        ):
        #import pdb; pdb.set_trace()
        for E, C in zip(X, y):
            pi_C    = -1 if E @ self.pi < 0 else +1
            if pi_C == C:
                self.run_pi += 1
                if self.run_pi > self.run_W:
                    self.__num_ok(X, y)
                    if self.num_ok_pi > self.num_ok_W:
                        self.W = self.pi
                        self.run_W = self.run_pi
                        self.num_ok_W = self.num_ok_pi
                        if self.num_ok_W == len(y):
                            # Correct classification overall
                            return True
            else:
                self.pi = self.pi + (C * E).reshape((self.input, 1))
                self.run_pi = 0
        return False

def true_error(dist: Tuple[List, List], model):
    miss = 0
    for E, C in dist:
        y_pred = model.solve(E)
        if y_pred != C:
            miss += 1

    return miss / len(dist[0])