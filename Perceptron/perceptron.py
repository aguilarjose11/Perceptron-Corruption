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
    input: int       =10,
    eta: float       =1,
    max_iter: int    =1000,
    rand_seed: int   =37,
    ignore_flag: bool=False,
    patience: int    =15):
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

        ignore_flag
        - Ignores warning given when convergance not reached.
        
        patience
        - Number of consecutive times weights are unchanged to assume convergance.
        """
        self.input = input
        self.pi         = np.random.rand(input, 1)
        self.W          = np.random.rand(input, 1)
        self.run_pi     = 0
        self.run_W      = 0
        self.num_ok_pi  = 0
        self.num_ok_W   = 0
        self.eta        = eta
        self.max_iter   = max_iter
        self.rand_seed  = rand_seed
        self.ignore_flag= ignore_flag
        self.patience   = patience
    
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
        #import pdb; pdb.set_trace()
        count_w = 0
        self.num_ok_pi = self.num_ok_W = self.run_pi = self.run_W = 0
        for i in range(self.max_iter):
            prev_w = self.W # Used for patience
            # Shuffle order in which training occurs
            index = random.sample(range(len(X)), len(X))
            E = X[index]
            C = y[index]
                
            if self.learn(E, C):
                return
            elif np.array_equal(prev_w, self.W):
                # The parameters did not change!
                count_w += 1
            else:
                # New parameters that were learned.
                count_w = 0
            # Have we converged?
            if count_w >= self.patience:
                # Patience reached, assume convergance has been achieved.
                return
        if not self.ignore_flag:
            print("Maximum iterations reached: No convergence. Ignore if data is non-separable")
        #import pdb; pdb.set_trace()


    def __num_ok(self,
        X: array,
        y: array
        ):
        """Calculate total number of correct values in entire dataset"""

        self.num_ok_pi  = 0
        self.num_ok_W   = 0
        # Equivalent to solving but with pi hypothesis instead
        z_pi =  np.sign(X @ self.pi)
        z_pi[z_pi == 0] = -1 
        z_W = self.solve(X)
        self.num_ok_pi = np.count_nonzero(z_pi == y)
        self.num_ok_W  = np.count_nonzero(z_W == y)

    def learn(self, 
        X: array, 
        y: array
        ):
        #import pdb; pdb.set_trace()
        
        for E, C in zip(X, y):
            z = E @ self.pi
            pi_C    = -1 if z < 0 else +1
            if pi_C == C:
                self.run_pi += 1
                if self.run_pi > self.run_W:
                    self.__num_ok(X, y)
                    if self.num_ok_pi >= self.num_ok_W:
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
