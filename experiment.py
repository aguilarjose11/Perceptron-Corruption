import numpy as np
import pandas as pd
import Perceptron.perceptron as pn
from Perceptron.data_gen import Universe, separable_regression
from sklearn.metrics import accuracy_score

from typing import List, Tuple
import random
import pickle


def corrupt_data(
    universe_len:  int,
    training_size: int,
    testing_size:  int,
    buckets:       int,
    )->Tuple[List[int], List[int]]:
    """Create uniform corruption buckets.
    
    The function shall pick random indices within the entire 
    data universe. The number of indices will be the same as
    abscribed in the call of the function. Then, the indices
    will be split into buckets of several sizes. These shall
    be returned.
    """
    
    indices    = np.array(random.sample(range(universe_len), training_size + testing_size))
    training   = indices[:training_size]
    testing    = indices[training_size:]
    assert len(testing) == testing_size
    
    bucket_len = int(len(training) / buckets)
    
    return np.array([training[bucket_len*i:bucket_len*(i+1)] for i in range(buckets)]), testing

# Data Generation

X, y = separable_regression(
    weights  =np.array([[13.5], [-4.3], [6.4]]),
    dim_dist =((-10, 10), (-10, 10), (1, 1)),
    dim_step =(100, 100, 1))
# Parameters for dataset creation
training_size =2000
testing_size  =400
n_buckets     = 50
# Pocket Algorithm hyper-parameters
max_iter      = 100
eta           = 1

# Obtain indices
train, test   = corrupt_data(
    universe_len=len(y),
    training_size=training_size,
    testing_size=testing_size,
    buckets=n_buckets)
# Perform corruption
n_runs = 100
exp_data = dict()
# Experiment
for run in range(n_runs):
    print(f"Start of run {run}.")
    #import pdb; pdb.set_trace()
    # shuffle bin indices
    bucket_index = random.sample(range(n_buckets), n_buckets)
    # begin bining
    empirical_score = []
    #true_score      = []
    for bins in range(1, n_buckets):
        
        # Create model; No innate bias included!
        model = pn.PocketPerceptron(
            input=3, 
            eta=eta, 
            max_iter=max_iter
        ) 
        # Grab training data
        m = np.concatenate(X[train[bucket_index[:bins]]])
        print(f"Training with {bins} buckets -- {len(m)}")
        # Train model
        model.train(m, np.concatenate(y[train[bucket_index[:bins]]]))
        # Store empirical and "true"
        pred = model.solve(X[test])
        empirical_score.append(accuracy_score(y[test], pred))
        #true_score.append(accuracy_score(y, model.solve(X)))
        
    exp_data[run] = empirical_score # (empirical_score, true_score) 
        
# Store results
pkl = open('exp_corruption.pkl', 'wb')
pickle.dump(exp_data, pkl)
pkl.close()
