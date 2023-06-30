#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
import Perceptron.perceptron as pn
from Perceptron.data_gen import Universe, separable_regression, data_distribution
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, zero_one_loss
from sklearn import datasets
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import math
from tqdm import tqdm
import argparse
import random


## Data Corruption Experiment
from typing import List, Tuple
import random

def corrupt_data(
    universe_len:  int,
    buckets:       int,
    test_split:    float=None,
    )->Tuple[List[int], List[int]]:
    """Create uniform corruption buckets.
    
    The function shall pick random indices within the entire 
    data universe. The number of indices will be the same as
    abscribed in the call of the function. Then, the indices
    will be split into buckets of several sizes. These shall
    be returned.
    
    Testing will not be split into buckets if specified.
    """
    
    assert 0 < test_split < 1 or test_split == None, "Invalid test split value"
    
    # Create shuffled indices
    indices  = np.random.choice(universe_len, universe_len, replace=False)
    
    training = testing = None
    
    if test_split is not None:
        split      = int(universe_len - (test_split * universe_len))
        training   = indices[:split]
        testing    = indices[split:]
    else:
        training = indices
    
    bucket_len = int(len(training) / buckets)
    # Split into buckets
    train_buckets = [ training[ bucket_len*i:bucket_len*(i+1) ] for i in range(buckets)]
    
    return np.array(train_buckets), testing


def sample_complexity(vc_dim, epsilon, delta):
    return math.ceil(
        (4/epsilon) * (vc_dim*math.log(12/epsilon) + math.log(2/delta))
    )


def pickle_data(
    root_dir, 
    results,
    args):
    
    # Make sure it is a directory!
    if root_dir[-1] != '/':
        root_dir += '/'
    
    # Create pickle structure
    pkl = {
        'results': dict(results),
        'args':    args,
    }
    
    
    # Create file name

    file_name = f"{args.label}_test_size_{args.test_size}.pkl"
    
    with open(f"{root_dir}{file_name}", 'wb') as pkl_file:
        pickle.dump(pkl, pkl_file)



def experiment(
    X,
    y,
    metric,
    test_split:   float,
    buckets:       int,
    n_runs:        int,
    verbose:       bool,
    n_buckets:     int, 
    max_iter:      int,
    eta:           float,
    ):
    
    assert len(X) == len(y), 'Shapes of input data and labels does not match!'
    
    # Bukcetize data
    training_size = int(len(X)*0.8)
    testing_size  = len(X) - training_size 
    train, test = corrupt_data(universe_len=len(X), 
                               buckets=buckets,
                               test_split=test_split)
    
    # Create dictionary to store results
    exp_data = defaultdict(lambda : [])
    
    # Experiment
    for run in range(n_runs):
        if verbose > 0:
            print(f"Start of run {run}.")
        
        
        # begin bining
        empirical_score = []
        for bins in range(1, n_buckets):

            # Create model; No innate bias included!
            model = pn.PocketPerceptron(
                input=X.shape[-1], 
                eta=eta, 
                max_iter=max_iter
            ) 
            
            # Grab training data
            m      = np.concatenate(X[train[:bins]])
            labels = np.concatenate(y[train[:bins]])
            
            if verbose > 1:
                print(f"Training with {bins} buckets -- {len(m)}")
            
            # Train model
            model.train(m, labels)
            
            # Store risk data
            if testing_size: 
                pred = model.solve(X[test])
                exp_data[bins].append(metric(y[test], pred))
            
            else: # No empirical testing. Take error over all data.
                pred = model.solve(X)
                exp_data[bins].append(metric(y, pred))

            #true_score.append(accuracy_score(y, model.solve(X)))
        #import pdb; pdb.set_trace()
    return dict(exp_data)


def make_parser():
    parser = argparse.ArgumentParser(description='Corruption Experiment', fromfile_prefix_chars='@')
    
    parser.add_argument('--test_size',     type=float, default=0.2, help='Test set portion')
    parser.add_argument('--buckets',       type=int, default=20, help='Number of buckets to split data into')
    parser.add_argument('--epochs',        type=int, default=10, help='Number of times experiment is run')
    parser.add_argument('--verbose', '-v',  action='count', default=0, help='Verbosity')
    parser.add_argument('--max_buckets',   type=int, default=19, help='Maximum number of buckets used')
    parser.add_argument('--max_iter',      type=int, default=100, help='Maximum number of iterations for convergance') #TODO Continue adding parameters!
    parser.add_argument('--learning_rate', type=float, default=1, help='Perceptron learning rate (eta)')
    parser.add_argument('--store_at',      type=str, default='pickles/', help='Directory where to save results')
    parser.add_argument('--label',         type=str, default='experiment', help='Label to add to pickle file')
    parser.add_argument('--data',          type=str, default=None, help='Select dataset. (sep_50_50), (sep_21_21_21), (banknotes)')
    parser.add_argument('--bounds',        type=int, nargs='+', default=[-10, 10, -10, 10], help='Bounds for each dimension. Numbers in pairs will be counted for each dimension.')
    parser.add_argument('--n_points',      type=int, default=100, help='Number of points to be drawn.')
    parser.add_argument('--concept',        type=int, nargs='+', default=[1, 1, -1], help='Concept parameters (include bias).')
    return parser

def main(args):
    datasets = ['sep_50_50', 'sep_21_21_21', 'banknotes']
    
    # Generate dataset
    if args.data is not None:
        assert args.data in datasets, 'Invalid dataset!'
        
        if args.data == datasets[0]:
            X, y = separable_regression(
                weights  =np.array([[605], [-358], [73]]),
                dim_step =(50, 50, 1))
        elif args.data == datasets[1]:
            X, y = separable_regression(
                weights  =np.array([[605], [-358], [-124], [73]]),
                dim_dist = ((-1, 1), (-1, 1), (-1, 1), (1, 1)), 
                dim_step =(21, 21, 21, 1))
        elif args.data == datasets[2]:
            with open('bank_note_dataset.pkl', 'rb') as pkl_file:
                data = pickle.load(pkl_file)
            X = np.array(data['X'])
            y = np.array(data['y'])

    else:
        assert len(args.bounds) % 2 == 0, 'Missing bounds: Odd number of values passed!'
        # bounds does not include bias (Always 1), concept must include it.
        assert len(args.bounds) // 2 == len(args.concept) - 1, f'Invalid number of dimensions inferred by bounds and concept! {len(args.bounds) % 2} vs {len(args.concept) - 1}'
        bounds = []
        for i in range(0, len(args.bounds), 2):
            b = [args.bounds[i], args.bounds[i+1]]
            bounds.append(b)
        X = []
        y = []
        c = pn.PocketPerceptron()
        c.W = np.array(args.concept).reshape([len(args.concept), 1])
        
        for _ in range(args.n_points):
            point, target = data_distribution(concept=c.solve, bounds=bounds, distribution=random.random, bias=True)
            X.append(point)
            y.append(target)
        X = np.array(X)
        y = np.array(y)
        import pdb; pdb.set_trace()
    # Select metric
    metric = zero_one_loss
    
    results = experiment(
        X          = X, 
        y          = y, 
        metric     = metric, 
        test_split = args.test_size, 
        buckets    = args.buckets, 
        n_runs     = args.epochs, 
        verbose    = args.verbose, 
        n_buckets  = args.max_buckets, 
        max_iter   = args.max_iter, 
        eta        = args.learning_rate)
    
    if args.store_at is not None:
        #pickle data!
        pickle_data(root_dir=args.store_at, results=results, args=args)


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    main(args)