#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import Perceptron.perceptron as pn
from Perceptron.data_gen import Universe, separable_regression, data_distribution
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, zero_one_loss
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import datasets
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import math
from argparse import ArgumentParser, RawTextHelpFormatter 
import random
import matplotlib.pyplot as plt


## Data Corruption Experiment
from typing import List, Tuple
import random
import os

# In[ ]:


# Utility Functions
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


# ## Theoretical Machine Learning Functions
# 
# The functions bellow provide wrappers for better interpreting theory from the literature.

# In[ ]:


# Experiment required functions
def sample_data(
    lows:      List[float],
    highs:     List[float],
    n_samples: int,
    seed:      int=None
) -> List[List[int]]:
    """Sample uniform distribution bounded by lows and highs
    
        Using a uniform distribution, perform sampling over the 
    distribution such that the space the distribution is sampling will 
    be bounded by the given bounds from the lows and highs. Lows and 
    highs will be arrays that contain the minimum and maximum values 
    per dimension on the data to be samples. For example, if we have 4 
    values in both lows and highs, then, at the time of sampling n_samples
    samples we will have n_samples of 4 attributes each: (n_samples, 4).
    """
    
    assert len(lows) == len(highs), f"Non-matching lows and highs: {len(lows) != {len(highs)}}"
    
    rng = np.random.default_rng(seed)
    data_shape = (n_samples, len(lows)) # See assertion #1
    data = rng.uniform(lows, highs, data_shape)
    return data

# splitting the dataset into bins can be done with: np.split(data, n_buckets)
# Recommend shuffling beforehand tho.

class Concept:
    """Label given data
    Using a model as truth, label given data.
    """
    def __init__(self, model):
        self.model = model
        
    def __call__(self, X):
        return self.model.solve(X)
    
    
class NPolynomial:
    
    def __init__(self, 
                 n:    int, 
                 low:  float=0, 
                 high: float=1, 
                 seed: int=42
                ):
        self.n     = n
        self.seed  = seed
        self.low   = low
        self.high  = high
        rng        = np.random.default_rng(seed)
        self.coeff = rng.uniform(low, high, (n, 1))
        self.exps  = [exp for exp in range(1, n+1)[::-1]]
        
    def solve(self, vals):
        var = np.power(vals, self.exps)
        activation = np.sign(var @ self.coeff)                                        
        activation[activation == 0] = -1
        return activation
    


# ## Data Corruption Functions and Experiment
# 
# These functions are used for carrying out data corruption and experiments.

# In[ ]:


'''Data Corruption Code for Experiments'''

def perceptron_data_corruption(
        train_data,
        train_labels,
        test_data,
        test_labels,
        model_params,
        verbose,
        history,
        seed,
        ):
        '''Corrupt given bucketized data

        Parameters
        ----------

        train_data
            Training data that is already bucketized
        train_labels
            Training labels that is already bucketized
        test_data
            Testing dataset
        test_labels
            Testing dataset labels
        model_params
            Perceptron model hyperparameters to train on
        verbose
            specify verbosity of messages
        history
            Shall be a dictionary that has initialized key-value pairs.
            The keys shall contain all of the buckets to be used. The
            values shall be lists that may or may not already contain
            scores from previous runs.
        seed
            Random seed used when choosing indices
        '''

        # Calculate number of buckets user is passing.
        n_buckets = len(train_data) # could add min/max params to specify buckets!
        
        rng = np.random.default_rng(seed)
        L_values = []
        
        # Begin with high corruption and then add more buckets
        for buckets in range(1, n_buckets + 1):

            if verbose > 1:
                print(f"\tBuckets used: {buckets}", flush=True)
            # Choose buckets to be used.
            indices = rng.choice(range(0, len(train_data)), size=buckets, replace=False)
            X       = np.concatenate(train_data[indices])
            Y       = np.concatenate(train_labels[indices])
            if verbose > 2:
                print(f"\tData points used: {len(X)}", flush=True)

            # Train model
            model = pn.PocketPerceptron(**model_params)
            model.train(X, Y)
            pred = model.solve(test_data)

            # Measure zero-one & store
            score = accuracy_score(pred, test_labels)
            #score_list.append(score)

            if verbose > 3:
                print(f"\t\tScore: {score}", flush=True)
            history[buckets].append(score)
            # Used by Gallant's learning bound.
            L_values.append(np.linalg.norm(model.W))
            
        history['L'].append(L_values)
        
        return history
    

def perceptron_corruption_experiment(
    X,
    y,
    test_size,
    train_size,
    n_buckets,
    model_params,
    n_runs,
    seed,
    verbose,
    ):
    '''Conduct corruption experiment and report results
    
    parameters
    ----------
    X
        Dataset to train/test on.
    y
        Labels of dataset to train/test on.
    test_size
        Size of testing dataset to be split into. See StratifiedShuffleSplit f-
        rom sklearn. can be either a fraction or an exact size.
    train_size
        Size of training dataset to be split into. can be a fraction or exact
        size.
    n_buckets
        Number of buckets to split training data into.
    model_parameters
        Dictionary containing Pocket Perceptron algorithm's constructor parame-
        ters. See Perceptron.perceptron.PocketPerceptron for list.
    n_runs
        Number of experiment iterations where during each iteration data is co-
        rrupted progressively.
    seed
        Random seed for generators used in concept and model initialization.
    verbose
        Specify verbosity of output.
    '''
    
    # Will have n_runs scores per bucket size used for training.
    history   = {buckets: [] for buckets in range(1, n_buckets + 1)} # +1 for no corruption
    # Magnitude of learned vector. Will contain lists of magnitudes.
    history['L'] = []
    

    
    for run in range(n_runs):
        if verbose > 0:
            print(f"Run #{run}", flush=True)
            
        '''Creation of training/testing datasets (bucketized)'''
        sss = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=test_size, 
            train_size=train_size,
            random_state=None #seed + run # This way data is shuffled differently every run!
        )
        for train_i, test_i in sss.split(X, y):
            train_data, train_labels = X[train_i], y[train_i]
            test_data, test_labels = X[test_i], y[test_i]
        # We just need to bucketize the training data now (Testing data used as is)
        train_data   = np.array_split(train_data, n_buckets) # split rises exception if not even!
        train_data   = np.array(train_data) # Helps in keeping bucket structure
        train_labels = np.array_split(train_labels, n_buckets)
        train_labels = np.array(train_labels)


        ''' Conduct corruption and obtain scores '''
        history = perceptron_data_corruption(
            train_data,
            train_labels,
            test_data,
            test_labels,
            model_params,
            verbose,
            history,
            seed=run + seed,
        )
        
    return history


# # Data Corruption Experiment

def create_parser():
    parser = ArgumentParser(description='CoLT Experiment', formatter_class=RawTextHelpFormatter)
    dataset_help = '''
Experiment to conduct. There are 4 designed and implemented.
    
syn-lin: Synthetic Linearly-Separable
syn-non: Synthetic Non Linearly-Separable
iris: Iris (Linearly-Separable)
skin: Skin/No Skin (Non Linearly-Separable)
    '''
    parser.add_argument('-e', '--experiment', type=str, help=dataset_help, required=True)
    lower_bound_help = '''
Lower bounds per dimension to use when using a synthetic dataset. Shall be a list of float values.
    '''
    parser.add_argument('-l', '--lower_bounds', nargs='+', type=float, default=[-10, -10, -10, -10], help=lower_bound_help,)
    upper_bound_help = '''
Upper bounds per dimension to use when using a synthetic dataset. Shall be a list of float values.
    '''
    parser.add_argument('-u', '--upper_bounds', nargs='+', type=float, default=[10, 10, 10, 10], help=upper_bound_help, )
    parser.add_argument('--bias', action='store_true', help='Flag for using bias.')
    parser.add_argument('--dataset_size', type=int, help='Number of datapoint to sample for dataset.', required=True)
    #
    parser.add_argument('-t', '--test_fraction', type=float, help='Fraction of whole dataset to use as testing.', default=0.2)
    parser.add_argument('-b', '--n_buckets', type=int, help='Number of buckets to split data into.', default=20)
    parser.add_argument('-r', '--n_runs', type=int, help='Number of times to repeat experiment.', default=10)
    #
    parser.add_argument('--eta', type=float, default=1, help='Learning rate of perceptron.' )
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum number of Perceptron iterations before convergance is assumed.')
    parser.add_argument('--w_init', nargs='+', type=float, default=[0.5, 0.5], help='Initial weight distribution [lower, upper] bounds.')
    # 
    parser.add_argument('-v', '--verbose',action='count', default=0, help='Verbosity of messages.' )
    parser.add_argument('-i', '--index', type=str, default='0', help='Inex of experiments. Helpful when running multiple repetitions of same experiment.')
    parser.add_argument('--result_root', type=str, default='.', help='Directory to store results.')
    parser.add_argument('-p', '--patience', type=int, help='Number of iterations where weights have not changed before assuming convergance.', default=20)
    parser.add_argument('-d', '--data', type=str, default=None, help='Specify a data file for provided data type.')

    return parser
    
def obtain_data(args):
    '''Return Data and Labels based on experiment to run'''
    
    experiment = args.experiment
    # Initial distribution of perceptron weights
    w_init_lows, w_init_highs = args.w_init
    
    if experiment == 'syn-lin': # Synthetic data experiment
        '''Generate Data'''
        # Lower and upper bounds for data distribution PER dimension
        # Bias only added if specified!
        dataset = 'datasets/syn_lin_data.pkl' if (args.data is None) else args.data
        if args.verbose > 0:
            print(f'Loaded dataset {dataset}', flush=True)
        with open(dataset, 'rb') as jar:
            data_pkl = pickle.load(jar) 
        syn_lin = pd.DataFrame(data_pkl['X'])
        if args.bias:
            syn_lin['bias'] = 1

        data = syn_lin.to_numpy()
        labels = data_pkl['y']
        '''Select appropirate concept (linear or non-linear)'''
        
    elif experiment == 'syn-non': # Use a non-linearly-separable concept (an 'ins'-degree polynomial)
        dataset = 'datasets/syn_non_data.pkl' if (args.data is None) else args.data
        if args.verbose > 0:
            print(f'Loaded dataset {dataset}', flush=True)
        with open(dataset, 'rb') as jar:
            data_pkl = pickle.load(jar)
        syn_non = pd.DataFrame(data_pkl['X'])
        if args.bias:
            syn_non['bias'] = 1

        data = syn_non.to_numpy()
        labels = data_pkl['y']
        '''Select appropirate concept (linear or non-linear)'''

    elif experiment == 'iris':
        # sklearn's
        iris = datasets.load_iris()
        data = pd.DataFrame(iris.data)
        if args.bias:
            data['bias'] = 1
        targets = pd.DataFrame(iris.target)

        # Separate separable and non-separable flowers
        targets.replace(0, -1, inplace=True)
        targets.replace(1, 1, inplace=True)
        targets.replace(2, 1, inplace=True)

        data = data.to_numpy()
        labels = targets.to_numpy()
    
    elif experiment == 'skin':
        # dataset location manually selected (change if needed)
        dataset = 'datasets/3000_skin.pkl' if (args.data is None) else args.data
        if args.verbose > 0:
            print(f'Loaded dataset {dataset}', flush=True)
        with open(dataset, 'rb') as jar:
            data_pkl = pickle.load(jar)
        skin = pd.DataFrame(data_pkl['X'])
        if args.bias:
            skin['bias'] = 1

        data = skin.to_numpy()
        labels = data_pkl['y']
    else:
        assert False, f"Invalid experiment selected: {experiment}"
    
    return data, labels


# In[ ]:


if __name__ == '__main__':
    # Do stuff ONLY if this is a script. Not Jupyter notebook.
    print("Corruption Experiment", flush=True)
    parser = create_parser()
    args = parser.parse_args()
    
    # Check if output file already exists  
    fname_out = f'{args.result_root}/{args.experiment}_{args.index}_results.pkl'
    if os.path.exists(fname_out):                                               
            # Results file does exist: exit                                     
            print("File %s already exists"%fname_out)                           
            exit() 
    
    # Select dataset to use.
    data, labels = obtain_data(args)
    
    # Perceptron learning hyper-parameters
    model_params = {
        'input'      : data.shape[-1],
        'eta'        : args.eta,
        'max_iter'   : args.max_iter,
        'rand_seed'  : None,
        'ignore_flag': True,
        'patience'   : args.patience,  
    }
    
    # Experiment Execution
    # Since Iris only has 100, will use the default test fraction for experiments.
    real_dataset =  args.experiment == "skin"
    history = perceptron_corruption_experiment(
        X               = data,
        y               = labels,
        test_size       = args.test_fraction,
        train_size      = None,
        n_buckets       = args.n_buckets,
        model_params    = model_params,
        n_runs          = args.n_runs,
        seed            = 42,
        verbose         = args.verbose,
    )
    
    # Save experiment
    pickle_data = {
        'history':    history,
        'n_data':     data.shape[0], # We know from data-set description. 
        'test_split': args.test_fraction,
        'n_runs':     args.n_runs,
        'n_buckets':  args.n_buckets,
        'max_iter':   args.max_iter,
        'n_attribs':  data.shape[1],
    }
    with open(fname_out, 'wb') as pkl:
        pickle.dump(pickle_data, pkl)
        
        

