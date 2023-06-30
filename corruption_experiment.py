#!/usr/bin/env python
# coding: utf-8

# Python Core Libraries
import os
import pickle
from argparse import ArgumentParser, RawTextHelpFormatter 

# Extra Libraries
from tqdm import tqdm

# Data Science Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE

# Research Code
import Perceptron.perceptron as pn


def perceptron_data_corruption(
        train_data,
        train_labels,
        test_data,
        test_labels,
        model_params,
        verbose,
        history,
        seed,
        return_model: bool=False,
        sgd: bool=False
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

        if return_model:
            best_score = None
            best_pred  = None
        
        # Begin with high corruption and then add more buckets
        for buckets in tqdm(range(1, n_buckets + 1), disable=verbose >= 2):

            # Choose buckets to be used.
            indices = rng.choice(range(0, len(train_data)), size=buckets, replace=False)
            #import pdb; pdb.set_trace()

            X       = np.concatenate(train_data[indices].values)
            Y       = np.concatenate(train_labels[indices].values)
            # Double check that the label array has correct shape
            Y = Y.reshape(-1, 1)

            # Train model
            if sgd:
                model = Perceptron(max_iter=model_params['max_iter'],
                                   n_iter_no_change=model_params['patience'],
                                   eta0=model_params['eta'])
                model.fit(X, Y)
                test_pred = model.predict(test_data)
                train_pred = model.predict(X)
            else:
                model = pn.PocketPerceptron(**model_params)
                model.train(X, Y)
                test_pred = model.solve(test_data)
                train_pred = model.solve(X)


            # Measure zero-one & store
            test_score = accuracy_score(test_pred, test_labels)
            train_score = accuracy_score(train_pred, Y)

            if return_model:
                # This is saved every epoch!
                best_pred = {
                    'X': test_data,
                    'y': test_labels,
                    'model': model,
                    'train_score': train_score,
                    'train_pred': test_pred,
                    'test_score': test_score,
                    'test_pred': train_pred,
                }
                history['best_model'].append(best_pred)

            history[buckets].append((test_score, train_score))
            # Used by Gallant's learning bound.
            if sgd:
                L_values.append(np.linalg.norm(model.coef_.T))
            else:
                L_values.append(np.linalg.norm(model.W))
            
        history['L'].append(L_values)

        return history
    

def perceptron_corruption_experiment(X,
                                     y,
                                     test_size,
                                     train_size,
                                     n_buckets,
                                     model_params,
                                     n_runs,
                                     seed,
                                     verbose,
                                     return_model: bool=False,
                                     sgd: bool=False,
                                     smote: bool=False
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
    if return_model:
        history['best_model'] = []
    
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
        test_labels = test_labels.reshape(-1, 1)

        # SMOTE
        if smote:
            sm = SMOTE(random_state=42)
            train_data, train_labels = sm.fit_resample(train_data, train_labels)
            train_labels = train_labels.reshape(-1, 1)

        # We just need to bucketize the training data now (Testing data used as is)
        train_data, train_labels = shuffle(train_data, train_labels)
        train_data   = np.array_split(train_data, n_buckets) # split rises exception if not even!
        train_data   = pd.Series(train_data) # Helps in keeping bucket structure
        train_labels = np.array_split(train_labels, n_buckets)
        train_labels = pd.Series(train_labels)




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
            return_model=return_model,
            sgd=sgd,
        )
        
    return history


# # Data Corruption Experiment

def create_parser():
    parser = ArgumentParser(description='CoLT Experiment', formatter_class=RawTextHelpFormatter)
    dataset_help = '''
Experiment to conduct. There are 7 designed and implemented.
    
syn-lin: Synthetic Linearly-Separable
syn-non: Synthetic Non Linearly-Separable
iris: Iris (Linearly-Separable)
skin: Skin/No Skin (Non Linearly-Separable)
SPECT: Data on cardiac Single Ptoton Emission Computer Tomography (SPECT) images.
TBankrupcy: Taiwaneese company bankrupcy dataset.
spambase: Dataset for clasifying spam emails
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
    parser.add_argument('--dataset_size', type=int, help='Number of datapoint to sample for dataset.', required=False)
    #
    parser.add_argument('-t', '--test_fraction', type=float, help='Fraction of whole dataset to use as testing.', default=0.2)
    parser.add_argument('-b', '--n_buckets', type=int, help='Number of buckets to split data into.', default=20)
    parser.add_argument('-r', '--n_runs', type=int, help='Number of times to repeat experiment.', default=10)
    #
    parser.add_argument('--eta', type=float, default=1, help='Learning rate of perceptron.' )
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum number of Perceptron iterations before convergance is assumed.')
    parser.add_argument('--w_init', nargs='+', type=float, default=[0.5, 0.5], help='Initial weight distribution [lower, upper] bounds.')
    parser.add_argument('--smote', action='store_true', default=False, help='Flag for applying SMOTE for class imbalance.')
    parser.add_argument('--sgd', action='store_true', default=False, help='Flag for using Stochasitc Gradient Descent on Perceptron.')
    # 
    parser.add_argument('-v', '--verbose',action='count', default=0, help='Verbosity of messages.' )
    parser.add_argument('-i', '--index', type=str, default='0', help='Inex of experiments. Helpful when running multiple repetitions of same experiment.')
    parser.add_argument('--result_root', type=str, default='.', help='Directory to store results.')
    parser.add_argument('--save_model', action='store_true', default=False, help='Flag for saving model.')
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
        labels = labels.to_numpy()
    
    elif experiment == 'SPECT':
        dataset = 'datasets/SPECT.pkl' if (args.data is None) else args.data
        if args.verbose > 0:
            print(f'Loaded dataset {dataset}', flush=True)
        with open(dataset, 'rb') as jar:
            data_pkl = pickle.load(jar)
        SPECT = data_pkl['X']
        if args.bias:
            SPECT['bias'] = 1
        
        data = SPECT.to_numpy()
        labels = data_pkl['y']
        
        labels.replace(0, -1, inplace=True)
        labels.replace(1, 1, inplace=True)
        labels = labels.to_numpy()

    elif experiment == 'bankrupcy':
        dataset = 'datasets/TBankrupcy.pkl' if (args.data is None) else args.data
        if args.verbose > 0:
            print(f'Loaded dataset {dataset}', flush=True)
        with open(dataset, 'rb') as jar:
            data_pkl = pickle.load(jar)
        TBankrupcy = data_pkl['X']
        if args.bias:
            TBankrupcy['bias'] = 1
        data = TBankrupcy.to_numpy()
        labels = data_pkl['y']
        
        labels.replace(0, -1, inplace=True)
        labels.replace(1, 1, inplace=True)
        labels = labels.to_numpy()
    
    elif experiment == 'spambase':
        dataset = 'datasets/spambase.pkl' if (args.data is None) else args.data
        if args.verbose > 0:
            print(f'Loaded dataset {dataset}', flush=True)
        with open(dataset, 'rb') as jar:
            data_pkl = pickle.load(jar)
        spambase = data_pkl['X']
        if args.bias:
            spambase['bias'] = 1
        data = spambase.to_numpy()
        labels = data_pkl['y']
        
        labels.replace(0, -1, inplace=True)
        labels.replace(1, 1, inplace=True)
        labels = labels.to_numpy()

    else:
        assert False, f"Invalid experiment selected: {experiment}"

    return data, labels




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
        return_model    = args.save_model,
        sgd             = args.sgd,
        smote           = args.smote,
    )
    
    # Save experiment. If saving model, see history['best_model']
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
        
        

