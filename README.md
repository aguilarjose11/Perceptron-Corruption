LOD 2023: Perceptrons Under Verifiable Random Data Corruption
===========================================================

Welcome!

This repository contains the code used for the upcoming paper Aguilar, J and Diochnos, D, Perceptrons Under Verifiable Random Data Corruption, In [LOD 2023](https://lod2023.icas.cc/). This research project was carried out by [Jose E. Aguilar Escamilla](https://www.linkedin.com/in/jose-aguilar-escamilla/) and was supervised by [Dr. Dimitrios I. Diochnos](http://diochnos.com/) at the [Gallgoly College of Engineering, School of Computer Science](https://www.ou.edu/coe/cs) @ [The University of Oklahoma (OU)](https://www.ou.edu/), and was part of the Ronald E. McNair Post-Baccalaureate Achievement Program.

Abstract
--------

We study perceptrons when datasets are randomly corrupted by noise and  subsequently such corrupted examples are discarded from the training process.  Overall, perceptrons appear to be remarkably stable; their accuracy drops slightly when large portions of the original datasets have been excluded from training  as a response to verifiable random data corruption. Furthermore, we identify a real-world dataset where it appears to be the case that perceptrons require longer time for training, both in the general case, as well as, in the framework that we consider. Finally, we explore empirically a bound on the learning rate of Gallant's ''pocket'' algorithm for learning perceptrons and observe that the bound is tighter for non-linearly separable datasets.

Requirements & Installation
---------------------------
- numpy
- pandas
- scikit-learn
- imbalanced-learn
- jupyter
- seaborn
- matplotlib
- tqdm

Code Organization
-----------------
The code is made up of 2 main groups:
- Perceptron corruption experiment code
- Jupyter notebooks for displaying results and cleaning datasets.

### Perceptron Experiment Code
The main files containing experiment code are:
- `corruption_experiment.py`
  - Contains code used for creating perceptrons, pulling in data, and performing an experiment.
- `experiment_execution.sh`
  - Experiment configuration file where the experiment is defined. This is what is launched to run an experiment.
- `Perceptron` Package
  - Contains the code defining the Perceptron pocket algorithm.

### Jupyter Notebooks
These notebooks either display data generated from the experiments, or clean data/generate synthetic datasets.
- `DataGeneration.ipynb`
  - Code used for generating linearly (and non) separable data.
- `Dataset Analysis.ipynb`
  - Shows basic information of SPECT, Bankruptcy, and Spambase datasets, storing them as pkl files.
- `DataVisualization.ipynb`
  - Used to observe changes in datasets by SMOTE.
- `Result_Visualizer.ipynb`
  - Visualizes Real-World dataset results from experiments.
- `ResultAnalysis-MultiDimensional.ipynb`
  - Visualizes synthetic dataset results from experiments.

Datasets
--------
To be added soon!


<sup><i>[Jose E. Aguilar Escamilla](https://www.linkedin.com/in/jose-aguilar-escamilla/) -- [The 9th International Conference on Machine Learning, Optimization, and Data Science.](https://lod2023.icas.cc/)</sup></i>

.