#!/bin/bash
#SBATCH --partition=normal
#SBATCH --job-name=spam_corr                                                      
#SBATCH --chdir=/scratch/joseaguilar/corruption/                                       
#SBATCH --error=corruption_%J_job_%04a_stderr.txt                                   
#SBATCH --output=corruption_%J_job_%04a_stdout.txt                                  
#SBATCH --ntasks=1                                                              
#SBATCH --mem=2G
#SBATCH --cpus-per-task=4
#SBATCH --time=09:58:58
#SBATCH --array=0-99

echo start time
date

# Schooner's environment is: tda-gpu
source activate tda-gpu

export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4

# Other experiments are SPECT , spambase , and bankrupcy aside of syn-lin, syn-non, skin, and iris
EXPERIMENT="spambase" # Do not forget to change testfract

# For SLURM only. It will help in parallel execution. See --array param.
# Remember that --index is simply an output file index value.
EXP_NUMBER=$SLURM_ARRAY_TASK_ID

# If you need to identify an experiment uniquely, add to the variable bellow!
INDEXLABEL=noSMOTE_$EXP_NUMBER

# For local execution only. Comment out if in schooner.
#INDEXLABEL=debugging

# Experiment set up
BUCKETS=100
NRUNS=1 # If running parallely, [NRUNS * --array] total runs

# Model parameters
ETA=1
MAXITER=2500
PATIENCE=2500
RESROOT=results # DO NOT forget to create the directory!

# bnakruptcy: 0.7 | spambase: 0.4 | SPECT: 0.27
TESTFRACT=0.4


python corruption_experiment.py --save_model --experiment $EXPERIMENT --bias -t $TESTFRACT -b $BUCKETS -r $NRUNS --eta $ETA --max_iter $MAXITER --w_init -1 1 -v --result_root $RESROOT --index $INDEXLABEL --patience $PATIENCE

echo end time
date
