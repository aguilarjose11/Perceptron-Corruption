#!/bin/bash
#SBATCH --partition=normal
#SBATCH --job-name=corr_exp                                                      
#SBATCH --chdir=/scratch/joseaguilar/corruption/                                       
#SBATCH --error=corruption_%J_job_%04a_stderr.txt                                   
#SBATCH --output=corruption_%J_job_%04a_stdout.txt                                  
#SBATCH --ntasks=1                                                              
#SBATCH --mem=2G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:59:58
#SBATCH --array=0-99

echo start time
date

source activate tda-gpu

# Other experiments are SPECT, spambase, and bankrupcy
EXPERIMENT="bankrupcy"

# For SLURM only. It will help in parallel execution. See --array param.
# Remember that --index is simply an output file index value.
#EXP_NUMBER=$SLURM_ARRAY_TASK_ID
#INDEXLABEL=$EXP_NUMBER

# For local execution only. Comment out if in schooner.
INDEXLABEL=debugging

# Experiment set up
BUCKETS=100                                                                      
NRUNS=1 # If running parallely, [NRUNS * --array] total runs

# Model parameters
ETA=0.1
MAXITER=2000             
PATIENCE=40 
RESROOT=results # DO NOT forget to create the directory!
TESTFRACT=0.2

python corruption_experiment.py --experiment $EXPERIMENT --bias -t $TESTFRACT -b $BUCKETS -r $NRUNS --eta $ETA --max_iter $MAXITER --w_init -1 1 -v --result_root $RESROOT --index $INDEXLABEL --patience $PATIENCE
                                                                                
echo end time                                                                   
date   
