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

source activate scikit-nwc

experimentList=("syn-lin" "syn-non" "iris" "skin")

# Variables
DTYPE='non'
                                                    
EXPERIMENT="syn-${DTYPE}"
#EXPERIMENT=${experimentList[$SLURM_ARRAY_TASK_ID]} # syn-lin syn-non iris skin #DATASIZE=2500 # Was 1200

# Dimensions: 4, 10, 25, 50, 100
DIMENSION=100


DATASIZE=3000

BUCKETS=100                                                                      
NRUNS=1 # Was 100

ETA=1 
MAXITER=7500             
PATIENCE=200                                                       
RESROOT=results
# Dimensions: 4, 10, 25, 50, 100
DATA="datasets/synthetic_${DTYPE}_n_3000_dims_${DIMENSION}_bounds_-10_10.pkl"

# Dimensional Experiment data already comes with bias!
python FinalExperiments.py --experiment $EXPERIMENT -l -10 -10 -10 -10 -u 10 10 10 10 --dataset_size $DATASIZE -t 0.2 -b $BUCKETS -r $NRUNS --eta $ETA --max_iter $MAXITER --w_init -1 1 -vvv --result_root $RESROOT --index "${DIMENSION}d-${SLURM_ARRAY_TASK_ID}" --patience $PATIENCE --data $DATA
                                                                                
echo end time                                                                   
date   
