#!/bin/bash
#SBATCH --partition=all
#SBATCH --job-name=corr_exp                                                      
#SBATCH --chdir=/scratch/joseaguilar/corruption/                                       
#SBATCH --error=corruption_%J_job_%04a_stderr.txt                                   
#SBATCH --output=corruption_%J_job_%04a_stdout.txt                                  
#SBATCH --ntasks=1                                                              
#SBATCH --mem=2G
#SBATCH --cpus-per-task=3
#SBATCH --time=23:59:58
#SBATCH --array=0-99

echo start time
date

source activate scikit-nwc

experimentList=("syn-lin" "syn-non" "iris" "skin")

# Variables                                                                     
EXPERIMENT="skin"
#EXPERIMENT=${experimentList[$SLURM_ARRAY_TASK_ID]} # syn-lin syn-non iris skin #DATASIZE=2500 # Was 1200

DATASIZE=3000

BUCKETS=100                                                                      
NRUNS=1 # Was 100

ETA=1 
MAXITER=2000             
PATIENCE=40 
RESROOT=results

python FinalExperiments_2.py --experiment $EXPERIMENT -l -10 -10 -10 -10 -u 10 10 10 10 --bias --dataset_size $DATASIZE -t 0.008 -b $BUCKETS -r $NRUNS --eta $ETA --max_iter $MAXITER --w_init -1 1 -vvv --result_root $RESROOT --index -$SLURM_ARRAY_TASK_ID --patience $PATIENCE --train_fraction $DATASIZE
                                                                                
echo end time                                                                   
date   
