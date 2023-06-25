#!/bin/bash
#SBATCH --partition=swat_plus                                                   
#SBATCH --job-name=corr_exp                                                      
#SBATCH --chdir=/scratch/joseaguilar/corruption/                                       
#SBATCH --exclusive=user                                                        
#SBATCH --error=corruption_%J_model_%04a_stderr.txt                                   
#SBATCH --output=corruption_%J_model_%04a_stdout.txt                                  
#SBATCH --ntasks=1                                                              
#SBATCH --mem=1G                                                               
#SBATCH --cpus-per-task=12                                                      
#SBATCH --exclusive=user                                                        
#SBATCH --time=12:00:00                                                         
#SBATCH --array=0-4

echo start time
date

source activate scikit-nwc

experimentList=("sys-lin" "sys-non" "iris" "skin")

# Variables                                                                     
EXPERIMENT=${experimentList[$SLURM_ARRAY_TASK_ID]} # syn-lin syn-non iris skin                                     
DATASIZE=1200                                                                   
BUCKETS=100                                                                      
NRUNS=100                                                                        
ETA=1                                                                           
MAXITER=2000                                                                    
RESROOT=results

python FinalExperiments.py --experiment $EXPERIMENT -l -10 -10 -10 -10 -u 10 10 10 10 --bias --dataset_size $DATASIZE -t 0.2 -b $BUCKETS -r $NRUNS --eta $ETA --max_iter $MAXITER --w_init -1 1 -vvv --result_root $RESROOT --index 0
                                                                                
echo end time                                                                   
date   
