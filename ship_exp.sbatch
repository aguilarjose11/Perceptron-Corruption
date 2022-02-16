#SBATCH --job-name=CoLTExp
#SBATCH --partition=normal
#SBATCH --ntasks=1
# memory in MB
#SBATCH --mem=2048
#SBATCH --output=colt_exp_%J_stdout.txt
#SBATCH --error=colt_exp_%J_stderr.txt
#SBATCH --time=47:00:00
#SBATCH --mail-user=joseaguilar@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/scratch/cs504301

. /home/fagg/tf_setup.sh
conda activate tf

echo start time
date

/usr/bin/time python experiment.py

echo end time
date


