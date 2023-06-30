#!/bin/bash

# Print starting time
echo start time
date

# If there is a virtual environment
source venv/bin/activate

# Variables
EXPERIMENT=skin # syn-lin syn-non iris skin
DATASIZE=1200
BUCKETS=10
NRUNS=10
ETA=1
MAXITER=1000
RESROOT=results

python FinalExperiments.py --experiment $EXPERIMENT -l -10 -10 -10 -10 -u 10 10 10 10 --bias --dataset_size $DATASIZE -t 0.2 -b $BUCKETS -r $NRUNS --eta $ETA --max_iter $MAXITER --w_init -1 1 -vvv --result_root $RESROOT --index 0

echo end time
date
