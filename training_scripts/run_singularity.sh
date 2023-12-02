#!/bin/bash

#qsub -l select=1:ncpus=1:mem=10gb:ngpus=1 -q gpu run_singularity.sh
#qsub -l select=1:ncpus=1:mem=10gb:ngpus=1 -q gpu -v c=5 run_singularity.sh
#qsub -v c=5 run_training.sh
#qsub -l select=1:ncpus=1:mem=10gb:ngpus=1 -q gpu -l walltime=9:00:00 run_singularity_12.sh
#qsub -l select=1:ncpus=1:mem=10gb:ngpus=1 -q gpu -l walltime=9:00:00 -v SCRIPT=PATH run_singularity.sh 
DATADIR=/storage/praha1/home/xhrani02/GrowingCA/Thesis/training_scripts
       
singularity run --nv /cvmfs/singularity.metacentrum.cz/NGC/TensorFlow\:23.08-tf2-py3.SIF "$DATADIR/$SCRIPT"