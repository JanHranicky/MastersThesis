#!/bin/bash

#qsub -v c=5 run_training.sh
DATADIR=/storage/praha1/home/xhrani02/GrowingCA/Thesis/
cd $DATADIR

python -m pip install matplotlib
#python --version > out.txt         
#singularity run --nv /cvmfs/singularity.metacentrum.cz/NGC/TensorFlow\:23.08-tf2-py3.SIF
python -m experiments.noisy_logo