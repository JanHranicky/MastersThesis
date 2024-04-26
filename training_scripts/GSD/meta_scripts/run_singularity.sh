DATADIR=/storage/praha1/home/xhrani02/GrowingCA/Thesis/training_scripts/GSD/meta_scripts

singularity exec --nv /cvmfs/singularity.metacentrum.cz/NGC/TensorFlow\:23.08-tf2-py3.SIF /bin/sh $DATADIR/$SCRIPT $RUN $CHANNELS $IMAGE $STATES