#options:
#    -h, --help show this help message and exit
#    -c CHANNELS, --channels CHANNELS Number of channels of the model 
#    -i ITERS, --iters ITERS Maximum number of iterations
#    -s STATES, --states STATES Size of the state space.      
#    -t TRAIN_INTERVAL, --train_interval TRAIN_INTERVAL Train interval of the network  
#    -m IMAGE, --image IMAGE Path to GT image
#    -r RUN, --run RUN Number of the run. If provided results will be stored in a subfolder   
#    -f FOLDER, --folder FOLDER Folder in which the reults will be stored      
#    -g FULL_RANGE, --full_range FULL_RANGE If set to true will validate all RGB channels of the image   
#
DATADIR=/storage/praha1/home/xhrani02/GrowingCA/Thesis/
SCRIPT_PATH=./training_scripts/GSD/run_dnca.sh

export TMPDIR=$SCRATCHDIR
cd $DATADIR

python -m pip --version
python --version

echo "Printing arguments"
echo "RUN" $1 
echo "CHANNELS" $2 
echo "IMAGE" $3
echo "STATES" $4

bash $SCRIPT_PATH $1 $2 $3 $4
