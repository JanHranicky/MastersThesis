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

# Set default values
DEFAULT_RUN="1"
DEFAULT_CHANNELS="1"
DEFAULT_IMAGE="./img/vut_logo_6x6_2px_padding.png"
DEFAULT_STATES="8"
DEFAULT_FOLDER="./checkpoints/DE/"
DEFAULT_OPERATOR="interm"
DEFAULT_ITERATION=1500

RUN="$1"
CHANNELS="$2"
IMAGE="$3"
STATES="$4"
FOLDER="$5"
OPERATOR="$6"
ITERATION="$7"

RUN="${RUN:-$DEFAULT_RUN}"
CHANNELS="${CHANNELS:-$DEFAULT_CHANNELS}"
IMAGE="${IMAGE:-$DEFAULT_IMAGE}"
STATES="${STATES:-$DEFAULT_STATES}"
FOLDER="${FOLDER:-$DEFAULT_FOLDER}"
OPERATOR="${OPERATOR:-$DEFAULT_OPERATOR}"
ITERATION="${ITERATION:-$DEFAULT_ITERATION}"

echo "Printing arguments"
echo "RUN" $RUN 
echo "CHANNELS" $CHANNELS 
echo "IMAGE" $IMAGE
echo "STATES" $STATES
echo "FOLDER" $FOLDER
echo "OPERATOR" $OPERATOR
echo "ITERATION" $ITERATION

python -m experiments.shade -c $CHANNELS -r $RUN -a $IMAGE -i $ITERATION -s $STATES -f $FOLDER -x $OPERATOR