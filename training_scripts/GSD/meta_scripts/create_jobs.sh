#!/bin/bash
img=("./img/duck.png" "./img/vut_logo.png" "./img/SEP_title.png" "./img/xhrani02.png" "./img/flag_of_france.png")

for i in "${img[@]}"; do #image loop
    echo "Current image: $i" 
    for c in {32,16,8,4,3,2,1}; do #channel loop
            for k in {1..40}; do #run loop
                echo "Run: $k"
                echo "Channels: $c"
                qsub -q gpu -l select=1:ncpus=1:ngpus=1:mem=10gb:scratch_local=10gb -l walltime=23:00:00 -v RUN=$k,CHANNELS=$c,IMAGE=$i,STATES=8,SCRIPT=run_dnca_meta.sh run_singularity.sh
            done
    done
done