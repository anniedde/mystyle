#!/bin/bash

# Set the base directory
base_dir="/playpen-nas-ssd/awang/mystyle_original/"

# Get the list of celebs
celeb_list=$(ls "$base_dir/training-runs")

# Set the initial device number
device=1

# Loop through each celeb and run the script
for celeb in $celeb_list; do
    # get list of experiments in the training folder
    exp_list=$(ls "$base_dir/training-runs/$celeb")

    # Initialize the total command
    total_command=""

    # Loop through each experiment and add its command to the total command
    for exp in $exp_list; do

        # if experiment exists in  out folder, skip
        if [ -d "$base_dir/out/$celeb/reconstructions/$exp" ]; then
            continue
        fi
        # Construct the command for the current experiment
        command="python batch_reconstruct_celeb.py --celeb $celeb --experiment $exp --device $device;"

        # Append the command to the total command
        total_command+=" $command"
    done

    # Run all experiments sequentially using the total command
    eval $total_command &
    echo $total_command
    
    # Increment the device number
    ((device++))
done

# Wait for all the scripts to finish
wait