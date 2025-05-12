#!/bin/bash

# Get the list of celebs
celeb_list=$(ls "/playpen-nas-ssd/awang/mystyle_original/out")

# Set the initial device number
devices=(1 2 3 4 6 7)

#initialize device index
device_index=0

# Loop through each celeb and run the script
for celeb in $celeb_list; do
    reconstructions_folder="/playpen-nas-ssd/awang/mystyle_original/out/$celeb/reconstructions"
    # if there is a file named plot.png in the reconstructions folder, skip
    if [ -f "$reconstructions_folder/metrics.json" ]; then
        continue
    fi

    # if the device index is out of bounds, exit for loop
    if [ $device_index -ge ${#devices[@]} ]; then
        break
    fi

    device=${devices[$device_index]}
    cmd="python graph.py --celeb $celeb --device $device"
    echo $cmd
    eval $cmd &

    # Increment the device number
    ((device_index++))
done

# Wait for all the scripts to finish
wait