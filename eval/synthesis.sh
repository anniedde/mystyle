#!/bin/bash

# Set the base directory
base_dir="/playpen-nas-ssd/awang/mystyle_original/"

# Get the list of celebs
celeb_list=('Harry' 'Sundar' 'IU' 'Margot' 'Michael')
experiments=('lower_bound' 'constrained_random_3' 'constrained_random_5' 'constrained_random_10' 'constrained_ransac_3' 'constrained_ransac_5' 'constrained_ransac_10' 'upper_bound')

# Set the initial device number
device=0

# Loop through each celeb and run the script
for celeb in "${celeb_list[@]}"; do
    (
        for experiment in "${experiments[@]}"; do
            MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$device python batch_synthesize_celeb.py --celeb $celeb --experiment $experiment --device $device
        done
    ) &
    ((device++))
done

# Wait for all the scripts to finish
wait
