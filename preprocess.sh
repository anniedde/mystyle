#!/bin/bash

# Access command line arguments
celeb=$1

# set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$2

# Define the start and end of the sequence
start=0
end=9

# Use a for loop to iterate over the sequence
for i in $(seq $start $end) all; do
    # Define the directory path
    dir_path="/playpen-nas-ssd/awang/data/mystyle/$celeb/$i/train/preprocessed"

    # Check if the directory does not exist
    if [ ! -d "$dir_path" ]; then
        echo "Running data preprocessing for celeb $celeb, video $i"

        python data_preprocess/0_align_face.py \
            --images_dir /playpen-nas-ssd/awang/data/mystyle/$celeb/$i/train/raw \
            --save_dir $dir_path \
            --trash_dir /playpen-nas-ssd/awang/data/mystyle/$celeb/$i/train/trash \
            --landmarks_model helper_models/dlib_landmarks_model.dat \
            --min_size 0 \
            --min_id_size 0
    else
        echo "Directory $dir_path already exists. Skipping."
    fi

    # do the same thing for the test set
    dir_path="/playpen-nas-ssd/awang/data/mystyle/$celeb/$i/test/preprocessed"
    if [ ! -d "$dir_path" ]; then
        echo "Running data preprocessing for celeb $celeb, video $i, test set"

        python data_preprocess/0_align_face.py \
            --images_dir /playpen-nas-ssd/awang/data/mystyle/$celeb/$i/test/raw \
            --save_dir $dir_path \
            --trash_dir /playpen-nas-ssd/awang/data/mystyle/$celeb/$i/test/trash \
            --landmarks_model helper_models/dlib_landmarks_model.dat \
            --min_size 0 \
            --min_id_size 0
    else
        echo "Directory $dir_path already exists. Skipping."
    fi
done

python data_preprocess/check_all_clear.py \
    --celeb $celeb