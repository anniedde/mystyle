#!/bin/bash

# Access command line arguments
celeb=$1

# set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$2


# Define the directory path
dir_path="/playpen-nas-ssd/awang/data/mystyle/$celeb/3/train/preprocessed"

# Check if the directory does not exist

echo "Running data preprocessing for celeb $celeb, video 3"

python data_preprocess/0_align_face.py \
    --images_dir /playpen-nas-ssd/awang/data/mystyle/$celeb/3/train/raw \
    --save_dir $dir_path \
    --trash_dir /playpen-nas-ssd/awang/data/mystyle/$celeb/3/train/trash \
    --landmarks_model helper_models/dlib_landmarks_model.dat \
    --min_size 0 \
    --min_id_size 0
