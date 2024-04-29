#!/bin/bash

# Access command line arguments
celeb=$1

# Define the start and end of the sequence
start=0
end=9

# Use a for loop to iterate over the sequence
for i in $(seq $start $end) all; do
    # Define the directory path
    images_path="/playpen-nas-ssd/awang/data/mystyle/$celeb/$i/train/preprocessed"
    anchors_path="/playpen-nas-ssd/awang/data/mystyle/$celeb/$i/train/anchors"

    # Check if the directory does not exist
    if [ ! -d "$anchors_path" ]; then
        echo "Inferring anchors for celeb $celeb, video $i"

        python infer_anchors.py \
            --verbose False \
            --images_dir $images_path \
            --output_dir $anchors_path \
            --encoder_checkpoint /playpen-nas-ssd/awang/mystyle_original/third_party/faces_w_encoder.pt
    else
        echo "Directory $anchors_path already exists. Skipping."
    fi

    # do the same thing for the test set
    images_path="/playpen-nas-ssd/awang/data/mystyle/$celeb/$i/test/preprocessed"
    anchors_path="/playpen-nas-ssd/awang/data/mystyle/$celeb/$i/test/anchors"

    # Check if the directory does not exist
    if [ ! -d "$anchors_path" ]; then
        echo "Inferring anchors for celeb $celeb, video $i"

        python infer_anchors.py \
            --verbose False \
            --images_dir $images_path \
            --output_dir $anchors_path \
            --encoder_checkpoint /playpen-nas-ssd/awang/mystyle_original/third_party/faces_w_encoder.pt
    else
        echo "Directory $anchors_path already exists. Skipping."
    fi
done

#CUDA_VISIBLE_DEVICES=3 python infer_anchors.py \
#    --images_dir /playpen-nas-ssd/awang/data/domain_expansion_margot_2/images \
#    --output_dir /playpen-nas-ssd/awang/data/domain_expansion_margot_2/anchors \
#    --encoder_checkpoint /playpen-nas-ssd/awang/mystyle_original/third_party/faces_w_encoder.pt