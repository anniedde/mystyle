#!/bin/bash

# get argument for celeb name
celeb=$1
experiment=$2
device=$3

#CUDA_VISIBLE_DEVICES=$device python /playpen-nas-ssd/awang/data/replay/ransac_buffer_10/initialize_folder.py --celeb $celeb 
#CUDA_VISIBLE_DEVICES=$device python /playpen-nas-ssd/awang/data/replay/ransac_buffer_10/move_files.py --celeb $celeb
CUDA_VISIBLE_DEVICES=$device python batch_train_celeb.py --celeb $celeb --experiment $experiment --device $device -r --end 20

CUDA_VISIBLE_DEVICES=$device python eval/batch_eval_celeb.py --celeb $celeb --experiment $experiment --device $device --models 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
CUDA_VISIBLE_DEVICES=$device python eval/eval_reconstruction.py --celeb $celeb --experiment $experiment --device $device
CUDA_VISIBLE_DEVICES=$device python eval/eval_synthesis.py --celeb $celeb --experiment $experiment --device $device