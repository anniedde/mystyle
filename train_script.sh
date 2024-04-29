#!/bin/bash

# Change directory to the location of your Python script
rm -rf /playpen-nas-ssd/awang/mystyle_original/out/domain_expansion_margot_0_real
python train.py \
    --images_dir /playpen-nas-ssd/awang/data/mystyle/Margot/1/train/preprocessed \
    --output_dir out/domain_expansion_margot_0_real \
    --generator_path third_party/ffhq.pkl \
    --anchor_dir /playpen-nas-ssd/awang/data/mystyle/Margot/1/train/anchors \
    --replay_dir /playpen-nas-ssd/awang/data/replay/mystyle_replay_buffer_10/Margot/1/replay/preprocessed \
    --device 2
