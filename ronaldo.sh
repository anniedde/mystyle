#!/bin/bash

python eval/batch_synthesize_celeb.py --celeb Michael --experiment constrained_random_5;
python eval/batch_synthesize_celeb.py --celeb Michael --experiment constrained_ransac_5;
python eval/batch_synthesize_celeb.py --celeb IU --experiment constrained_ransac_5;

wait

CUDA_VISIBLE_DEVICES=0 python eval/batch_eval_celeb.py --celeb Michael --experiment constrained_random_5  &
CUDA_VISIBLE_DEVICES=1 python eval/batch_eval_celeb.py --celeb Michael --experiment constrained_ransac_5  &
(
    CUDA_VISIBLE_DEVICES=2 python eval/batch_eval_celeb.py --celeb Harry --experiment constrained_random_10 
    CUDA_VISIBLE_DEVICES=2 python eval/batch_eval_celeb.py --celeb Harry --experiment constrained_ransac_buffer_10_mean 
) &
(
    CUDA_VISIBLE_DEVICES=3 python eval/batch_eval_celeb.py --celeb Sundar --experiment constrained_ransac_10
    CUDA_VISIBLE_DEVICES=3 python eval/batch_eval_celeb.py --celeb Sundar --experiment constrained_random_buffer_10_mean 
)&

wait

cd /playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion
conda deactivate
conda activate test_env_from_yaml
python batch_synthesize_celeb.py --celeb IU --experiments random,lower_bound --which_gpus 0;