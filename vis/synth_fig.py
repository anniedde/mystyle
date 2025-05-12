import cv2
import numpy as np
#import torch
import os

celebs = ['Margot', 'Harry', 'IU', 'Michael', 'Sundar']
out = '/playpen-nas-ssd/awang/mystyle_original/out'
save_dir = '/playpen-nas-ssd/awang/mystyle_original/vis/all_synth_examples'

for celeb in celebs:
    celeb_save_dir = f'{save_dir}/{celeb}'
    os.makedirs(celeb_save_dir, exist_ok=True)
    for test_cluster in range(10):
        for synth_img_idx in range(20):
            # create 1x4 grid of 1024 x 1024 images with input and reconstructed images
            grid = np.zeros((1024*1, 1024*8, 3), dtype=np.uint8)
            
            experiments = ['lower_bound', 'constrained_random_3', 'constrained_ransac_3', 'constrained_random_5', 'constrained_ransac_5', 'constrained_random_10', 'constrained_ransac_10', 'upper_bound']
            for i, experiment in enumerate(experiments):
                model = 'all' if experiment == 'upper_bound' else '9'
                img_path = os.path.join(out, celeb, 'synthesis', experiment, model, str(test_cluster), 'images', f'{synth_img_idx}.jpg')
                img = cv2.imread(img_path)

                # place input on top of recon_img in the i-th column of the grid
                col_start, col_end = i*1024, (i+1)*1024
                grid[:, col_start:col_end] = img
            
            save_path = f'{celeb_save_dir}/{test_cluster}_{synth_img_idx}.png'
            cv2.imwrite(save_path, grid)

