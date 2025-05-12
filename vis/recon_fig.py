import cv2
import numpy as np
#import torch
import os

celebs = ['Margot', 'Harry', 'IU', 'Michael', 'Sundar']
out = '/playpen-nas-ssd/awang/mystyle_original/out'
save_dir = '/playpen-nas-ssd/awang/mystyle_original/vis/all_recon_examples'

for celeb in celebs:
    celeb_save_dir = f'{save_dir}/{celeb}'
    os.makedirs(celeb_save_dir, exist_ok=True)
    for test_cluster in range(10):
        for test_img_idx in range(10):
            # create 2x4 grid of 1024 x 1024 images with input and reconstructed images
            grid = np.zeros((1024*1, 1024*9, 3), dtype=np.uint8)
            experiments = ['lower_bound', 'constrained_random_3', 'constrained_ransac_3', 'constrained_random_5', 'constrained_ransac_5', 'constrained_random_10', 'constrained_ransac_10', 'upper_bound']
            for i, experiment in enumerate(experiments):
                print(f'Processing celeb={celeb}, test_cluster={test_cluster}, test_img_idx={test_img_idx}, experiment={experiment}')
                if experiment == 'upper_bound':
                    img_folder = os.path.join(out, celeb, 'reconstructions', experiment, str(test_cluster))
                else:
                    img_folder = os.path.join(out, celeb, 'reconstructions', experiment, '9', str(test_cluster))
                input_img = cv2.imread(os.path.join(img_folder, 'input_images', f'{test_img_idx}.jpg'))
                recon_img = cv2.imread(os.path.join(img_folder, 'recon_images', f'{test_img_idx}.jpg'))

                # place input on top of recon_img in the i-th column of the grid
                col_start, col_end = (i+1)*1024, (i+2)*1024
                grid[:, col_start:col_end] = recon_img
            grid[:, :1024] = input_img
            
            save_path = f'{celeb_save_dir}/{test_cluster}_{test_img_idx}.png'
            cv2.imwrite(save_path, grid)

