
import os
import sys
from pathlib import Path
from argparse import ArgumentParser

from utils import latent_space_ops, io_utils

from torchvision.utils import save_image
import torch
import numpy as np
latent_path = '/playpen-nas-ssd/awang/mystyle_original/out/Michael_extended/reconstructions/constrained_ransac_6/5/4/latents/0.pt'
model_path='/playpen-nas-ssd/awang/mystyle_original/training-runs/Michael_extended/constrained_ransac_6/5/mystyle_model.pt'
# required for pickle to magically find torch_utils for loading official FFHQ checkpoint
sys.path.append('third_party/stylegan2_ada_pytorch')

latent = torch.load(latent_path).to('cuda')
generator = io_utils.load_net(model_path).to('cuda')

imgs = generator(latent, noise_mode='const', force_fp32=True)

save_image(imgs, 'test.jpg', nrow=1, normalize=True)#, range=(-1, 1))
