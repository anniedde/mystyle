import os, sys
sys.path.append('/playpen-nas-ssd/awang/mystyle_original')
sys.path.append('/playpen-nas-ssd/awang/mystyle_original/third_party/stylegan2_ada_pytorch')
import argparse
from utils.misc import notify
import torch
from utils.data_utils import PersonalizedDataset
from utils.io_utils import save_images, save_latents
from tqdm import tqdm
from lpips import LPIPS
from DISTS_pytorch import DISTS
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms
import pickle
import json

sys.path.append('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/utils_copy')
from id_utils import PersonIdentifier

def denorm(img):
    # img: [b, c, h, w]
    img_to_return = (img + 1) * 127.5
    img_to_return = img_to_return.permute(0, 2, 3, 1).clamp(0, 255)
    return img_to_return # [b, h, w, c]

def subplot(ax, map, metric, times):
    models = list(map.keys())
    colors = plt.cm.viridis([x/len(models) for x in range(len(models))])
    vals = []
    for i, model in enumerate(models):
        avg_val = np.mean(map[model], axis=0)
        vals.append(avg_val)
    ax.bar(models, vals, color=colors)
    ax.set_title(metric)
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=-45, ha='left')

def graph_reconstruction_results(celeb, device, models_list=None):
    lpips = LPIPS(net='alex').eval().to(device)
    dists = DISTS().to(device)
    person_identifier = PersonIdentifier('/playpen-nas-ssd/luchao/projects/mystyle/pretrained/model_ir_se50.pth', None, None)

    root_folder = os.path.join('/playpen-nas-ssd/awang/mystyle_original/out/', celeb, 'reconstructions')
    if not models_list:
        models_list = [name for name in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, name))]
        sorted_models_list = [None] * len(models_list)
        sorted_models_list[0] = 'lower_bound'
        sorted_models_list[-1] = 'upper_bound'
        # remove lower_bound and upper_bound from models_list
        sorted_models_list[1:-1] = sorted([model for model in models_list if model not in ['lower_bound', 'upper_bound']])
        models_list = sorted_models_list

    with open(os.path.join(root_folder, 'metrics.json'), 'r') as f:
            maps = json.load(f)

    lpips_map = maps['lpips']
    # only keep the entries that are in models_list
    lpips_map = {k: v for k, v in lpips_map.items() if k in models_list}
    psnr_map = maps['psnr']
    psnr_map = {k: v for k, v in psnr_map.items() if k in models_list}
    dists_map = maps['dists']
    dists_map = {k: v for k, v in dists_map.items() if k in models_list}
    id_error_map = maps['id_error']
    id_error_map = {k: v for k, v in id_error_map.items() if k in models_list}
    times = sorted(os.listdir(os.path.join(root_folder, models_list[0])))
    times = sorted(times, key=lambda x: (x[0].isdigit(), x))

    fig, axs = plt.subplots(2, 2)
    fig.set_figwidth(13)
    fig.set_figheight(10)
    fig.set_tight_layout(True)
    subplot(axs[0, 0], lpips_map, 'LPIPS', times)
    subplot(axs[0, 1], psnr_map, 'PSNR', times)
    subplot(axs[1, 0], dists_map, 'DISTS', times)
    subplot(axs[1, 1], id_error_map, 'ID Error', times)

    fig.suptitle('Reconstruction Evaluation for {}'.format(celeb))
    plt.savefig(os.path.join(root_folder, 'plot_avgs.png'))

    plt.clf()

def process_args():
    parser = argparse.ArgumentParser(description='Batch Train Celeb')
    
    # Required arguments
    parser.add_argument('--celeb', type=str, help='Name of the celebrity', required=True)

    # Optional arguments
    parser.add_argument('--device', type=str, help='GPUs to use', required=False, default='0')
    parser.add_argument('--models', type=str, help='List of models to evaluate', required=False, default=None)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    return args

if __name__ == '__main__':
    args = process_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = args.models.split(',') if args.models else None
    print('models:', models)
    graph_reconstruction_results(args.celeb, device, models)
