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

def subplot(ax, maps, metric, models):

    colors = plt.cm.viridis([x/len(models) for x in range(len(models))])
    for i, model in enumerate(models):
        print('model: ', model)
        all_vals = [map[model] for map in maps]
        print('all vals: ', all_vals)
        all_vals = np.array(all_vals)
        print('all_vals shape: ', all_vals.shape)
        means = np.mean(all_vals, axis=0)
        stds = np.std(all_vals, axis=0)

        ax.plot(range(10), means, '-o', label=model, color=colors[i])
        ax.fill_between(range(10), means - stds, means + stds, alpha=0.2, color=colors[i])
    ax.set_xticks(range(10), range(10))
    ax.set(xlabel='time', ylabel=metric)
    ax.set_title(metric)
    ax.legend()

def graph_reconstruction_results(celebs, models_list):
    if celebs is None:
        celebs = os.listdir('/playpen-nas-ssd/awang/mystyle_original/out')
    lpips_maps = []
    psnr_maps = []
    dists_maps = []
    id_error_maps = []

    for celeb in celebs:
        if not os.path.isdir(os.path.join('/playpen-nas-ssd/awang/mystyle_original/out', celeb)):
            continue
        root_folder = os.path.join('/playpen-nas-ssd/awang/mystyle_original/out', celeb, 'reconstructions')
        with open(os.path.join(root_folder, 'metrics.json'), 'r') as f:
            maps = json.load(f)

        lpips_maps.append(maps['lpips'])
        psnr_maps.append(maps['psnr'])
        dists_maps.append(maps['dists'])
        id_error_maps.append(maps['id_error'])

    # get the models that are present in all of the lpips maps
    models = list(lpips_maps[0].keys())
    for lpips_map in lpips_maps[1:]:
        models = [model for model in models if model in lpips_map.keys()]
    if models_list is not None:
        models = models_list

    print('lpips maps: ', lpips_maps)

    fig, axs = plt.subplots(2, 2)
    fig.set_figwidth(13)
    fig.set_figheight(10)
    subplot(axs[0, 0], lpips_maps, 'LPIPS', models)
    subplot(axs[0, 1], psnr_maps, 'PSNR', models)
    subplot(axs[1, 0], dists_maps, 'DISTS', models)
    subplot(axs[1, 1], id_error_maps, 'ID Error', models)

    fig.suptitle('Reconstruction Evaluation for All Celebs')
    plt.savefig(os.path.join('/playpen-nas-ssd/awang/mystyle_original/out/reconstruction_plot.png'))

    plt.clf()

def process_args():
    parser = argparse.ArgumentParser(description='Batch Train Celeb')
    
    # Optional arguments
    parser.add_argument('--device', type=str, help='GPUs to use', required=False, default='0')
    parser.add_argument('--models', type=str)
    parser.add_argument('--celebs', type=str)
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    return args

if __name__ == '__main__':
    args = process_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = args.models.split(',') if args.models is not None else None
    celebs = args.celebs.split(',') if args.celebs is not None else None
    graph_reconstruction_results(celebs, models)
