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

def evaluate_metrics(folder, lpips, dists, person_identifier):
    input_folder = os.path.join(folder, 'input_images')
    output_folder = os.path.join(folder, 'recon_images')
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    lpips_loss_list, psnr_list, dists_loss_list, id_error_list = [], [], [], []
    for file_name in os.listdir(input_folder):
        assert file_name in os.listdir(output_folder), f'{file_name} not in {output_folder}'

        output_img = Image.open(os.path.join(output_folder, file_name)).convert('RGB') # [512, 512, 3] [H, W, C]
        input_img = Image.open(os.path.join(input_folder, file_name)).convert('RGB') # [512, 512, 3] [H, W, C]
        
        input_img = transform(input_img).to(device).unsqueeze(0) # [1, 512, 512, 3] [B, H, W, C]
        output_img = transform(output_img).to(device).unsqueeze(0) # [1, 512, 512, 3] [B, H, W, C]
        
        # ----------------------------------- LPIPS ----------------------------------- #
        # lpips requires normalized images [-1, 1]
        lpips_loss = lpips(input_img, output_img).squeeze().item()
        lpips_loss = round(lpips_loss, 3)

        # ----------------------------------- psnr ----------------------------------- #
        # psnr requires denormalized images
        input_img_denorm = denorm(input_img)
        output_img_denorm = denorm(output_img)
        # type casting to uint8 then back to float32 is necessary for psnr calculation
        input_img_denorm = input_img_denorm.to(torch.uint8).to(torch.float32)
        output_img_denorm = output_img_denorm.to(torch.uint8).to(torch.float32)
        mse = torch.mean((input_img_denorm - output_img_denorm) ** 2)
        psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
        psnr = psnr.squeeze().item()
        psnr = round(psnr, 3)

        # ----------------------------------- DISTS ---------------------------------- #
        # DISTS requires normalized images between [0, 1] rather than [-1, 1]
        input_dists, output_dists = (input_img + 1) / 2.0, (output_img + 1) / 2.0
        dists_loss = dists(input_dists, output_dists)
        dists_loss = dists_loss.squeeze().item()
        dists_loss = round(dists_loss, 3)

        # --------------------------------- id_error --------------------------------- #
        # # id_error requires denormalized images
        input_img_denorm = denorm(input_img)
        output_img_denorm = denorm(output_img)
        input_feature = person_identifier.get_feature(input_img_denorm.squeeze())
        output_feature = person_identifier.get_feature(output_img_denorm.squeeze())
        sim = person_identifier.compute_similarity(input_feature, output_feature)
        sim = sim.item()
        sim = round(sim, 3)
        id_error = 1 - sim

        lpips_loss_list.append(lpips_loss)
        psnr_list.append(psnr)
        dists_loss_list.append(dists_loss)
        id_error_list.append(id_error)
            
    lpips_mean = sum(lpips_loss_list) / len(lpips_loss_list)
    psnr_mean = sum(psnr_list) / len(psnr_list)
    dists_mean = sum(dists_loss_list) / len(dists_loss_list)
    id_error_mean = sum(id_error_list) / len(id_error_list)

    return lpips_mean, psnr_mean, dists_mean, id_error_mean

def subplot(ax, map, metric, times):
    colors = plt.cm.viridis([x/len(map.keys()) for x in range(len(map.keys()))])
    for i, model in enumerate(map):
        length = len(map[model])
        ax.plot(range(length), map[model], '-o', label=model, color=colors[i])
    ax.set_xticks(range(length), times)
    ax.set(xlabel='time', ylabel=metric)
    ax.set_title(metric)
    ax.legend()

def graph_synthesis_results(celeb, device, models_list=None):
    root_folder = os.path.join('/playpen-nas-ssd/awang/mystyle_original/out/', celeb, 'synthesis')
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

    id_error_map = maps['id_error']
    id_error_map = {k: v for k, v in id_error_map.items() if k in models_list}
    fid_map = maps['fid']
    fid_map = {k: v for k, v in fid_map.items() if k in models_list}
    times = [str(i) for i in range(10)]

    fig, axs = plt.subplots(1, 2)
    fig.set_figwidth(13)
    fig.set_figheight(5)
    subplot(axs[0], id_error_map, 'ID Error', times)
    subplot(axs[1], fid_map, 'FID', times)

    fig.suptitle('Synthesis Evaluation for {}'.format(celeb))
    plt.savefig(os.path.join(root_folder, 'plot.png'))

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
    graph_synthesis_results(args.celeb, device, models)
