import os, sys
import argparse
import shutil
import importlib
from utils.misc import notify
import torch
from utils.data_utils import PersonalizedDataset
from utils.io_utils import save_images, save_latents
from tqdm import tqdm
from lpips import LPIPS
from pathlib import Path

lpips = LPIPS(net='alex').eval()

def process_args():
    parser = argparse.ArgumentParser(description='Batch Train Celeb')
    
    # Required arguments
    parser.add_argument('--celeb', type=str, help='Name of the celebrity', required=True)
    parser.add_argument('--experiment', type=str, help='Experiment name', required=True)

    # Optional arguments
    parser.add_argument('--device', type=str, help='GPUs to use', required=False, default='0')
    
    args = parser.parse_args()

    return args

def get_net(celeb, experiment):
    training_runs_dir = '/playpen-nas-ssd/awang/mystyle_original/training-runs'
    if experiment == 'upper_bound':
        path = os.path.join(training_runs_dir, celeb, experiment, 'all', 'mystyle_model.pt')
    else:
        path = os.path.join(training_runs_dir, celeb, experiment, '9', 'mystyle_model.pt')

    return torch.load(path)

def reconstruction_loss(synth, target):
    dist = lpips(synth, target)
    l2_dist = (synth - target).square().mean()

    loss = dist + l2_dist

    return loss

def project(net, img, w_init):
    w_opt = torch.tensor(w_init, requires_grad=True, device='cuda')
    optimizer = torch.optim.Adam(w_opt, betas=(0.9, 0.999), lr=0.005)

    for step in tqdm(range(500)):
        synth = net(w_opt, noise_mode='const', force_fp32=True)
        loss = reconstruction_loss(synth, img)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    return w_opt

def project_all_images(net, dataset, save_dir):
    for sample in dataset:
        img = sample['img']
        w_init = sample['w_code']

        w_final = project(net, img, w_init)
        final_image = net(w_final, noise_mode='const', force_fp32=True)

        # save final_image
        save_path = save_dir.joinpath('recon_images', sample['name'])
        save_images(final_image, save_path)

        # save input image
        input_save_path = save_dir.joinpath('input_images', sample['name'])
        save_images(img, input_save_path)

        # save latent
        latent_save_path = save_dir.joinpath('latents', sample['name']).with_suffix('.pt')
        save_latents(w_final, latent_save_path)

def graph_result(celeb):


if __name__ == '__main__':
    args = process_args()
    net = get_net(args.celeb, args.experiment)
    net.eval()

    celeb_dir = os.path.join('/playpen-nas-ssd/awang/data/mystyle', args.celeb)
    out_dir = os.path.join('/playpen-nas-ssd/awang/mystyle_original/out', args.celeb, 'reconstructions', args.experiment)
    for video in range(0, 10):
        # reconstruct all test images for the video
        test_image_dir = os.path.join(celeb_dir, str(video), 'test', 'preprocessed')
        test_latent_dir = os.path.join(celeb_dir, str(video), 'test', 'anchors')
        save_dir = Path(os.path.join(out_dir, str(video)))

        dataset = PersonalizedDataset(test_image_dir, test_latent_dir)
        project_all_images(net, dataset, save_dir)

    graph_result(args.celeb, args.experiment)

    
