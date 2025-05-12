import os, sys
sys.path.append('/playpen-nas-ssd/awang/mystyle_original')
from generate import main as run_generate
sys.path.append('/playpen-nas-ssd/awang/mystyle_original/third_party/stylegan2_ada_pytorch')
import argparse
from utils.misc import notify
import torch
from utils.data_utils import PersonalizedDataset
from utils.io_utils import save_images, save_latents
from utils.training_commands import get_training_run_folder
from tqdm import tqdm
from lpips import LPIPS
from pathlib import Path
import json
from PIL import Image
import os
from cleanfid import fid
import torchvision.transforms as transforms
import multiprocessing

sys.path.append('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion')
from utils_copy import id_utils


id_model = '/playpen-nas-ssd/luchao/projects/mystyle/pretrained/model_ir_se50.pth'
person_identifier = id_utils.PersonIdentifier(id_model, None, None)

def denorm(img):
    # img: [b, c, h, w]
    img_to_return = (img + 1) * 127.5
    img_to_return = img_to_return.permute(0, 2, 3, 1).clamp(0, 255)
    return img_to_return # [b, h, w, c]

def open_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = Image.open(image_path).convert('RGB')
    img = transform(img).to(device).unsqueeze(0)
    img = denorm(img).squeeze()
    return img

def evaluate_metrics(results_folder, reference_folder, reference_name, person_identifier):
    ### ID error
    mean_id_sim = 0

    reference_features = [person_identifier.get_feature(open_image(os.path.join(reference_folder, im))) for im in os.listdir(reference_folder) if im.endswith('.png')]

    for im in tqdm(os.listdir(os.path.join(results_folder))):
        im_path = os.path.join(results_folder, im)

        # get lowest distance from reference set
        im_feature = person_identifier.get_feature(open_image(im_path))
        sims = [person_identifier.compute_similarity(im_feature, reference_feature).item() for reference_feature in reference_features]
        max_sim = max(sims)
        mean_id_sim += max_sim

    mean_id_sim /= len(os.listdir(os.path.join(results_folder)))
    
    ### FID
    if not fid.test_stats_exists(reference_name, mode='clean'):
        print('test stat does not exist')
        fid.make_custom_stats(reference_name, reference_folder, mode="clean")
        
    fid_score = fid.compute_fid(results_folder, dataset_name=reference_name, mode="clean", dataset_split="custom")

    return fid_score, mean_id_sim

def get_net_path(celeb, experiment, model):
    training_runs_dir = '/playpen-nas-ssd/awang/mystyle_original/training-runs'
    if experiment == 'upper_bound':
        path = os.path.join(training_runs_dir, celeb, experiment, 'all', 'mystyle_model.pt')
    else:
        if model == 0:
            experiment = 'lower_bound'
        path = os.path.join(training_runs_dir, celeb, experiment, str(model), 'mystyle_model.pt')

    return path

def get_net(celeb, experiment, model):
    return torch.load(get_net_path(celeb, experiment, model))

def reconstruction_loss(synth, target, lpips):
    dist = lpips(synth, target)
    l2_dist = (synth - target).square().mean()

    loss = dist + l2_dist

    return loss

def project(net, img, w_init, lpips, device):
    w_opt = w_init.clone().detach().requires_grad_(True).to(device)
    optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=0.005).to(device)

    steps = 800
    for step in tqdm(range(steps)):
        synth = net(w_opt, noise_mode='const', force_fp32=True)
        loss = reconstruction_loss(synth, img, lpips)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return w_opt

def project_single_image(net, sample, save_dir, device, lpips):
    
    save_path = save_dir.joinpath('recon_images', sample['name']).with_suffix('.jpg')
    if save_path.exists():
        print(f'Projected image for model {model} video {video} already exists. Skipping.')
        return
    img = sample['img'].to(device)
    w_init = sample['w_code'].to(device)

    w_final = project(net, img, w_init, lpips, device)
    final_image = net(w_final, noise_mode='const', force_fp32=True)

    # save final_image
    save_images(final_image, save_path)

    # save input image
    input_save_path = save_dir.joinpath('input_images', sample['name'])
    save_images(img, input_save_path)

    # save latent
    latent_save_path = save_dir.joinpath('latents', sample['name']).with_suffix('.pt')
    save_latents(w_final, latent_save_path)

def process_args():
    parser = argparse.ArgumentParser(description='Batch Eval Celeb')

    
    
    # Required arguments
    parser.add_argument('--celeb', type=str, help='Name of the celebrity', required=True)
    parser.add_argument('--experiment', type=str, help='Experiment name', required=True)
    parser.add_argument('--model', type=str, help='models to evaluate', required=True)
    parser.add_argument('--sample', type=str, help='Sample to evaluate', required=True)
    parser.add_argument('--save_dir', type=str, help='Directory to save results', required=True)
    parser.add_argument('--device', type=str, help='Device to run on', default='0')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = process_args()
    device = torch.device('cuda')
    lpips = LPIPS(net='alex').eval().to(device)
    
    celeb, experiment, model, sample, save_dir = args.celeb, args.experiment, args.model, args.sample, args.save_dir
    net = get_net(celeb, experiment, model).eval()
        
    #print(f'Projecting image {sample["name"]} for model {model} video {video}.')
    project_single_image(net, sample, save_dir, lpips)