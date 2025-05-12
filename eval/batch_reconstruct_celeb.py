import os, sys
sys.path.append('/playpen-nas-ssd/awang/mystyle_original')
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

lpips = LPIPS(net='alex').eval()

def get_net_path(celeb, experiment):
    training_runs_dir = '/playpen-nas-ssd/awang/mystyle_original/training-runs'
    if experiment == 'upper_bound':
        path = os.path.join(training_runs_dir, celeb, experiment, 'all', 'mystyle_model.pt')
    else:
        path = os.path.join(training_runs_dir, celeb, experiment, '9', 'mystyle_model.pt')

    return path

def get_net(celeb, experiment):
    return torch.load(get_net_path(celeb, experiment))

def reconstruction_loss(synth, target):
    dist = lpips(synth, target)
    l2_dist = (synth - target).square().mean()

    loss = dist + l2_dist

    return loss

def project(net, img, w_init):
    w_opt = w_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=0.005)

    steps = 800
    for step in tqdm(range(steps)):
        synth = net(w_opt, noise_mode='const', force_fp32=True)
        loss = reconstruction_loss(synth, img)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return w_opt

def project_all_images(net, dataset, save_dir, device):
    if os.path.isdir(save_dir.joinpath('recon_images')) and len(os.listdir(save_dir.joinpath('recon_images'))) == len(dataset):
        print('All reconstructions done already. Skipping.')
        return

    for i, sample in enumerate(dataset):
        print(f'Projecting image {i+1}/{len(dataset)}')
        save_path = save_dir.joinpath('recon_images', sample['name'])
        if save_path.exists():
            print(f'Projected {sample["name"]} already exists. Skipping.')
            continue

        img = sample['img'].to(device)
        w_init = sample['w_code'].to(device)

        w_final = project(net, img, w_init)
        final_image = net(w_final, noise_mode='const', force_fp32=True)

        # save final_image
        save_images(final_image, save_path)

        # save input image
        input_save_path = save_dir.joinpath('input_images', sample['name'])
        save_images(img, input_save_path)

        # save latent
        latent_save_path = save_dir.joinpath('latents', sample['name']).with_suffix('.pt')
        save_latents(w_final, latent_save_path)

def generate(celeb, experiment, latent_dir, output_path, device):
    generator_path = get_net_path(celeb, experiment)
    os.chdir('/playpen-nas-ssd/awang/mystyle_original')
    #os.chdir("/playpen-nas-ssd/awang/semantic_editing/mystyle")
    command = 'python generate.py ' \
            + f'--anchors_path={latent_dir} ' \
            + f'--generator_path={generator_path} ' \
            + f'--output_path={output_path} ' \
            + f'--device={device}'
    os.system(command)
    #os.chdir('/playpen-nas-ssd/awang/mystyle_original/eval')

def synthesize_images(net, dataset, save_dir, device):
    pass

def eval_synthesis_ffhq(synthesized_dataset):
    pass

def process_args():
    parser = argparse.ArgumentParser(description='Batch Train Celeb')
    
    # Required arguments
    parser.add_argument('--celeb', type=str, help='Name of the celebrity', required=True)
    parser.add_argument('--experiment', type=str, help='Experiment name', required=True)

    # Optional arguments
    parser.add_argument('--device', type=str, help='GPUs to use', required=False, default='0')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    return args

if __name__ == '__main__':
    args = process_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    net = get_net(args.celeb, args.experiment).eval().to(device)
    lpips.to(device)

    celeb_dir = os.path.join('/playpen-nas-ssd/awang/data/mystyle', args.celeb)
    reconstruction_out_dir = os.path.join('/playpen-nas-ssd/awang/mystyle_original/out', args.celeb, 'reconstructions', args.experiment)
    for video in range(0, 10):
        # reconstruct all test images for the video
        test_image_dir = Path(os.path.join(celeb_dir, str(video), 'test', 'preprocessed'))
        test_latent_dir = Path(os.path.join(celeb_dir, str(video), 'test', 'anchors'))
        save_dir = Path(os.path.join(reconstruction_out_dir, str(video)))

        dataset = PersonalizedDataset(test_image_dir, test_latent_dir)

        print(f'Projecting images for {args.celeb} video {video} {args.experiment}')
        project_all_images(net, dataset, save_dir, device)

    for video in range(0, 10):
        # generate images from latents for the training run for this video
        latent_dir = Path(os.path.join(celeb_dir, str(video), 'train', 'anchors'))
        out_dir = Path(os.path.join('/playpen-nas-ssd/awang/mystyle_original/out', args.celeb, 'synthesis', args.experiment, str(video)))

        print(f'Generating images for {args.celeb} video {video} {args.experiment}')
        generate(args.celeb, args.experiment, latent_dir, out_dir, args.device)

