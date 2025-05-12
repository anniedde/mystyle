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


def process_args():
    parser = argparse.ArgumentParser(description='Batch Eval Celeb')
    
    # Required arguments
    parser.add_argument('--celeb', type=str, help='Name of the celebrity', required=True)
    parser.add_argument('--experiment', type=str, help='Experiment name', required=True)

    # Optional arguments
    parser.add_argument('--models', type=str, help='models to evaluate', required=False, default='0,1,2,3,4,5,6,7,8,9')
    parser.add_argument('--device', type=str, help='GPUs to use', required=False, default='0')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    return args

def get_net_path(celeb, experiment, model):
    training_runs_dir = '/playpen-nas-ssd/awang/mystyle_original/training-runs'
    if experiment == 'upper_bound':
        path = os.path.join(training_runs_dir, celeb, experiment, 'all', 'mystyle_model.pt')
    else:
        if model == 0:
            experiment = 'lower_bound'
        path = os.path.join(training_runs_dir, celeb, experiment, str(model), 'mystyle_model.pt')

    if not os.path.exists(path):
        raise Exception(f'Net path {path} does not exist.')

    return path

def get_net(celeb, experiment, model):
    return torch.load(get_net_path(celeb, experiment, model))

def reconstruction_loss(synth, target, lpips):
    #print('devices: ', synth.device, target.device)
    dist = lpips(synth, target)
    l2_dist = (synth - target).square().mean()

    loss = dist + l2_dist

    return loss

def project(net, img, w_init, lpips, device):
    w_opt = w_init.clone().detach().requires_grad_(True).to(device)
    optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=0.005)

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
        print(f'Projected image at {save_path} already exists. Skipping.')
        return
    img = sample['img'].to(device)
    w_init = sample['w_code'].to(device)

    try:
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
    except Exception as e:
        print(e)
        notify(f'Error in projecting image {sample["name"]}: {e}')


def project_all_images(net, dataset, save_dir, device):
    if os.path.isdir(save_dir.joinpath('recon_images')) and len(os.listdir(save_dir.joinpath('recon_images'))) == len(dataset):
        print('All reconstructions done already. Skipping.')
        return
    
    for i, sample in enumerate(dataset):
        print(f'Projecting image {i+1}/{len(dataset)}')
        save_path = save_dir.joinpath('recon_images', sample['name']).with_suffix('.jpg')
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

def generate(celeb, experiment, latent_dir, output_path, model, device):
    if os.path.isdir(output_path):
        print(f'{output_path} already exists. Skipping.')
        return

    generator_path = get_net_path(celeb, experiment, model)
    #os.chdir('/playpen-nas-ssd/awang/mystyle_original')
    """
    command = 'python generate.py ' \
            + f'--anchors_path={latent_dir} ' \
            + f'--generator_path={generator_path} ' \
            + f'--output_path={output_path} ' \
            + f'--device={device}'
    """
    raw_args = [
    '--anchors_path', latent_dir,
    '--generator_path', generator_path,
    '--output_path', output_path,
    '--device', device
]
    print(f'Generating images for {output_path}...')
    run_generate(raw_args)
    #print(command)
    #os.system(command)
    #os.chdir('/playpen-nas-ssd/awang/mystyle_original/eval')

def evaluate_synthesis(celeb, year, experiment, model):
    try:
        eval_dir = f'/playpen-nas-ssd/awang/mystyle_original/out/{celeb}/synthesis/{experiment}/{model}/{year}'
        json_file_path = os.path.join(eval_dir, 'metrics.json')
        if os.path.exists(json_file_path):
            print(f'Metrics already evaluated for {celeb} {model} {year}.')
            return
        else:
            reference_dir = f'/playpen-nas-ssd/awang/data/mystyle/{celeb}/{year}/test/preprocessed'
            images_dir = os.path.join(eval_dir, 'images')
            reference_name = f'{celeb}_{year}_mystyle'.lower()
            fid_value, mean_id_sim = evaluate_metrics(images_dir, reference_dir, reference_name, person_identifier)
            ret = {'fid': fid_value, 'id_sim': mean_id_sim}
            # Write mean_id_error to a JSON file
            with open(json_file_path, 'w') as f:
                json.dump(ret, f)
    except Exception as e:
        print(e)
        #notify(f'Error in evaluating synthesis results for {celeb} {model}: {e}')
    
    #notify(f'Finished evaluating synthesis results for {celeb} {model}, t={year}.')

def worker(gpu_id, task_queue):
    print(f'Worker {gpu_id} started.')
    device = torch.device(f'cuda:{gpu_id}')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    lpips = LPIPS(net='alex').eval().to(device)
    while not task_queue.empty():
        celeb, experiment, model, sample, save_dir = task_queue.get()
        net = get_net(celeb, experiment, model).eval().to(device)
        
        #print(f'Projecting image {sample["name"]} for model {model} video {video}.')
        project_single_image(net, sample, save_dir, device, lpips)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  # Set the start method to 'spawn'
    task_queue = multiprocessing.Queue()

    args = process_args()
    num_gpus = len(args.device.split(','))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    if args.experiment == 'upper_bound':
        models = ['all']
    else:
        models = [int(m) for m in args.models.split(',')]

    print('Adding tasks to queue...')
    for model in models:
        try:
            net = get_net(args.celeb, args.experiment, model).eval().to(device)
        except Exception as e:
            print(e)
            exit(1)
        #lpips.to(device)

        celeb_dir = os.path.join('/playpen-nas-ssd/awang/data/mystyle', args.celeb)
        out_dir = os.path.join('/playpen-nas-ssd/awang/mystyle_original/out', args.celeb, 'reconstructions', args.experiment, str(model))  
        for video in range(0, model + 1):
            # reconstruct all test images for the video
            test_image_dir = Path(os.path.join(celeb_dir, str(video), 'test', 'preprocessed'))
            test_latent_dir = Path(os.path.join(celeb_dir, str(video), 'test', 'anchors'))
            save_dir = Path(os.path.join(out_dir, str(video)))

            dataset = PersonalizedDataset(test_image_dir, test_latent_dir)
            for sample in dataset:
                save_path = save_dir.joinpath('recon_images', sample['name']).with_suffix('.jpg')
                if save_path.exists():
                    print(f'Projected {sample["name"]} for model {model} video {video} already exists. Skipping.')
                else:
                    task_queue.put((args.celeb, args.experiment, model, sample, save_dir))

    print('Tasks added to queue. Starting worker processes...')
    # Create and start worker processes

    try:
        processes = []
        for gpu_id in range(num_gpus):
            p = multiprocessing.Process(target=worker, args=(gpu_id, task_queue))
            p.start()
            processes.append(p)

        # Wait for all processes to finish
        for p in processes:
            p.join()
    except Exception as e:
        print(e)
        exit(1)