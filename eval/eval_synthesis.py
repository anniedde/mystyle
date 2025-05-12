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
from cleanfid import fid
import math

sys.path.append('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/utils_copy')
from id_utils import PersonIdentifier

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
    mean_id_error = 0

    reference_features = [person_identifier.get_feature(open_image(os.path.join(reference_folder, im))) for im in os.listdir(reference_folder)]

    for im in tqdm(os.listdir(os.path.join(results_folder, 'images'))):
        im_path = os.path.join(results_folder, 'images', im)

        # get lowest distance from reference set
        im_feature = person_identifier.get_feature(open_image(im_path))
        sims = [person_identifier.compute_similarity(im_feature, reference_feature).item() for reference_feature in reference_features]
        max_sim = max(sims)
        mean_id_error += (1 - max_sim)

    mean_id_error /= len(os.listdir(os.path.join(results_folder, 'images')))
    
    ### FID
    fid_score = fid.compute_fid(results_folder, dataset_name=reference_name, mode="clean", dataset_res='na', dataset_split="custom")

    return fid_score, mean_id_error

def eval_synthesis_results(celeb, model, device):
    person_identifier = PersonIdentifier('/playpen-nas-ssd/awang/mystyle_original/third_party/model_ir_se50.pth', None, None)
    root_folder = os.path.join('/playpen-nas-ssd/awang/mystyle_original/out/', celeb, 'synthesis')

    # Load existing metrics if they exist
    if os.path.exists(os.path.join(root_folder, 'metrics.json')):
        with open(os.path.join(root_folder, 'metrics.json'), 'r') as f:
            maps = json.load(f)
        fid_map = maps['fid']
        id_error_map = maps['id_error']

        if model in fid_map and model in id_error_map:
            print('Metrics already exist for model {}. Skipping.'.format(model))
            return
    else:
        fid_map, id_error_map = {}, {}
        maps = {}

    fid_list, id_error_list = [], []

    times = [str(i) for i in range(10)]
    for t in times:
        print('Evaluating synthesis for celeb: {}, experiment: {}, time: {}'.format(celeb, model, t))

        results_folder = os.path.join(root_folder, model, t)
        reference_folder = os.path.join('/playpen-nas-ssd/awang/data/mystyle', celeb, t, 'test', 'preprocessed')
        reference_name = f'{celeb.lower()}_{t}'
        if not fid.test_stats_exists(reference_name, mode='clean'):
            print(f'Creating custom stats for {reference_name}...')
            fid.make_custom_stats(reference_name, reference_folder, mode="clean")
        
        fid_score, id_error = evaluate_metrics(results_folder, reference_folder, reference_name, person_identifier)
        id_error_list.append(id_error)
        fid_list.append(fid_score)
        
    #### Save to metrics.json
    id_error_map[model] = id_error_list
    fid_map[model] = fid_list

    maps['id_error'] = id_error_map
    maps['fid'] = fid_map
    
    with open(os.path.join(root_folder, 'metrics.json'), 'w') as f:
        json.dump(maps, f)

def process_args():
    parser = argparse.ArgumentParser(description='Batch Train Celeb')
    
    # Required arguments
    parser.add_argument('--celeb', type=str, help='Name of the celebrity', required=True)
    parser.add_argument('--experiment', type=str, help='Name of the experiment', required=True)

    # Optional arguments
    parser.add_argument('--device', type=str, help='GPUs to use', required=False, default='0')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    return args

if __name__ == '__main__':
    args = process_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eval_synthesis_results(args.celeb, args.experiment, device)
