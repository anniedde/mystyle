import sys, os
import shutil
import logging
from pathlib import Path
import traceback
import requests

sys.path.append('/playpen-nas-ssd/awang/mystyle_original')
from utils import id_utils, io_utils

import numpy as np
import PIL
import PIL.Image
import scipy
import scipy.ndimage
import dlib
import argparse
from tqdm import tqdm

def notify(message):
    chat_id = '6712696502'
    TOKEN = '6643471688:AAH_8A5SrUe9eI-nAs90No_CI1T8H2KYqQE'
    user_id = '6712696502'
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url).json() # this sends the message

def parse_args(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--celeb", type=str, required=True)

    args = parser.parse_args(raw_args)
    return args


def main(raw_args=None):
    args = parse_args(raw_args)

    vids = [f'{i}' for i in range(10, 20)]
    vids.append('all')

    root_dir = f'/playpen-nas-ssd/awang/data/mystyle/{args.celeb}'

    for vid in vids:
        for split in ['train', 'test']:
            preprocessed_dir = os.path.join(root_dir, vid, split, 'preprocessed')
            raw_dir = os.path.join(root_dir, vid, split, 'raw')

            for file in os.listdir(raw_dir):
                assert file.endswith('.png'), 'File must be a .png file'
                assert file in os.listdir(preprocessed_dir), f'{file} not found in preprocessed_dir for {args.celeb} {vid} {split}'

    print('All clear!')

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        notify(f'Error in check_all_clear.py: {e}')
        traceback.print_exc()
        exit(1)