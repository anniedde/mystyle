import os
import argparse
import requests
import shutil

encoder_checkpoint = '/playpen-nas-ssd/awang/mystyle_original/third_party/faces_w_encoder.pt'
pretrained_ffhq = '/playpen-nas-ssd/awang/mystyle_original/third_party/ffhq.pkl'

def notify(message):
    chat_id = '6712696502'
    TOKEN = '6643471688:AAH_8A5SrUe9eI-nAs90No_CI1T8H2KYqQE'
    user_id = '6712696502'
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url).json() # this sends the message

def run(device, images_dir, output_dir, generator_path, encoder_checkpoint=encoder_checkpoint):
    # create multiline string
    cmd = f'''
        python train.py \
        --device={device} \
        --images_dir={images_dir} \
        --output_dir={output_dir} \
        --generator_path={generator_path} \
        --encoder_checkpoint={encoder_checkpoint}
    '''
    print(cmd)
    os.system(cmd)

def get_last_snapshot(celeb, experiment_name, vid_num):
    if vid_num == 0:
        return pretrained_ffhq
    else:
        snapshot_path = f'/playpen-nas-ssd/awang/mystyle_original/training-runs/{celeb}/{experiment_name}/{vid_num - 1}/mystyle_model.pt'
        assert os.path.exists(snapshot_path), f'There is no previous snapshot to start running {celeb}, {experiment_name}, video {vid_num} from'
        return snapshot_path

def get_data_folder(celeb, vid_num):
    return f'/playpen-nas-ssd/awang/data/mystyle/{celeb}/{vid_num}/train/preprocessed'

def get_training_run_folder(celeb, experiment_name, vid_num):
    return f'/playpen-nas-ssd/awang/mystyle_original/training-runs/{celeb}/{experiment_name}/{vid_num}'

def run_lower_bound(args):
    celeb = args.celeb

    for vid_num in range(args.start, args.end):
        generator_path = get_last_snapshot(celeb, 'lower_bound', vid_num)
        images_dir = get_data_folder(celeb, vid_num)
        training_run_dir = get_training_run_folder(celeb, 'lower_bound', vid_num)
        try:
            notify(f'Starting training {celeb} lower bound vid {vid_num}')
            run(
                device=args.device,
                images_dir=images_dir,
                output_dir=training_run_dir,
                generator_path=generator_path
            )
            notify(f'Finished training {celeb} lower bound vid {vid_num}')
        except Exception as e:
            print(e)
            notify(f'Error training {celeb} lower bound vid {vid_num}: {e}')
            break

    notify(f'Finished training {celeb} lower bound')

def run_upper_bound(args):
    celeb = args.celeb

    generator_path = pretrained_ffhq
    images_dir = get_data_folder(celeb, 'all')
    training_run_dir = get_training_run_folder(celeb, 'upper_bound', 'all')
    try:
        notify(f'Starting training {celeb} upper bound.')
        run(
            device=args.device,
            images_dir=images_dir,
            output_dir=training_run_dir,
            generator_path=generator_path
        )
        notify(f'Finished training {celeb} upper bound')
    except Exception as e:
        print(e)
        notify(f'Error training {celeb} upper bound: {e}')