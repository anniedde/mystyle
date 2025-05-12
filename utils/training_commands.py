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

def run(device, images_dir, output_dir, generator_path, encoder_checkpoint=encoder_checkpoint, replay_dir=None):
    # create multiline string
    cmd = f'''
        python train.py \
        --device={device} \
        --images_dir={images_dir} \
        --output_dir={output_dir} \
        --generator_path={generator_path} \
        --encoder_checkpoint={encoder_checkpoint} \
    '''
    if replay_dir:
        cmd += f'--replay_dir={replay_dir}'
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

def get_replay_folder(celeb, vid_num, replay_type):
    return f'/playpen-nas-ssd/awang/data/replay/{replay_type}/{celeb}/{vid_num}/replay/preprocessed'

def get_training_run_folder(celeb, experiment_name, vid_num):
    return f'/playpen-nas-ssd/awang/mystyle_original/training-runs/{celeb}/{experiment_name}/{vid_num}'

def run_lower_bound(args):
    celeb = args.celeb

    for vid_num in range(args.start, args.end):
        generator_path = get_last_snapshot(celeb, 'lower_bound', vid_num)
        images_dir = get_data_folder(celeb, vid_num)
        training_run_dir = get_training_run_folder(celeb, 'lower_bound', vid_num)
        if os.path.exists(training_run_dir):
            if os.path.exists(os.path.join(training_run_dir, 'mystyle_model.pt')):
                notify(f'Skipping training {celeb} lower bound vid {vid_num} because mystyle_model.pt already exists')
                continue
            else:
                shutil.rmtree(training_run_dir)
        
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

def run_replay_ransac(args):
    celeb = args.celeb

    for vid_num in range(max(args.start, 1), args.end):
        if vid_num == 1:
            generator_path = get_last_snapshot(celeb, 'lower_bound', vid_num)
        else:
            generator_path = get_last_snapshot(celeb, 'ransac', vid_num)
        images_dir = get_data_folder(celeb, vid_num)
        training_run_dir = get_training_run_folder(celeb, 'ransac', vid_num)
        replay_dir = get_replay_folder(celeb, vid_num)
        try:
            notify(f'Starting training {celeb} ransac vid {vid_num}')
            run(
                device=args.device,
                images_dir=images_dir,
                output_dir=training_run_dir,
                generator_path=generator_path,
                replay_dir=replay_dir
            )
            notify(f'Finished training {celeb} ransac vid {vid_num}')
        except Exception as e:
            print(e)
            notify(f'Error training {celeb} ransac vid {vid_num}: {e}')
            break

    notify(f'Finished training {celeb} ransac')

def run_replay_clustering(args):
    celeb = args.celeb

    for vid_num in range(max(args.start, 1), args.end):
        if vid_num == 1:
            generator_path = get_last_snapshot(celeb, 'lower_bound', vid_num)
        else:
            generator_path = get_last_snapshot(celeb, 'clustering', vid_num)
        images_dir = get_data_folder(celeb, vid_num)
        training_run_dir = get_training_run_folder(celeb, 'clustering', vid_num)
        replay_dir = get_replay_folder(celeb, vid_num, 'clustering')
        try:
            notify(f'Starting training {celeb} clustering replay vid {vid_num}')
            run(
                device=args.device,
                images_dir=images_dir,
                output_dir=training_run_dir,
                generator_path=generator_path,
                replay_dir=replay_dir
            )
            notify(f'Finished training {celeb} clustering replay vid {vid_num}')
        except Exception as e:
            print(e)
            notify(f'Error training {celeb} clustering replay vid {vid_num}: {e}')
            break

    notify(f'Finished training {celeb} clustering replay')

def run_replay_ransac_buffer_10(args):
    celeb = args.celeb

    for vid_num in range(max(args.start, 1), args.end):
        if vid_num == 1:
            generator_path = get_last_snapshot(celeb, 'lower_bound', vid_num)
        else:
            generator_path = get_last_snapshot(celeb, 'ransac_buffer_10', vid_num)
        images_dir = get_data_folder(celeb, vid_num)
        training_run_dir = get_training_run_folder(celeb, 'ransac_buffer_10', vid_num)
        replay_dir = get_replay_folder(celeb, vid_num, 'ransac_buffer_10')
        if os.path.exists(training_run_dir):
            if os.path.exists(os.path.join(training_run_dir, 'mystyle_model.pt')):
                notify(f'Skipping training {celeb} lower bound vid {vid_num} because mystyle_model.pt already exists')
                continue
            else:
                shutil.rmtree(training_run_dir)
        try:
            notify(f'Starting training {celeb} ransac_buffer_10 replay vid {vid_num}')
            run(
                device=args.device,
                images_dir=images_dir,
                output_dir=training_run_dir,
                generator_path=generator_path,
                replay_dir=replay_dir
            )
            notify(f'Finished training {celeb} ransac_buffer_10 replay vid {vid_num}')
        except Exception as e:
            print(e)
            notify(f'Error training {celeb} ransac_buffer_10 replay vid {vid_num}: {e}')
            break

    notify(f'Finished training {celeb} ransac_buffer_10 replay')

def run_replay(args):
    celeb = args.celeb
    experiment_name = args.experiment
    assert experiment_name in os.listdir('/playpen-nas-ssd/awang/data/replay'), f'Data for experiment {experiment_name} does not exist'

    for vid_num in range(max(args.start, 1), args.end):
        if vid_num == 1:
            generator_path = get_last_snapshot(celeb, 'lower_bound', vid_num)
        else:
            generator_path = get_last_snapshot(celeb, experiment_name, vid_num)
        images_dir = get_data_folder(celeb, vid_num)
        training_run_dir = get_training_run_folder(celeb, experiment_name, vid_num)
        replay_dir = get_replay_folder(celeb, vid_num, experiment_name)
        if os.path.exists(training_run_dir):
            if os.path.exists(os.path.join(training_run_dir, 'mystyle_model.pt')):
                notify(f'Skipping training {celeb} {experiment_name} vid {vid_num} because mystyle_model.pt already exists')
                continue
            else:
                shutil.rmtree(training_run_dir)
        try:
            notify(f'Starting training {celeb} {experiment_name} replay vid {vid_num}')
            run(
                device=args.device,
                images_dir=images_dir,
                output_dir=training_run_dir,
                generator_path=generator_path,
                replay_dir=replay_dir
            )
            notify(f'Finished training {celeb} {experiment_name} replay vid {vid_num}')
        except Exception as e:
            print(e)
            notify(f'Error training {celeb} {experiment_name} replay vid {vid_num}: {e}')
            break

    notify(f'Finished training {celeb} {experiment_name} replay')
