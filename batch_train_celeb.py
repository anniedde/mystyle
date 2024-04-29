import os
import argparse
import requests
import shutil
import importlib

def notify(message):
    chat_id = '6712696502'
    TOKEN = '6643471688:AAH_8A5SrUe9eI-nAs90No_CI1T8H2KYqQE'
    user_id = '6712696502'
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url).json() # this sends the message

def process_args():
    parser = argparse.ArgumentParser(description='Batch Train Celeb')
    
    # Required arguments
    parser.add_argument('--celeb', type=str, help='Name of the celebrity', required=True)
    parser.add_argument('--experiment', type=str, help='Experiment name', required=True)

    # Optional arguments
    parser.add_argument('--start', type=int, help='Resume from video number', default=0)
    parser.add_argument('--end', type=int, help='End at video number', default=10)
    parser.add_argument('--device', type=str, help='GPUs to use', required=False, default='0')
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = process_args()
    module = importlib.import_module('utils.training_commands')
    function_name = f'run_{args.experiment}'
    run_function = getattr(module, function_name)
    run_function(args)

