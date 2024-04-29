# check that cv2 imshow brings up window
import os
#os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
import cv2
from utils import id_utils
import os
#import tkinter as tk
#from tkinter import messagebox

import torch
from third_party.arcface.arcface import Backbone

import cv2
import numpy as np

import sys, os
import shutil
import logging
from pathlib import Path

sys.path.append('/playpen-nas-ssd/awang/mystyle_original')
from utils import id_utils

import pickle
from pathlib import Path

from third_party.stylegan2_ada_pytorch import dnnlib
from utils import latent_space_ops

import torch
import av #torchvisions
#from PIL import Image

cv2.namedWindow('test', cv2.WINDOW_AUTOSIZE)
print('hi')
img = cv2.imread('/playpen-nas-ssd/awang/data/videos/Michael/0/selected_frames/000706_0.05_67.48_0.29.png')
cv2.imshow('ImageWindow', img)
print('showed')
cv2.waitKey()
cv2.destroyAllWindows()
print(os.environ['DISPLAY'])