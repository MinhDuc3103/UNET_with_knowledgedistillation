import logging
import numpy as np
import torch
import os
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from predict import *
from unet import UNet, Student
from PIL import Image

images_dir = Path('./pet_data/imgs/')
out_dir_path = Path('./kd_output/')
model_weights = './stu_with_kd/checkpoint_epoch5.pth'

imgs = sorted([splitext(file)[0] for file in listdir(images_dir)])
net = Student(n_channels=3, n_classes=3, bilinear=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


net.to(device=device)
state_dict = torch.load(model_weights, map_location=device)
mask_values = state_dict.pop('mask_values', [0, 1])
net.load_state_dict(state_dict)
for _, i in enumerate(imgs):
	print(_)
	img_dir = os.path.join(images_dir, i +'.jpg')
	out_dir = os.path.join(out_dir_path, i +'.png')
	img = Image.open(img_dir)
	mask = predict_img(net=net,
		           full_img=img,
		           scale_factor=1.0,
		           out_threshold=0.5,
		           device=device)
	result = mask_to_image(mask, mask_values)
	result.save(out_dir)

	
	
