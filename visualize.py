import matplotlib.pyplot as mpl
from PIL import Image
import numpy as np
import sys
import os
from pathlib import Path
from os import listdir
from os.path import splitext, isfile, join


images_dir = Path('./pet_data/imgs/')
kd_dir = Path('./kd_output/')
no_kd_dir = Path('./no_kd_output/')
mask = Path('./pet_data/masks/')
teacher_dir = Path('./teacher_output/')

imgs = sorted([splitext(file)[0] for file in listdir(images_dir)])

for i in range(40, 60):
	kd_output_dir = os.path.join(kd_dir, imgs[i] +'.png')
	no_kd_output_dir = os.path.join(no_kd_dir, imgs[i] +'.png')
	true_mask = os.path.join(mask, imgs[i] + '.png')
	teacher_output = os.path.join(teacher_dir, imgs[i] + '.png')
	img1 = Image.open(kd_output_dir)
	img2 = Image.open(no_kd_output_dir)
	img3 = Image.open(true_mask)
	img4 = Image.open(teacher_output)
	plt0 = mpl.subplot(1,4,1)
	plt0.set_title("With KD")
	plt0.imshow(img1)
	#out = np.zeros((a.shape[-2], a.shape[-1], len(mask_values[0])), dtype=np.uint8
	plt1 = mpl.subplot(1, 4 ,2)
	plt1.set_title("Without KD")
	plt1.imshow(img2)

	plt2 = mpl.subplot(1, 4, 3)
	plt2.set_title("True mask")
	plt2.imshow(img3)

	plt3 = mpl.subplot(1, 4, 4)
	plt3.set_title("Teacher output")
	plt3.imshow(img4)
	mpl.show()
'''
np.set_printoptions(threshold=sys.maxsize)
a = Image.open("./output11.png")
c = np.array(a)
plt0 = mpl.subplot(1,3,1)
x = Image.open("./pet_data/imgs/Abyssinian_1.jpg")
plt0.set_title("Image")
plt0.imshow(x)
#out = np.zeros((a.shape[-2], a.shape[-1], len(mask_values[0])), dtype=np.uint8)
print(np.unique(c))
plt1 = mpl.subplot(1, 3 ,2)
plt1.set_title("Predict")
plt1.imshow(c)
b = Image.open(f"./pet_data/masks/Abyssinian_1.png")
d = np.array(b)
#print(np.unique(d))
plt2 = mpl.subplot(1, 3, 3)
plt2.set_title("True mask")
plt2.imshow(d)
mpl.show()
'''
