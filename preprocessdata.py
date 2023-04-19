import matplotlib.pyplot as mpl
from PIL import Image
import os
from os.path import splitext, isfile, join
'''
for i in range(1):
	a = Image.open(f"./output{i+1}.jpg")
	mpl.subplot(1, 2 ,1)
	mpl.imshow(a)
	b = Image.open(f"./small_pet_data/masks/Abyssinian_1.png")
	mpl.subplot(1, 2, 2)
	mpl.imshow(b)
	mpl.show()
'''
directory = "./pet_data/imgs"
directory2 = "./pet_data/masks"
for filename in os.listdir(directory):
	f = os.path.join(directory, filename)
	ext = splitext(filename)[1]
	f2 = os.path.join(directory2, splitext(filename)[0] +'.png')
	if ext != '.jpg':
		os.remove(f)
	else:
		a = Image.open(f)
		if (len(a.split()) != 3):
			os.remove(f)
			os.remove(f2)

