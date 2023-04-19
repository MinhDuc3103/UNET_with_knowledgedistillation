import matplotlib.pyplot as mpl
from PIL import Image

for i in range(1):
	a = Image.open(f"./output{i+1}.jpg")
	mpl.subplot(1, 2 ,1)
	mpl.imshow(a)
	b = Image.open(f"./pet_data/masks/Abyssinian_1.png")
	mpl.subplot(1, 2, 2)
	mpl.imshow(b)
	mpl.show()
