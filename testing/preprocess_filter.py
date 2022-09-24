import matplotlib.pyplot as plt
import scipy.io as io
import numpy as np
import copy
import cv2
from PIL import Image
from PIL import ImageEnhance
import os
import glob
from matplotlib import cm
np.set_printoptions(threshold=4000)

sizes = []

for filepath in glob.glob(f"../imagery/filtered/**/*.png"):
	img = Image.open(filepath)
	sizes.append(img.size)
sizes.sort()

print(sizes)





# def apply_filter(filepath):

# 	# W1600545658_1_cal.rpjb, W1597976395_1_cal.rpj1
# 	# reading file into data
# 	filename = filepath.split('/')[-1].split('_')[0]

# 	# reading image
# 	idl = io.readsav((filepath))
# 	datapoints = idl.rrpi
# 	datapoints=copy.copy(datapoints)
# 	m, n = datapoints.shape

# 	img = Image.fromarray(np.uint8(cm.gray(datapoints)*255))
# 	enhancer = ImageEnhance.Brightness(img)
# 	img = enhancer.enhance(4)
# 	img = img.quantize(2)

# 	datapoints_q = np.array(img)

# 	zero_index = np.where(datapoints_q[50] == 0)[0]

# 	print(zero_index[0],zero_index[-1])

# 	datapoints=datapoints[:, zero_index[0]+50:zero_index[-1]-50]#cropping pixels to where the spokes are
# 	m,n=datapoints.shape

# 	plt.imshow(img, cmap = 'gray')
# 	plt.show()
# 	exit()


# 	# minda=(min(datapoints.flatten()))
# 	# for i in range(m):
# 	# 	med=np.median(datapoints[i,:])
# 	# 	datapoints[i,:] =[(datapoints[i,j]-med) for j in range(n)]


# 	return datapoints, filename

# def save_image(filt_image, filename):
# 	plt.figure()
# 	plt.axis('off')
# 	fig = plt.imshow(filt_image,cmap = plt.get_cmap('gray'),origin='upper')
# 	plt.savefig(f"{filename}.png",bbox_inches='tight',transparent=True, pad_inches=0, dpi=300)
# 	plt.close()

# if __name__ == "__main__":

# 	for filepath in glob.glob(f"*.rpjb"):
# 		filt_image, filename = apply_filter(filepath)
# 		save_image(filt_image, filename)
# 		print(filename+" has been saved")




