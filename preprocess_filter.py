import matplotlib.pyplot as plt
import scipy.io as io
import numpy as np
import copy
import cv2
from PIL import Image
import os
import glob


def apply_filter(filepath):

	# W1600545658_1_cal.rpjb, W1597976395_1_cal.rpj1
	# reading file into data
	filename = filepath.split('/')[-1].split('_')[0]

	# reading image
	idl = io.readsav((filepath))
	datapoints = idl.rrpi
	datapoints=copy.copy(datapoints)


	## filtering image
	# might want to check all images when given for upper/lower bounds cut off
	crop = np.where(datapoints[65, :] > 0.09)
	datapoints=datapoints[150:350, crop[0][0]+65:crop[0][-1]-65]#cropping pixels to where the spokes are
	m,n=datapoints.shape


	# minda=(min(datapoints.flatten()))
	# for i in range(m):
	# 	med=np.median(datapoints[i,:])
	# 	datapoints[i,:] =[(datapoints[i,j]-med) for j in range(n)]


	return datapoints, filename

def save_image(filt_image, filename, folder):
	plt.figure()
	plt.axis('off')
	fig = plt.imshow(filt_image,cmap = plt.get_cmap('gray'),origin='upper')
	plt.savefig(f"imagery/filtered/{folder}/{filename}.png",bbox_inches='tight',transparent=True, pad_inches=0, dpi=300)
	plt.close()
	exit()

if __name__ == "__main__":
	folder = '081_SPKMVLFLP'
	isExist = os.path.exists(f'imagery/filtered/{folder}/')

	if not isExist:
		os.makedirs(f'imagery/filtered/{folder}/')

	for filepath in glob.glob(f"../data/rpj/{folder}/*.rpjb"):
		filt_image, filename = apply_filter(filepath)
		save_image(filt_image, filename, folder)
		print(filename+" has been saved")




