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
import pandas as pd
import sklearn.preprocessing as sk

np.set_printoptions(threshold=4000)




def apply_filter(filepath):

	# W1600545658_1_cal.rpjb, W1597976395_1_cal.rpj1
	# reading file into data
	filename = filepath.split('/')[-1].split('_')[0]

	# reading image
	idl = io.readsav((filepath))
	datapoints = idl.rrpi
	datapoints=copy.copy(datapoints)
	m, n = datapoints.shape


	#quantizing to remove shadow
	# test to see if this works across folders
	q = Image.fromarray(np.uint8(datapoints*255))
	enhancer = ImageEnhance.Brightness(q)
	q = enhancer.enhance(4)
	q = q.quantize(2)
	datapoints_q = np.array(q)

	for y, row in enumerate(datapoints_q):
		zeroes = np.where(row == 1)
		for x in zeroes:
			datapoints[y, x] = 0



	#crop here
	y = 150
	non_zero = np.where(datapoints[y] != 0)[0]
	print(type(non_zero))


	datapoints=datapoints[160:340, non_zero[0]:non_zero[-1]]#cropping pixels to where the spokes are
	m,n=datapoints.shape

	
	for i in range(m):
		# removes the top 20 brightest pixels by replacing them with the 21st brightest per row of pixels
		top20 = np.argsort(datapoints[i])[-21:]
		for j in top20:
			datapoints[i,j] = datapoints[i,top20[0]]

		#subtract med from all pixels
		med=np.median(datapoints[i,:])
		datapoints[i,:] =[(datapoints[i,j]-med) for j in range(n)]


	return datapoints, filename

def save_image(filt_image, filename):
	plt.figure()
	plt.axis('off')
	fig = plt.imshow(filt_image,cmap = plt.get_cmap('gray'),origin='upper')
	plt.savefig(f"testing/081_{filename}_cf.png",bbox_inches='tight',transparent=True, pad_inches=0, dpi=300)
	plt.close()

if __name__ == "__main__":

	for filepath in glob.glob(f"../data/rpj/081_SPKMVLFLP/*.rpjb"):
		filt_image, filename = apply_filter(filepath)
		save_image(filt_image, filename)
		print(filename+" has been saved")




