import matplotlib.pyplot as plt
import scipy.io as io
import numpy as np
import copy
import cv2
from PIL import Image
from PIL import ImageEnhance as IE
import os
import glob
from matplotlib import cm
import pandas as pd
import sklearn.preprocessing as sk
from scipy.ndimage.filters import gaussian_filter

np.set_printoptions(threshold=4000)


def apply_quantize(pixel_values):
	#quantizing to remove shadow
	# test to see if this works across folders
	q = Image.fromarray(np.uint8(pixel_values*255))
	enhancer = IE.Brightness(q)
	q = enhancer.enhance(4)
	q = q.quantize(2, Image.Quantize.MEDIANCUT)#different methods for how to quant: Quantize.MEDIANCUT, MAXCOVERAGE, FASTOCTREE
	quant = np.array(q)

	for y, row in enumerate(quant):
		zeroes = np.where(row == 1)
		for x in zeroes:
			pixel_values[y, x] = 0
	return pixel_values, quant

def apply_median(pixel_values):

	m,n=pixel_values.shape

	for i in range(m):
		# removes the top 20 brightest pixels by replacing them with the 21st brightest per row of pixels
		top20 = np.sort(pixel_values[i])[-21:]
		if top20[20]-top20[0] > 2*top20.std():
			for j in np.argsort(pixel_values[i])[-21:]:
				pixel_values[i,j] = top20[0]

		#subtract med from all pixels
		med=np.median(pixel_values[i,:])
		pixel_values[i,:] =[(pixel_values[i,j]-med) for j in range(n)]

	return pixel_values



def apply_filter(filepath):

	# W1600545658_1_cal.rpjb, W1597976395_1_cal.rpj1
	# reading file into data
	filename = filepath.split('/')[-1].split('_')[0]

	# reading image
	idl = io.readsav((filepath))
	pixel_values = idl.rrpi
	pixel_values=copy.copy(pixel_values)
	m, n = pixel_values.shape


	## filtering image
 	# might want to check all images when given for upper/lower bounds cut off
	# crop = np.where(pixel_values[65, :] > 0.09)
	# pixel_values=pixel_values[150:350, crop[0][0]+65:crop[0][-1]-65]#cropping pixels to where the spokes are
	# m,n=pixel_values.shape

	pixel_values, quant = apply_quantize(pixel_values)


	bounds_x = []
	bounds_y = []
	last_sum = 0
	#^1 means XOR. If 0 or 1, bitwise exclusive or. Flipping 0s and 1s
	# this for loop finds the smallest and largest index that has a 1. 
	for i in range(0, len(quant)):
		quant[i] = np.where((quant[i] == 0)|( quant[i] == 1),quant[i]^1, quant[i])
		if len(np.nonzero(quant[i])[0]) != 0:
			bounds_x.append(max(np.nonzero(quant[i])[0]))
			bounds_x.append(min(np.nonzero(quant[i])[0]))
			if last_sum == 0:
				bounds_y.append(i)
		else:
			if last_sum != 0:
				bounds_y.append(i)

		last_sum = len(np.nonzero(quant[i])[0])




	pixel_values=pixel_values[min(bounds_y)+20:max(bounds_y)-20, min(bounds_x):max(bounds_x)]

	for i in range(0, len(pixel_values)):
		pixel_values[i][pixel_values[i] == 0] = np.median(pixel_values[i].flatten())


	pixel_values = apply_median(pixel_values)
	pixel_values = gaussian_filter(pixel_values, sigma = 3)
	

	return pixel_values, filename

def save_image(filt_image, filename):
	plt.figure()
	plt.axis('off')
	fig = plt.imshow(filt_image,cmap = plt.get_cmap('gray'),origin='upper')
	plt.savefig(f"../data/testing/new_cropping_with_rays/081_SPKMVLFLP_{filename}_cf.png",bbox_inches='tight',transparent=True, pad_inches=0, dpi=300)
	plt.close()

if __name__ == "__main__":
	rays = ["W1597972495_1_cal.rpjb", "W1597974445_1_cal.rpjb", "W1597985170_1_cal.rpjb", "W1597996870_1_cal.rpjb", "W1597997845_1_cal.rpjb"]
	for filepath in glob.glob(f"../data/rpj/081_SPKMVLFLP/*.rpjb"):
		if filepath.split('/')[4] in rays: 
			filt_image, filename = apply_filter(filepath)
			save_image(filt_image, filename)
			print(filepath)
			#print(filename+" has been saved")




