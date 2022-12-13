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


	# Quantize - make a copy of the image space, but put a 1 where data is and a 0 where shadow/where data is lacking
	# Can only filter rectangular data, so if we find max and min of x and ys of data, we can make perfect rectangle around it
	# then fill in any remaining space, and filter small values out using our quantized version. 
	# Then tada, rectangle of non-rectangluar data!
	plt.imshow(pixel_values, cmap="gray")
	plt.show()
	#will likely need new method of quantizing, seeing as this doens't exactly do what I want
	pixel_values, quant = apply_quantize(pixel_values)
	


	plt.imshow(quant, cmap="gray")
	plt.show()

	plt.imshow(pixel_values, cmap="gray")
	plt.show()

	# lame way of doing this. to find x and y, every time I find a small index on y axis or 
	bounds_x = []
	bounds_y = []
	last_sum = 0


	# ^1 means XOR. If 0 or 1, bitwise exclusive or. Flipping 0s and 1s
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



	pixel_values=pixel_values[min(bounds_y):max(bounds_y), min(bounds_x):max(bounds_x)]



	plt.imshow(pixel_values, cmap="gray")
	plt.show()
	for i in range(len(pixel_values)):
		#pixel_values[i][pixel_values[i] == 0] = np.median(pixel_values[i].flatten())
		for j in range(len(pixel_values[i])):
			if pixel_values[i][j] == 0.0:
				pixel_values[i][j] = np.median(pixel_values[i].flatten())



	plt.imshow(pixel_values, cmap="gray")
	plt.show()
	for i in range(len(pixel_values)):
		pixel_values[i][pixel_values[i] == 0] = np.median(pixel_values[i,np.nonzero(pixel_values[i])[0]])


	plt.imshow(pixel_values, cmap="gray")
	plt.show()
	pixel_values = apply_median(pixel_values)


	plt.imshow(pixel_values, cmap="gray")
	plt.show()
	pixel_values = gaussian_filter(pixel_values, sigma = 8)


	plt.imshow(pixel_values, cmap="gray")
	plt.show()

	exit()
	


	return pixel_values, filename

def save_image(filt_image, filename):
	plt.figure()
	plt.axis('off')
	fig = plt.imshow(filt_image,cmap = plt.get_cmap('gray'),origin='upper')
	plt.savefig(f"../data/testing/new_cropping_with_rays/102_SPKTRKLF{filename}_cf.png",bbox_inches='tight',transparent=True, pad_inches=0, dpi=300)
	plt.close()

if __name__ == "__main__":
	rays = ["W1597972495_1_cal.rpjb", "W1597974445_1_cal.rpjb", "W1597985170_1_cal.rpjb", "W1597996870_1_cal.rpjb", "W1597997845_1_cal.rpjb"]
	rays_path = "../data/rpj/081_SPKMVLFLP/*.rpjb"

	curves = ["W1612302742_1_CALIB.rpjb"]
	curves_path = "../data/rpj/102_SPKTRKLF/*.rpjb"



	for filepath in glob.glob(rays_path):
		if filepath.split('/')[4] in rays: 
			filt_image, filename = apply_filter(filepath)
			save_image(filt_image, filename)
			print(filepath)
			#print(filename+" has been saved")




