import matplotlib.pyplot as plt
import scipy.io as io
import numpy as np
import copy
import cv2
from PIL import Image
from PIL import ImageEnhance as IE
from PIL import ImageStat as IS
import os
import glob
from matplotlib import cm
import sklearn.preprocessing as sk
from scipy.ndimage import gaussian_filter

np.set_printoptions(threshold=4000)

#quantizing to remove shadow
def apply_quantize(pixel_values):
	p = 255*(pixel_values-pixel_values.min())/(pixel_values.max()-pixel_values.min())
	q = Image.fromarray(np.uint8(p))

	#Jack the brightenss to make quantize easuer
	enhancer = IE.Brightness(q)
	q = enhancer.enhance(4)
	q = q.quantize(10, Image.Quantize.MAXCOVERAGE)#different methods for how to quant: Quantize.MEDIANCUT, MAXCOVERAGE, FASTOCTREE
	quant = np.array(q)
	quant[quant != 0] = 1 # Quantizer can grab shape easier if we give it a few bins. To make a binary mask, force all non-zero values to 1

	

	#Go over 
	for y, row in enumerate(quant):
		zeroes = np.where(row == 0)
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
	# With Lucy's code, we can only filter rectangular data, so if we find max and min of x and ys of data, we can make perfect rectangle around it
	# then fill in any remaining space, and filter small values out using our quantized version. 
	# Then tada, rectangle of non-rectangluar data!

	pixel_values, quant = apply_quantize(pixel_values)
	

	

	########################################################################
	# crop pixel_values so that the max/min x values match the min/max non-zero x values of the quant mask

	nonzeros = np.asarray(np.nonzero(quant)).flatten()
	nonzeros[nonzeros == 0] = 9999999


	x_min = nonzeros.min()
	nonzeros[nonzeros == 9999999] = 0
	x_max = nonzeros.max()

	pixel_values=pixel_values[:, x_min:x_max]

	

	
	################################################################
	# The below bits of code are a list of fitlers. The two forloops seem to be doign tbe same thing, will test in a bit. 
	# They go through the image and replace all instances of 0 values with the median of that row. 
	# This removes the striping caused by the median filter we've seen before. 
	# We then apply the median filter, which also removes any detected cosmic rays. 
	# The final filter is a gaussian_filter. 
	for i in range(len(pixel_values)):
		for j in range(len(pixel_values[i])):
			if pixel_values[i][j] == 0.0:
				pixel_values[i][j] = np.median(pixel_values[i].flatten())

	for i in range(len(pixel_values)):
		pixel_values[i][pixel_values[i] == 0] = np.median(pixel_values[i,np.nonzero(pixel_values[i])[0]])



	pixel_values = apply_median(pixel_values)
	plt.imshow(pixel_values, cmap="gray")
	plt.show()

	pixel_values = gaussian_filter(pixel_values, sigma = 3)
	plt.imshow(pixel_values, cmap='gray')
	plt.show()


	return filename, pixel_values

def save_image(new_path, filt_image):
	plt.figure()
	plt.axis('off')
	fig = plt.imshow(filt_image,cmap = plt.get_cmap('gray'))
	plt.savefig(new_path,bbox_inches='tight',transparent=True, pad_inches=0, dpi=300)
	plt.close()

if __name__ == "__main__":
	# rays = ["W1597972495_1_cal.rpjb"]#, "W1597974445_1_cal.rpjb", "W1597985170_1_cal.rpjb", "W1597996870_1_cal.rpjb", "W1597997845_1_cal.rpjb"]
	# rays_path = "../data/rpj_old/081_SPKMVLFLP/*.rpjb"

	# curves = ["W1612302742_1_CALIB.rpjb"]
	# curves_path = "../data/rpj_old/102_SPKTRKLF/*.rpjb"
	rpj_new = "/Users/willbyrne/Documents/work/code/hamilton/unet_spokes/data/rpj_new/"
	thumb = "/Users/willbyrne/Documents/work/code/hamilton/unet_spokes/data/2023_thumbnails/"

	for filepath in glob.glob(rpj_new+"**/*.rpjb"):
		split = filepath.split("/")
		folder = split[-2]
		

		if not os.path.exists(thumb+folder+"/"):
			os.mkdir(thumb+folder+"/")
		
		filename, filt_image = apply_filter(filepath)

		new_thumb_path = thumb+folder+"/"+filename+".png"

		save_image(new_thumb_path, filt_image)
		exit()

		


