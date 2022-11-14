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


def apply_quantize(image):
	#quantizing to remove shadow
	# test to see if this works across folders
	q = Image.fromarray(np.uint8(datapoints*255))
	enhancer = ImageEnhance.Brightness(q)
	q = enhancer.enhance(4)
	q = q.quantize(2)#different methods for how to quant
	datapoints_q = np.array(q)

	for y, row in enumerate(datapoints_q):
		zeroes = np.where(row == 1)
		for x in zeroes:
			datapoints[y, x] = 0
	return datapoints, datapoints_q



def apply_filter(filepath):

	# W1600545658_1_cal.rpjb, W1597976395_1_cal.rpj1
	# reading file into data
	filename = filepath.split('/')[-1].split('_')[0]

	# reading image
	idl = io.readsav((filepath))
	datapoints = idl.rrpi
	datapoints=copy.copy(datapoints)
	m, n = datapoints.shape

	## filtering image
 	# might want to check all images when given for upper/lower bounds cut off
	crop = np.where(datapoints[65, :] > 0.09)
	datapoints=datapoints[150:350, crop[0][0]+65:crop[0][-1]-65]#cropping pixels to where the spokes are
	m,n=datapoints.shape

	#datapoints, datapoints_q = apply_quantize(datapoints)

	#taking quantize data and 
	# top_id, bottom_id = 0,0
	# #get the row at the bottom based on quant
	# for row_id_bot in range(len(datapoints_q)-1, 0, -1):
	# 	if np.sum(datapoints_q[row_id_bot]) != len(datapoints_q[row_id_bot]):
	# 		bottom_id = row_id_bot-5
	# 		bottom_row = np.where(datapoints_q[row_id_bot-10] == 0)[0]
	# 		break;

	# for row_id_top in range(bottom_id, 0, -1):
	# 	if np.sum(datapoints_q[row_id_top]) == len(datapoints_q[row_id_top]):
	# 		top_id = row_id_top+5
	# 		break;
	# save_image(datapoints_q, "quant_"+filename)

	# print(filename)
	# print(bottom_id)
	# print(top_id)
	# # plt.imshow(datapoints_q, cmap = "gray")
	# # plt.show()

	# datapoints = datapoints[row_id_top:row_id_bot, bottom_row[0]:bottom_row[-1]]
	# m, n = datapoints.shape

	

	for i in range(m):
		# removes the top 20 brightest pixels by replacing them with the 21st brightest per row of pixels
		top20 = np.sort(datapoints[i])[-21:]
		if top20[20]-top20[0] > 2*top20.std():
			for j in np.argsort(datapoints[i])[-21:]:
				datapoints[i,j] = top20[0]

		#subtract med from all pixels
		med=np.median(datapoints[i,:])
		datapoints[i,:] =[(datapoints[i,j]-med) for j in range(n)]


	return datapoints, filename

def save_image(filt_image, filename):
	plt.figure()
	plt.axis('off')
	fig = plt.imshow(filt_image,cmap = plt.get_cmap('gray'),origin='upper')
	plt.savefig(f"../data/testing/new_median/rays_smoothed/081_SPKMVLFLP_{filename}_cf.png",bbox_inches='tight',transparent=True, pad_inches=0, dpi=300)
	plt.close()

if __name__ == "__main__":

	fig, axes = plt.subplots(2, 5, sharey=True, figsize=(30,3))
	fig.suptitle('Cosmic Ray Smoothing: Comparing Old median to New median filter')
	fig.tight_layout()


	i = 0
	for img_path in glob.glob("../data/testing/new_median/rays/*.png"):
		img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
		axes[0,i].imshow(img, cmap = 'gray')
		i+= 1

	i = 0
	for img_path in glob.glob("../data/testing/new_median/rays_smoothed/*.png"):
		img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
		axes[1,i].imshow(img, cmap = 'gray')
		i+= 1
	plt.show()
	exit()

	sns.imshow(ax=axes[1], data = data)
	sns.imshow(ax=axes[2], x=squirtle.index, y=squirtle.values)



	# rays = ["W1597972495_1_cal.rpjb", "W1597974445_1_cal.rpjb", "W1597985170_1_cal.rpjb", "W1597996870_1_cal.rpjb", "W1597997845_1_cal.rpjb"]
	# for filepath in glob.glob(f"../data/rpj/081_SPKMVLFLP/*.rpjb"):
	# 	if filepath.split('/')[4] in rays: 
	# 		filt_image, filename = apply_filter(filepath)
	# 		save_image(filt_image, filename)
	# 	#print(filename+" has been saved")




