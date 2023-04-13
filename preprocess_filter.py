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
import seaborn as sns
import sklearn.preprocessing as sk
from scipy.ndimage import gaussian_filter

np.set_printoptions(threshold=4000)


def longest_seq(row):
    LS = {"x_start": 0,
      "x_end": 0,
      "biggest_sum": 0}

    seq_sum,x_end, x_start = 0,0,0
    for num in range(0,len(row)-1):
        if row[num] == 0:
            if seq_sum >= LS["biggest_sum"]:
                x_end = num-1
                LS["x_start"] = x_start
                LS["x_end"] = x_end
                LS["biggest_sum"] = seq_sum
            seq_sum,x_end, x_start = 0,0,0
        else:
            seq_sum+= 1
            if seq_sum == 1:
                x_start = num
    return(LS)

def get_quant_stats(quant):
    top_quant = quant[0:int(quant.shape[0]*.08), :]

    LS = {"biggest_sum": 0,
      "x_start": 0,
      "x_end": 0,
	  "y_longst_top": 0}
    
    for y in range(0, len(top_quant)-1):
        curr_LS = longest_seq(top_quant[y])
        if curr_LS["biggest_sum"] >= LS["biggest_sum"]:
            LS = curr_LS
            LS["y_longst_top"] = y
    return LS

    # for y in range(0,len(top_quant)-1):
    #     current_sum = 0
    #     for x in range(0,len(top_quant[y])-1):
    #         if top_quant[y][x] == 1:
    #             current_sum += 1
    #             if top_quant[y][x-1] == 0:
    #                 start_x = x
    #         elif top_quant[y][x] == 0 and current_sum > LS['size']:
    #             LS['size'] = current_sum
    #             LS['x_start'] = start_x
    #             LS['x_end'] = x
    #             LS['y_longst_top'] = y


#quantizing to remove shadow
def apply_quantize(pixel_values):
	p = 255*(pixel_values-pixel_values.min())/(pixel_values.max()-pixel_values.min())
	q = Image.fromarray(np.uint8(p))

	#Jack the brightenss to make quantize easuer
	enhancer = IE.Brightness(q)
	q = enhancer.enhance(4)
	q = q.quantize(10, Image.Quantize.MAXCOVERAGE)#different methods for how to quant: Quantize.MEDIANCUT, MAXCOVERAGE, FASTOCTREE
	quant = np.array(q)
    
	# if there is a cosmic ray, 0 background will be set to 1 cuz of massive brighntess dif
	# quant[quant == 1] = 0
    # Quantizer can grab shape easier if we give it a few bins. To make a binary mask, force all non-zero values to 1
	quant[quant != 0 ] = 1 


	return pixel_values, quant

def apply_median(pixel_values):

	m,n=pixel_values.shape

	for i in range(m):
		# removes the top 20 brightest pixels by replacing them with the 21st brightest per row of pixels
		top20 = np.sort(pixel_values[i])[-21:]
		if top20[20]-top20[0] > 2*top20.std():
			for j in np.argsort(pixel_values[i])[-21:]:
				pixel_values[i,j] = top20[0]

		# subtract med from all pixels
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
	y, x = pixel_values.shape
	
	# remove anything thats too dark

	# Just so happens that shadow pixels happen to lie below 1 std from the mean
	# And that no pixels in appropriate geometric locations lie below that point
	flt = pixel_values.flatten()
	p_std = flt.std()
	p_m = flt.mean()

	pixel_values[pixel_values < (p_m - p_std)] = 0




	# Quantize - make a copy of the image space, but put a 1 where data is and a 0 where shadow/where data is lacking
	# With Lucy's code, we can only filter rectangular data, so if we find max and min of x and ys of data, we can make perfect rectangle around it
	# then fill in any remaining space, and filter small values out using our quantized version. 
	# Then tada, rectangle of non-rectangluar data!

	# pixel_values = apply_median(pixel_values)

	pixel_values, quant = apply_quantize(pixel_values)
	# plt.imshow(pixel_values, cmap = "gray")
	# plt.show()
	# plt.imshow(quant, cmap = "gray")
	# plt.show()


	

	

	########################################################################
	# crop pixel_values so that the max/min x values match the min/max non-zero x values of the quant mask

	LS = get_quant_stats(quant)
	# print(LS)
	# plt.imshow(quant, cmap="gray")
	# plt.show()
	# plt.imshow(pixel_values, cmap="gray")	
	# plt.show()
	# print(LS)
	# # plt.imshow(pixel_values, cmap = "gray")
	# # plt.show()
	# plt.imshow(quant, cmap = "gray")
	# plt.show()

	ybuffer = int(y*.1)
	xbuffer = int(x*.05)
	# new_path = "data/testing/crop_method_test/085_SPKMVLFLP_croptesting/"+"croptestR_"+filename+".png"


	pixel_values = pixel_values[ybuffer:y-ybuffer, LS["x_start"]+xbuffer:LS["x_end"]-xbuffer]


	return filename, pixel_values

def save_image(new_path, filt_image):
	plt.figure()
	plt.axis('off')
	fig = plt.imshow(filt_image,cmap = plt.get_cmap('gray'))
	plt.savefig(new_path,bbox_inches='tight',transparent=True, pad_inches=0, dpi=300)
	plt.close()

if __name__ == '__main__':
	testing_path = "data/testing/crop_method_test/full_test/"
	# probelm: W1602460352
	# suggestion: open rpi instead of rrpi and say remove anything less than 20 instead of less than 0.005
	# print(glob.glob("2023_rpjb/good/088_SPKMVLFLP/W1602467288*.rpjb"))

	for test_img in glob.glob("data/2023_rpjb/good/117_SPKMVLFHP_001/*.rpjb"):
		print(glob.glob("data/2023_rpjb/good/117_SPKMVLFHP_001/*.rpjb")[6])
		exit()
		folder = test_img.split("/")[-2]
		filename, pixel_values = apply_filter(test_img)
		save_image(testing_path+folder+"/"+filename+".png", pixel_values)

		index = glob.glob(f"data/2023_rpjb/good/{folder}/*.rpjb").index(test_img)+1
		length = len(glob.glob(f"data/2023_rpjb/good/{folder}/*.rpjb"))
		print(f"{index} out of {length} done ...")
	print("Complete!")
