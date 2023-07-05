import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

import numpy as np
import glob
import os

import scipy.io as io
import copy
from PIL import Image
from PIL import ImageEnhance as IE



np.set_printoptions(threshold=4000)

# Find the longest sequence of repeating 1s in a row. Record the end indexes, and the length of the sequence. 
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

# run longest sequence over all rows of pixel_values. Record the longest sequence in the top 8% of rows. 
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

# removes cosmic rays from the image. Might need to increase the total number of brightest values that get replaced.
def remove_cosmic_rays(pixel_values):
    m,n=pixel_values.shape

    for i in range(0, m):
        top20 = np.sort(pixel_values[i])[-21:]
        for j in np.argsort(pixel_values[i])[-21:]:
            pixel_values[i,j] = top20[0]        

    return pixel_values

# "Median filter" that lucy came up with. Does not follow the traiditonal median filter style
def apply_lucy_median(pixel_values):
	m,n=pixel_values.shape

	# First step. subtract the median of a given row of pixels from each pixel in that row
	# fix this so that if pixel_value - min < 0, set pixel_value to 0
	for i in range(m):
		med=np.median(pixel_values[i,:])
		pixel_values[i,:] = [(pixel_values[i,j]-med) for j in range(n)]
    
	# second step. Add absolute value of the lowest value pixel of a given image to each pixel 

	return pixel_values

# Adds a buffer to the image that makes sure the image has the given dimensions.
# This is dumb, will add more comments later.  
def buffer_image(pixel_values, propper_x, propper_y):
    
    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 10)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value


    med = np.median(pixel_values.flatten())
    old_y, old_x = pixel_values.shape	
    # These propper values are reliant on the cropping alrogithm. That means I'll need to change these later. 	
    act_x = 0
    act_y = 0	    
    if propper_y >= old_y:
        act_y = propper_y
    else:
        act_y = old_y	
    if propper_x >= old_x:
        act_x = propper_x
    else:
        act_x = old_x


    x_pad = int((propper_x - old_x)/2)+1
    y_pad = int((propper_y - old_y)/2)+1
    real_pad = 0

    if x_pad > y_pad:
        real_pad = x_pad
    else:
        real_pad = y_pad
    
    if real_pad < 0:
        mid_y = int(old_y/2)
        mid_x = int(old_x/2)

        biggest_dems = pixel_values[mid_y-80:mid_y+80, mid_x-368:mid_x+368]
    else:
        biggest_dems = np.pad(pixel_values, real_pad, pad_with, padder=med)


        while biggest_dems.shape[1] != propper_x:
            if biggest_dems.shape[1] % 2 == 0:
                biggest_dems = biggest_dems[:, :biggest_dems.shape[1]-1]
            else:
                biggest_dems = biggest_dems[:, 1:]
    
        while biggest_dems.shape[0] != propper_y:
            if biggest_dems.shape[0] % 2 == 0:
                biggest_dems = biggest_dems[:biggest_dems.shape[0]-1, :]
            else:
                biggest_dems = biggest_dems[1:, :]
        
    


    return biggest_dems

# saves the new image to the given path with no figure marks
def save_image(new_path, filt_image):
    plt.imsave(fname=new_path, arr=filt_image, cmap='gray', format='png', origin="lower")



# Calls all filters in the propper order. 
def apply_filters(filepath, plots = {'raw': False, 'cosmic_ray': False, 'outside_zero' : False, 'quant' : False, 'cropped' : False, 'lucy_median': False, 'forrier':False}):
		

	# W1600545658_1_cal.rpjb, W1597976395_1_cal.rpj1
	# reading file into data
	filename = filepath.split('/')[-1].split('_')[0]

	# reading image
	idl = io.readsav((filepath))
	pixel_values = idl.rrpi
	pixel_values=copy.copy(pixel_values)
	y, x = pixel_values.shape
	if plots['raw'] == True:
		print('raw')
		plt.imshow(pixel_values, cmap = 'gray', origin = 'lower')
		plt.show()

    
	pixel_values = remove_cosmic_rays(pixel_values)
	if plots['cosmic_ray'] == True:
		print('cosmic_ray')
		plt.imshow(pixel_values, cmap = 'gray', origin = 'lower')
		plt.show()
	
	# remove anything thats too dark

	# Just so happens that shadow pixels happen to lie below 1 std from the mean
	# And that no pixels in appropriate geometric locations lie below that point
	flt = pixel_values.flatten()
	p_std = flt.std()
	p_m = flt.mean()
	pixel_values[pixel_values < (p_m - p_std)] = 0
	if plots['outside_zero'] == True:
		print('outside_zero')
		plt.imshow(pixel_values, cmap = 'gray', origin = 'lower')
		plt.show()

	pixel_values, quant = apply_quantize(pixel_values)
	LS = get_quant_stats(quant)
	if plots['quant'] == True:
		print('quant')
		plt.imshow(quant, cmap = 'gray', origin = 'lower')
		plt.show()
    
	ybuffer = int(y*.1)
	xbuffer = int(x*.05)
	pixel_values = pixel_values[ybuffer:y-ybuffer, LS["x_start"]+xbuffer:LS["x_end"]-xbuffer]
	if plots['cropped'] == True:
		print('cropped')
		plt.imshow(pixel_values, cmap = 'gray', origin = 'lower')
		plt.show()
    
	return filename, pixel_values


# used to make the ~2000 training images
if __name__ == '__main__':
	testing_path = "data/2023_imagery/filtered/"
	# problem: W1602460352
	# suggestion: open rpi instead of rrpi and say remove anything less than 20 instead of less than 0.005
	# print(glob.glob("2023_rpjb/good/088_SPKMVLFLP/W1602467288*.rpjb"))

	image_path = glob.glob(f"data/2023_rpjb/good/*/{'W1597978345'}*")
	print(image_path)
	apply_filters(image_path[0])
	print("Complete!")
