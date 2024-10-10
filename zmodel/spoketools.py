import numpy as np
import math
import operator
import os
import sys 
import glob
import matplotlib.image as mpimg
import operator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pandas as pd
import copy
import scipy.signal as sc
from scipy.ndimage import gaussian_filter


#Functions within this spoketools.py library
#cropdata, fft2lpf, getspokes_row,getdark,getspokes_row2,gaus,peakwidth,clean_single,
#clean_short,connectline,clean_rows,getint_s,getint,getint_nr_s,getint_nr,sharpedg,findbound,findspoke_num,sortspk,
#cleanedge_spk,gauss_kern,blur_image,butter_lowpass,butter_lowpass_filter

#Functions used by FindDiff_2
#(A, B) means A is dependent on B
#cropdata, fft2lpf, (blur_image, gauss_kern), getdark, findbound, findspoke_num, (getint_nr, getint_nr_s), (cleanedge_spk, sortspk), input_file

#Functions used by FindDiff_gauss
#(A, B) means A is dependent on B
# getspokes_row2, (peakwidth, gauss), clean_single, clean_short, connectline, (reduce_highFn, (butter_lowpass_filter, butter_lowpass)), input_file


#functions that have been removed due to lack of use anywhere
#smoothdat, smoothdata_med, getspokes_row, clean_rows, getint_s,getint, sharpedg, expand_spokes, expand_spokes_row


spkcount=1
sharpedge=-2
bound=100
nonspk=-1000
exp_spk=999
peaks_ind=500


def global_threshold(datapoints, flag = 'dark', plot = False):
	
	"""
	Calculates a threshold pixel which seperates the background from the spokes and returns a 
	 white and black image with spokes and background respectively.
	
	Inputs:
		
		The cropped.and filtered 'datapoints' 
		
		A 'flag' saying whether the image is dark spokes or light spokes
		
		If you want to 'plot' the histogram along with the threshold pixel intensity.
		
	Outputs:
		
		A 'binary_array' where white is the spokes while black is the background
	"""
	
	dataflat=datapoints.flatten() #flattens 2d array to 1d
	
	mean = np.mean(dataflat)
	variance = np.var(dataflat)
	sigma = np.sqrt(variance)
	
	if flag == 'dark':
		critpix = mean-sigma #for dark spokes
		binary_array = np.where(datapoints < critpix, 255, 0).astype(np.uint8) #we want this for finding dark spokes. every pixel darker than getd is set to 255, and the rest to 0. 
	
	if flag == 'light':
		critpix_lower, critpix_upper = mean, mean + 2*sigma
		critpix = (critpix_lower, critpix_upper)
		
		binary_array = np.where((critpix_lower < datapoints) &
						   (datapoints < critpix_upper) , 255, 0).astype(np.uint8) #we want this for finding dark spokes. every pixel darker than getd is set to 255, and the rest to 0. 

	
	if plot == True:
		
		fig, ax = plt.subplots()
		ax.hist(dataflat, bins=np.linspace(min(dataflat),max(dataflat),200))
		ax.vlines(critpix, ymin=0, ymax=ax.get_ylim()[1], color='red')  # Adjust ymax to the maximum of the histogram
	
	m,n=datapoints.shape
	
	return binary_array


def get_critpix_col(datapoints):
	
	"""
	This thresholding function is different from global thresholding in that it calculates a 
	 threshold pixel for every column instead of a single threshold for the entire image. For now,
	 It is primarily used for bright spokes.
	
	Inputs:
		
		'datapoints' is an array of data. (ideally filtered)
		
	Outputs:
		
		Returns a 'binary_array' where white is the spokes detected while black is the background
		 This needs to be of the 'uint8' data type because of restrictions in the open_cv functions
		 Hopefully, this won't cause a lot of problems in the future :)
		
		'mean_array', 'sigma_array', 'critpix_array' are the mean, standard deviation, threshold
		 value of pixel intensities in each column of 'datapoints'.
		
	Note:
		
		Not sure how to use the mean, sigma, critpix values for things other than this function.
		 Have to think further.
		 
	
	"""
	
	binary_array = np.zeros_like(datapoints)
	
	rows, cols = datapoints.shape
	
	critpix_array = np.zeros(cols)
	mean_array = np.zeros(cols)
	sigma_array = np.zeros(cols)
	
	for i in range(0, cols):
		
		dataflat = datapoints[:, i]
		
		
		mean = np.mean(dataflat)
		variance = np.var(dataflat)
		sigma = np.sqrt(variance)
		
		critpix_lower, critpix_upper = mean + sigma , mean + 2*sigma
		
# 		critpix_lower, critpix_upper = mean - sigma , mean + 2*sigma
		
		mean_array[i] = mean
		sigma_array[i] = sigma
		critpix_array[i] = (critpix_lower)
		
		if critpix_lower < 0: # This is a condition where the critical pixel must be non negative for all columns. It makes sense intuitively that bright spoke pixels should not have negative intensity values. This helps remove some of the artifacts towards the right edge of the image where even the dark stuff is picked up as bright spokes (because we are esentially picking the brithest x% of the pixels in every column.)
			 critpix_lower = 0
		
		binary_array[:, i] = np.where((critpix_lower < dataflat),
							255, 0)
		
	return binary_array.astype('uint8'), mean_array, sigma_array, critpix_array

def deep_fourier(data, vert_height , vert_width = 0):
	
	
	img_float32 = np.float32(data)
	
	fshift = np.fft.fft2(img_float32)
	fshift = np.fft.fftshift(fshift)
	
	rows, cols = fshift.shape
	
	vert_row= int(rows/2 - vert_height)
	vert_col= int(cols/2 - vert_width)
	hor_row= int(rows/2 - 2)
	hor_col= int(cols/2 - 1)
	
	mask = np.ones(fshift.shape)  #hpf
	mask[0+vert_row:rows-vert_row, 0+vert_col:cols-vert_col] = 0 #for cross mask
	mask[0+hor_row:rows-hor_row, 0+hor_col:cols-hor_col] = 0 
	
	mask_on_fshift = fshift*mask
	mask_on_shift = np.fft.ifftshift(mask_on_fshift)
	final_img_back = np.real(np.fft.ifft2(mask_on_shift))
	
	return final_img_back


def apply_filter_variation(data, variation, median_kernel_size=7, sigma=4 ):
	"""
    About: This subroutine is for applying filters on data in any given order
    
	Input: 'data' is the data on which the filters are to be applied, and
           'variation' is the combination of filters applied. order matters &
           it should be a 'string' data type
    example: filter(datapoints,'fm') applies first the Fourier and then the 
	median filter on datapoints
	
	Output: returns the filtered data
	"""
	
	if variation == None:
		data = data
	
	else:
		string = [*variation]
	# print(string)
	
		for i in range(len(string)):
			if string[i]=='m':
				data = sc.medfilt2d(data, kernel_size=median_kernel_size)
			elif string[i]=='g':
				data = gaussian_filter(data, sigma)
			elif string[i]=='f':
				data = fft2lpf(data,passfiltrow=0,passfiltcol=3)
			elif string[i]=='c':
				data = cosmic_ray_rejection(data)
	
			else:
				print(r"{} in {} is not accepted. Please select from avaiable filters 'm','g','f','c'".format(string[i],variation))
				sys.exit(1)
		
	return data


def label_axes(datapoints, file, ax, name, color='gray'):

	'''
	This is a modified version of Will Byrne's label_figure function which plots the data with correct axes information.
	It plots some array of datapoints in a given row 'r', column 'c', gives it the title 'name', and uses the cmap 'color'.
	'''
	
	ax.imshow(datapoints,cmap = color,origin='lower')
	ax.set_title(name,fontsize=15)
	
	N = 5
	new_labels = np.round(np.linspace(file.mnlon, file.mxlon,N),2)
	xmin, xmax = ax.get_xlim()
	ax.set_xticks(np.linspace(xmin, xmax, N))
	ax.set_xticklabels(new_labels, minor = False)
	
	N = 3
	new_labels = np.round(np.linspace(file.mnrad, file.mxrad,N),2)
	ymin, ymax = ax.get_ylim()
	ax.set_yticks(np.linspace(ymin, ymax, N))
	ax.set_yticklabels(new_labels, minor = False)

	ax.set_ylabel("Radius")
	ax.set_xlabel("longitude")
	

def spoke_indices(datapoints, spokenum):
	'''gives the pixel indices of a given spoke. can use it in subtracting the sky intensity from the spokes'''
	
	
	indices = []
	
	for y in range(0, datapoints.shape[0]):
		for x in range(0, datapoints.shape[1]):
			if datapoints[y,x] == spokenum:
				indices.append((x,y))
				
	
	return indices

def rectangle_endpoints(contour):
	x_values = []
	y_values = []

	for point in contour:
		x,y = point[0]
		x_values.append(x)
		y_values.append(y)
		
	
	index_min = np.argmin(x_values)
	index_max = np.argmax(x_values)
	
	leftmost_endpoint = x_values[index_min], y_values[index_min]
	rightmost_endpoint = x_values[index_max], y_values[index_max]
	
	return leftmost_endpoint, rightmost_endpoint

def sky_rectangle(contour):
	
	x_values = []
	y_values = []
	
	for point in contour:
		x,y = point[0]
		x_values.append(x)
		y_values.append(y)
				
	mnlon,mxlon = np.min(x_values), np.max(x_values)
	mnrad, mxrad = np.min(y_values), np.max(y_values)
	
	top_left = mnlon, mxrad
	top_right = mxlon,mxrad
	bottom_left = mnlon,mnrad
	bottom_right = mxlon, mnrad
	
	return top_left, top_right, bottom_left, bottom_right

def sky_indices(contour,spokes):
	'''gives the indices for the sky background around a spoke. all zero intensity pixels in a rectangle around the spoke are counted as sky. rectangle is defined by four points. calculate minrad,maxrad,minlon,maxlon of points in a spoke and define a rectangle from this.'''
	top_left, top_right, bottom_left, bottom_right = sky_rectangle(contour)
	
	
	new_indices = []
	
	for y in range(bottom_left[1],top_left[1]+1):
		for x in range(top_left[0],bottom_right[0]+1):
			if spokes[y,x] == 0:
				new_indices.append((x,y))
				
	return new_indices

def sky_brightness(datapoints, contour, spokes):
	'''calculates the average sky brightness value in a rectangular region around a spoke. indices for rectangle are obtained from sky_indices'''
	indices = sky_indices(contour, spokes)
	
	sky_intensity_values = []
	
	for x, y in indices:
		pixel_intensity = datapoints[y, x]  # Note the y, x order for indexing
		sky_intensity_values.append(pixel_intensity)
		
	avg_sky = np.mean(sky_intensity_values)
	
	return avg_sky


def subtract_sky(original_data, contour, spokes, spokenum):
	
	data_without_sky = copy.deepcopy(original_data)
	
	sky_intensity = sky_brightness(data_without_sky,contour, spokes)
	spoke_location = spoke_indices(spokes, spokenum)
	
	for x,y in spoke_location:
		data_without_sky[y,x] = data_without_sky[y,x] - sky_intensity
		
	return data_without_sky

def intensity_correction(dataor, filtered_contours, spokes):
	
	data_corrected = dataor
	
	for i in range(0,len(filtered_contours)):
		data_corrected = subtract_sky(data_corrected, filtered_contours[i], spokes, i+1)
#		print(f"done x {i}")
		
	return data_corrected

def remove_outliers_in_row(row):
	'''In a single row of data, this function first finds peaks whose widths are less than 25 pixels, and replaces these pixels by the endpoints of the peaks'''
	peaks, _ = sc.find_peaks(row, prominence=0.0002)
	results_half = sc.peak_widths(row, peaks, rel_height=1, wlen = 40)
	
	for i in range(len(results_half[0])):
		if results_half[0][i] < 25:
			row[int(results_half[2][i]):int(results_half[3][i])] = np.mean([row[int(results_half[2][i])],row[int(results_half[2][i])]])
	
	return row

def cosmic_ray_rejection(datapoints):
	'''Applies remove_outliers_in_row to all the rows in datapoints, and returns an array removing all bright peaks that are less than 25 pixels long'''
	rows, cols = datapoints.shape
	
	for i in range(rows):
		datapoints[i,:] = remove_outliers_in_row(datapoints[i,:])
		
	return datapoints

def get_critical_pixel(datapoints):
	
	'''
	All pixels that are darker than 'getd', which is just mean-sigma, is counted
	as a spokes.
	This function is supposed to take as input the cropped. filtered data and calculate 
	the mean and standard deviation of the data array, giving the critical pixel value.
	'''
	dataflat=datapoints.flatten() #flattens 2d array to 1d
	
	# plt.hist(dataflat,bins=np.linspace(min(dataflat),max(dataflat),200),density=True, stacked=True normed=True)
	# plt.hist(dataflat,bins=np.linspace(min(dataflat),max(dataflat),200),density=True,stacked=True)
	#print(counts)
	mean = np.mean(dataflat)
	variance = np.var(dataflat)
	sigma = np.sqrt(variance)
	x = np.linspace(min(dataflat), max(dataflat), 100)
	# plt.plot(x, norm.pdf(x, mean, sigma))
	getd=mean-sigma #for dark spokes
# 	getd = mean+0.25*sigma #for light spokes
	m,n=datapoints.shape
	
	return getd


def color_spokes(binary_image, min_area_threshold = 200, max_area_threshold = 10**4):
	
	'''
	This funcrion will take in a binary array where 0s are the background, and 1s are the spokes
	and returns an array where each spoke have a unique integer. You can also remove small objects 
	that may not be spokes by changing the value of the second variable
	
	binary_image should be array of uint8 for some reason. it failed when the array was float32.
	min_area_threshold is the minimum square pixels a spokes should be. Anything smaller is removed.
	
	note: you need to import libraries cv2 and copy to run this function.
	'''
	
	
		# cv2.findContours finds the contours( boundaries of the spokes). returns a list (and a 'heirarchy', which we don't need and hence the '_')
		# first required input is a binary image/array, with 255 being spokes data, 0 being background
		# second arguemnt is cv2.RETR_EXTERNAL, which is one of the modes where contours inside contours are ignored. so all points inside the spoke are simply connected and have no holes in the topological sense. check other modes if you want to leave holes as is
		# third argument is about speed vs accuracy of the algorithm. the mode cv2.CHAIN_APPROX_SIMPLE is a compromise between the two. I have not found significant difference in using the other modes.
		
	contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	blank_image = np.zeros_like(binary_image)
	spokes = copy.copy(blank_image)
	
		# this function draws the 'contours' on the 'blank_image'. that explains the first two args
		# third argument is about which contour in the list of 'contours' to draw. a negative number tells it to draw all of them
		# fourth argument specifies the color of the contour. 255 is white (or yellow in matplotlib color scale)
		# fifth arguemnt specifies how thick the contour should be. a negative number fills in the contour.
		
	cv2.drawContours(blank_image, contours, -1, 255, thickness=1)
	
	
	filtered_contours = [] # cv2.contourArea calculate the area within a contour. we are making a new list 'filtered_contours' that will only contain spokes that are atleast 200 square pixels in area. can specify the area when calling the function.
	for cnt in contours:
	    if cv2.contourArea(cnt) > min_area_threshold and cv2.contourArea(cnt) < max_area_threshold:
	        filtered_contours.append(cnt)
			
	
	for i, contour in enumerate(filtered_contours):  #with the filtered list of contours, the last step is two draw them again on a new blank image, here called 'spokes'. we return this array and plotting it will show all the spokes in different colors.
	    cv2.drawContours(spokes, [contour], -1, i + 1, thickness=-1) # instead of all contours being 255, I'm giving each contour a different color.
	
	return spokes, filtered_contours


def spokestext(datapoints, spokes, rad_array, lon_array, file_path ):
	
	'''
	datapoints should be the array just before filtering. we want to write down the intensity values before the filters change them
	spokes should be your result from color_spokes. it is an array with background=0, and each spoke has a different integer value
	rad_array should be the array of radius values from the rpjb file
	lon_array is the array of longitude valies from the rpjb file
	file_path is where you want to save the text file
	
	note: you need to import pandas as pd to run this function
	'''
	
	data = []
	
	unique_spokes = np.unique(spokes)
	unique_spokes = unique_spokes[unique_spokes!=0]
	
	for spokenum in unique_spokes:
		pixels = np.argwhere(spokenum==spokes)
		for pixel in pixels:
			rad, long = pixel[0], pixel[1]
			data.append([rad_array[rad], lon_array[long], datapoints[rad,long], spokenum])
			
	
	df = pd.DataFrame(data, columns=['# Rad', 'Long', 'Intensity', 'Spoke Number'])
	df.to_csv(file_path, sep='\t', index=False)# Write the DataFrame to a text fil
	


# Will Byrne's method of removing comsic rays from an image
# For every row of pixels, grab the top 21 brightest pixels
# replace them all with the 21st brightest
def remove_cosmic_rays(datapoints):
    m,n=pixel_values.shape

    for i in range(0, m):
        top20 = np.sort(pixel_values[i])[-21:]
        for j in np.argsort(pixel_values[i])[-21:]:
            pixel_values[i,j] = top20[0]        

    return pixel_values


# properly labeling figures based on their radius and longitude from 
# rpjb files. bin is short for binary. 
def label_figure(bin, datapoints, ax=None):
    ax = ax or plt.gca()

    ax.imshow(datapoints, cmap = "gray", origin="lower")
    
    
    N = 5
    new_labels = np.round(np.linspace(bin.mnlon, bin.mxlon,N),2)
    
    xmin, xmax = ax.get_xlim()
    ax.set_xticks(np.linspace(xmin, xmax, N))
    ax.set_xticklabels(new_labels, minor = False)
    
    
    N=3
    new_labels = np.round(np.linspace(bin.mnrad, bin.mxrad,N),2)
    
    
    ymin, ymax = ax.get_ylim()
    ax.set_yticks(np.linspace(ymin, ymax, N))
    ax.set_yticklabels(new_labels, minor = False)
    
    ax.set_ylabel("Radius")
    ax.set_xlabel("longitude")
    
    return ax

def input_file(file_exten):
	input_file = input(f'please enter a valid {file_exten} file: ')
	if not os.path.exists('./'+input_file):
		print(f'{input_file} does not exist. Please try again')
		exit()#add this to spoke_vis and FindDiff_gauss
	return input_file

# function to crop data and give
def cropdata(datapoints,mxlon,mnlon,mxrad,mnrad):
	m,n=datapoints.shape
	nonzind=np.nonzero(datapoints[0,:])
	lon_array=np.linspace(mnlon,mxlon,n)
	rad_array=np.linspace(mnrad,mxrad,m)
	
	lon_array=lon_array[999:nonzind[0][-1]]
	rad_array=rad_array[200:700]
	return lon_array,rad_array,datapoints[200:700,999:nonzind[0][-1]]

import cv2
# 2d fft get rid of noise
def fft2lpf(datapoints,passfiltrow=0,passfiltcol=3):
	# passfilt(row/col): how many rows/col to get rid of
	img_float32 = np.float32(datapoints)
	
# 	plt.figure()
# 	plt.subplot(3,1,1)
# 	plt.title('original')
# 	plt.imshow(img_float32, cmap = plt.get_cmap('gray'),origin='upper')
# 	plt.show()
	
	
	dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)

	rows, cols = datapoints.shape

	# create a mask first, center square is 1, remaining all zeros
	mask = np.zeros((rows, cols, 2), np.uint8)
	mask[0+passfiltrow:rows-passfiltrow, 0+passfiltcol:cols-passfiltcol] = 1
	#print(mask)

	# apply mask and inverse DFT
	fshift = dft*mask
	img_back = cv2.idft(fshift)
	
# 	plt.figure()
	#print(img_back-img_float32)
# 	plt.subplot(3,1,2)
# 	plt.imshow(img_back[:,:,0],cmap = plt.get_cmap('gray'),origin='upper')
# 	plt.title('after 2d fft')
	#plt.show()
	
	return(img_back[:,:,0])

# get darkest pixels:
def getdark(datapoints,spokesdata,darkpix):          #in Find_Diff_2, spokesdata is a zero array 
	#darkpix: how many dark pixels to get to count as spokes
	m,n=datapoints.shape
	dataflat=datapoints.flatten()    #flattening the 2D array into 1D
	data,index=zip(*sorted(zip(dataflat,range(len(dataflat)))))  #sorting the array into an increasing order of pixel values
	#print([index[0:darkpix]])
	for i in range(len(data)):       
		#print(data[i])
		if data[i]>darkpix:   #darkpix is a number that we get from calculations for the histogram. if any pixel in the loop is bigger than darkpix, the loop breaks
			break
			
	#print(i)
	back2d=np.unravel_index([index[0:i]],(m,n)) # translating the 1D indices to 2D indices
	spokesdata[back2d]=spkcount  # this just means that: any pixel i with some pixel value less than darkpix, the pixel value gets reassigned to spkcount which is just 1
	#print(back2d)
	#plt.plot(back2d[1],back2d[0],'r.')
	return(spokesdata)
	

import matplotlib.pyplot as plt
from scipy.signal import find_peaks
# function to get points with spokes (global row version do peak finding)
def getspokes_row2(datapoints,spokesdata,prom=0.05,width_s=[1,1000],pw=0):
	# datapoints: data
	# spokesdata: data that contains spokes/non-spokes information
	# prom: Required prominence of peaks, only take <prom> fraction of the data.
	# width_s: Required width range of peaks in samples. 
	# rel_height_s: Chooses the relative height at which the peak width is measured as a percentage of its prominence. 1.0 calculates the width of the peak at its lowest contour line while 0.5 evaluates at half the prominence height. Must be at least 0. See notes for further explanation.
	# pw: mark the whole peak as spokes based on the width from find peak, 0 means no and 1 means yes
	prex=np.zeros(200)
	aftx=np.zeros(200)
	m,n=datapoints.shape

	print('getting data')
	widths=[[] for i in range(m)]
	width_heights=[[] for i in range(m)]
	prominences=[[] for i in range(m)]
	peaks_ar=[[] for i in range(m)]
	for i in range(m):
		prex[-1]=-datapoints[i,0]-(-datapoints[i,0]-min(-datapoints[i,:]))/len(prex)
		aftx[0]=-datapoints[i,-1]-(-datapoints[i,-1]-min(-datapoints[i,:]))/len(prex)
		for j in range(1,len(prex)):
			prex[len(prex)-j-1]=prex[-1]-(prex[-1]-min(-datapoints[i,:]))/len(prex)*j
			aftx[j]=aftx[0]-(aftx[0]-min(-datapoints[i,:]))/len(prex)*j

		x=np.append(np.append(prex,-datapoints[i,:]),aftx)
		peaks, dicts = find_peaks(x,prominence=0.0,width=1)
		#print(dicts)
		widths[i]=dicts['widths']
		width_heights[i]=dicts['width_heights']
		prominences[i]=dicts['prominences']
		peaks_ar[i]=peaks-len(prex)
		
	

	print('finished')
	
	prominencesf= []
	width_heightsf=[]
	for i in prominences:
		for j in i:
			prominencesf.append(j)
		
	for i in width_heights:
		for j in i:
			width_heightsf.append(j)
	
	counts, bins =np.histogram(prominencesf,100,normed=False)
	for i in range(len(counts)):
		if counts[i]<prom*counts[0]:
			break
	
	checkp=bins[i]
	print('sorting data')
	for i in range(m):
		for ind in reversed(range(len(widths[i]))):
			if width_heights[i][ind]<-0.06 or prominences[i][ind]<checkp or widths[i][ind]>width_s[1] or widths[i][ind]<width_s[0]:
				np.delete(peaks_ar[i],ind)
			else:
				spokesdata[i,peaks_ar[i][ind]]=spkcount
				
				if pw==1:
					leftid=peaks_ar[i][ind]-widths[i][ind]/5.
					rightid=peaks_ar[i][ind]+widths[i][ind]/5.
					if leftid<0:
						spokesdata[i,0:rightid]=spkcount
					elif rightid>n-1:
						spokesdata[i,leftid:n-1]=spkcount
					else:
						spokesdata[i,leftid:rightid]=spkcount
	

	print('finished')
	whereii,wherejj=np.where(spokesdata==spkcount)				

	
		
	return spokesdata, whereii, wherejj

from scipy.optimize import curve_fit
#from scipy import asarray as ar,exp    #YK edit
# get peak width by seeing decreasing from peak for each row
def gaus(x,a,x0,sigma): # define gaussian function
    return abs(a)*exp(-(x-x0)**2./(2.*sigma**2.))
    
    
def peakwidth(datapoints,spokesdata,peaknumb):
	m,n=datapoints.shape
	peaki,peakj=np.where(spokesdata==peaknumb)
	#print(peaki)
	peakbright_ar=datapoints[peaki,peakj]
	checki=range(min(peaki),max(peaki)+1) # how many rows it covered
	for i in range(len(checki)):
		#print(checki[i])
		ind_tot=np.where(peaki==checki[i])
		indi=ind_tot[0]
		#print(indi)
		# get peak index
		if len(indi)==0:
			x=range(indj-leftlen,indj+rightlen)
			indj=np.where(datapoints[checki[i],x]==min(datapoints[checki[i],x]))[0]
			if len(indj)!=1:
				indj=int(indj[0])
			else:
				indj=int(indj)
		else:
			indj_pre=peakj[np.where(peaki==checki[i])[0]]
			if len(indj_pre)==1:
				indj=int(indj_pre)
			else:
				datapeak=datapoints[np.ones(len(indj_pre))*indi,indj_pre] # find out the mins and which one is the min of the mins
				ind,_=np.where(datapeak==min(datapeak))
				indj=int(indj_pre[ind])
				
			
			# get the width to fit gaussian by checking when its increasing
			leftlen=0
			rightlen=0
			#print(len(datapoints[checki[i],0:indj]))
			inc=0
			for j in range(1,len(datapoints[checki[i],0:indj])): # left side
				if -datapoints[checki[i],indj-j]<-datapoints[checki[i],indj-j+1]:
					inc=0
					leftind=indj-j
					leftlen=leftlen+1
					
					
				else:
					break
			inc=0
			for j in range(indj,n-1): # right side
				#print(j)
				#print(checki[i])
				if -datapoints[checki[i],j+1]<-datapoints[checki[i],j]:
					inc=0
					rightind=j+1
					rightlen=rightlen+1
				else:
					break
					
			x=range(indj-leftlen,indj+rightlen)
			
			
		
		p0=[float(abs(-datapoints[checki[i],indj]-min(-datapoints[checki[i],x]))),float(indj),float(0.3*(max(x)-min(x)))]		
			
		try:
			popt,pcov = curve_fit(gaus,x,-datapoints[checki[i],x]-min(-datapoints[checki[i],x]),p0,maxfev=10000)
		except:
			pass
		
		
		width=popt[-1]
		spokesdata[checki[i],int(popt[1])-int(width):int(popt[1])+int(width)]=peaknumb
		plt.plot([int(popt[1])-int(width),int(popt[1])+int(width)],[checki[i],checki[i]],'b.',markersize=2)
		plt.plot(popt[1],checki[i],'r.',markersize=2)
		
	return spokesdata
		
		
	
	


# function to get rid of single peaks
def clean_single(spokesdata,extend=5,checkvalue=spkcount):
	# spokesdata: data that contains spokes/non-spokes information
	# extend: how many pixels to check connection in vertical
	# checkvalue: values to check
	m,n=spokesdata.shape
	indi,indj=np.where(spokesdata==checkvalue)
	for i in range(len(indi)):	
		if indi[i]==m-1:
			findi=np.append(np.where(indi==indi[i]-1), np.where(indi==indi[i]))
		elif indi[i]==0:
			findi=np.append(np.where(indi==indi[i]+1), np.where(indi==indi[i]))
		else:
			findi=np.append((np.append(np.where(indi==indi[i]+1), np.where(indi==indi[i]-1))),np.where(indi==indi[i]))
			
		#print(findi)
		if indj[i]==n-extend:
			checkj=[(indj[k]<indj[i] and indj[k]>indj[i]-extend) for k in findi]
			if any(checkj):
				if sum(checkj)==1:
					spokesdata[indi[i],indj[i]]=0
				else:
					continue
			else:
				spokesdata[indi[i],indj[i]]=0
		elif indj[i]==0:
			checkj=[(indj[k]>indj[i] and indj[k]<indj[i]+extend) for k in findi]
			if any(checkj):
				if sum(checkj)==1:
					spokesdata[indi[i],indj[i]]=0
				else:
					continue
			else:
				spokesdata[indi[i],indj[i]]=0
		else:
			checkj=[(indj[k]<indj[i]+extend and indj[k]>indj[i]-extend) for k in findi]
			if any(checkj):
				if sum(checkj)==1:
					spokesdata[indi[i],indj[i]]=0
				else:
					continue
			else:
				#print(spokesdata[indi[i],indj[i]])
				spokesdata[indi[i],indj[i]]=0
	
	return spokesdata

		 
# function to get rid of short peaks and identify different spokes
def clean_short(spokesdata,ss,extend=5):
	# extend: see clean_single() function		
	# ss: how many pixels to consider as short spokes
	m,n=spokesdata.shape
	indi,indj=np.where(spokesdata==spkcount) 
	skc=2
	while len(indi)>0:
		spokesdata[indi[0],indj[0]]=skc
		spki=[indi[0]]
		spkj=[indj[0]]
		spkinc_i=[indi[0]]
		spkinc_j=[indj[0]]
		spkinc_i_old=[]
		spkinc_j_old=[]
		inc=20
		while inc>0:
			inc=0
			for i in range(len(spkinc_i)):
				# get i's
				if spkinc_i[i]==m-1:
					findi=[(k==spkinc_i[i] or k==spkinc_i[i]-1) for k in indi]
				elif spkinc_i[i]==0:
					findi=[(k==spkinc_i[i] or k==spkinc_i[i]+1) for k in indi]
				else:
					findi=[(k==spkinc_i[i] or k==spkinc_i[i]+1 or k==spkinc_i[i]-1) for k in indi]
				checki=indi[findi]
				checkj=indj[findi]
				
				# check j's
				for j in range(len(checkj)):
					if (spokesdata[checki[j],checkj[j]]==spkcount) and checkj[j]<spkinc_j[i]+extend and checkj[j]>spkinc_j[i]-extend:
						spokesdata[checki[j],checkj[j]]=skc
						spki.append(checki[j])
						spkj.append(checkj[j])
						spkinc_i_old.append(checki[j])
						spkinc_j_old.append(checkj[j])
						inc=inc+1
			spkinc_i=spkinc_i_old
			spkinc_j=spkinc_j_old
			spkinc_i_old=[]	
			spkinc_j_old=[]
		#print(spkj,spki)
		if len(spki)<ss:
			spokesdata[spki,spkj]=0
		else:
			plt.plot(spkj,spki,'.')
		skc=skc+1
		indi,indj=np.where(spokesdata==spkcount)
	return spokesdata	

# function to connect peaks if they are close to each other and get rid of the short spokes
def connectline(spokesdata,ss,ss_h,spokesarr,extend=10,extend_h=50):	
	# spokesarr: spokes id to check
	# ss: how many pixels to consider as short spokes in rows
	# ss_h: how many pixels to consider as short spokes in columns
	# extend: how many rows to connect	
	# extend_h: how many columns to connect
	
	where_spk_i=[]
	where_spk_j=[]
	spk_ind=[]
	for i in spokesarr:
		where_spk_i_s,where_spk_j_s=np.where(spokesdata==i)
		if len(where_spk_i_s)==0:
			continue
			
		where_spk_i_s,where_spk_j_s=zip(*sorted(zip(where_spk_i_s,where_spk_j_s)))
		where_spk_i.append(where_spk_i_s)
		where_spk_j.append(where_spk_j_s)
		spk_ind.append(i)
	spc=2
	for i in range(len(where_spk_i)):
		for j in range(len(where_spk_i)):
			overlapar=len(set(where_spk_i[i]) & set(where_spk_i[j]))
			#print(overlapar)
			if j==i or overlapar>2:
				continue
			minii=min(where_spk_i[i])
			maxij=max(where_spk_i[j])
			
			minij=min(where_spk_i[j])
			maxii=max(where_spk_i[i])
			
			indminii=np.where(where_spk_i[i]==min(where_spk_i[i]))[0]
			indmaxii=np.where(where_spk_i[i]==max(where_spk_i[i]))[0]
			indminij=np.where(where_spk_i[j]==min(where_spk_i[j]))[0]
			indmaxij=np.where(where_spk_i[j]==max(where_spk_i[j]))[0]
			
			if len(indminii)==1:
				minji=where_spk_j[i][int(indminii)]
			else:
				minji=where_spk_j[i][int(indminii[0])]
			
			if len(indmaxii)==1:
				maxji=where_spk_j[i][int(indmaxii)]
			else:
				maxji=where_spk_j[i][int(indmaxii[0])]
			
			if len(indminij)==1:
				minjj=where_spk_j[j][int(indminij)]
			else:
				minjj=where_spk_j[j][int(indminij[0])]
				
			if len(indmaxij)==1:
				maxjj=where_spk_j[j][int(indmaxij)]
			else:
				maxjj=where_spk_j[j][int(indmaxij[0])]
			
			if minii>maxij and minii-maxij<extend:
				#print(minii,maxij)
				if abs(minji-maxjj)<extend_h:
					indn=min(spk_ind[i],spk_ind[j])
					spokesdata[where_spk_i[i],where_spk_j[i]]=indn
					spokesdata[where_spk_i[j],where_spk_j[j]]=indn
					spk_ind[i]=indn
					spk_ind[j]=indn
					
					whereind=np.where(spokesdata==indn)
			
			elif minij>maxii and minij-maxii<extend:
				if abs(maxji-minjj)<extend_h:
					indn=min(spk_ind[i],spk_ind[j])
					spokesdata[where_spk_i[i],where_spk_j[i]]=indn
					spokesdata[where_spk_i[j],where_spk_j[j]]=indn
					spk_ind[i]=indn
					spk_ind[j]=indn
					whereind=np.where(spokesdata==indn)
				
	indnew=np.unique(spk_ind)
	#print(indnew)
	nspk=0
	i=0
	for i in range(len(indnew)):
		#print(i)
		spi,pij=np.where(spokesdata==indnew[i])
		if abs(max(spi)-min(spi))<ss or abs(max(pij)-min(pij))<ss_h:
			nspk=nspk+1
			spokesdata[spi,pij]=0
		else:
			#print('nspk',nspk)
			#print('i',i)
			#print('2+i-nspk',2+i-nspk)
			spokesdata[spi,pij]=2+i-nspk
			plt.plot(pij,spi,'.')
	if i==nspk:
		print('spokes No.',0)		
	else:
		print('spokes No.',2+i-nspk-1)
	return spokesdata								

# function to fill in boundaries without recursive function
def getint_nr_s(spokesdata,bounddata):
	# spokesdata: data that contains spokes/non-spokes information
	# bounddata: the boundary number
	m,n=spokesdata.shape
	b=bounddata
	boundrange=range(bound+1,int(max(spokesdata.flatten())+1))
	fb=np.where(spokesdata==b)
	#print(fb)
	fbi=fb[0]
	fbj=fb[1]
	if len(fbi)==0:
		return spokesdata
	coloredfbi=np.array(range(min(fb[0]),max(fb[0])))
	coloredfbj=np.zeros(len(coloredfbi))
	# color the easy ones
	for i in np.unique(fb[0]):
		if i==0 or i==min(fb[0]):
			continue
		else:
			#print(i)
			index_co=np.where(coloredfbi==i)[0]
			ja_pre=[fbj[j] for j in range(len(fbj)) if (fbi[j]==i)]
			if len(ja_pre)==2:
				spokesdata[i,ja_pre[0]+1:ja_pre[1]]=b+bound
				
			else:
				ja_pre=sorted(ja_pre)
				ja=[[ja_pre[j],ja_pre[j+1]] for j in range(len(ja_pre)-1) if ja_pre[j+1]-ja_pre[j]>1] # get gap pairs
				if len(ja)==1:
					spokesdata[i,ja[0][0]+1:ja[0][1]]=b+bound
				else:	
					#print(ja_pre)
					jar=range(min(ja_pre),max(ja_pre)+1)
					finishc=0
					for k in range(len(jar)):
						if spokesdata[i,jar[k]]==spkcount or spokesdata[i,jar[k]]==nonspk or ((spokesdata[i,jar[k]] in boundrange) and spokesdata[i,jar[k]]!=b):
							spokesdata[i,jar[k]] = (b+bound)
					
					
					checkcon=np.zeros(len(jar)) # fill in blanks 
					for k in range(len(jar)):
						if k+1>=len(jar):
							if spokesdata[i,jar[k]]==b and spokesdata[i,jar[k-1]]==(b+bound):
								checkcon[k]=2
						elif k-1<0:
							if spokesdata[i,jar[k]]==b and spokesdata[i,jar[k+1]]==(b+bound):
								checkcon[k]=1
						else:
							#print(jar[k])
							#print(jar[k+1])
							if spokesdata[i,jar[k]]==b and spokesdata[i,jar[k+1]]==(b+bound):
								checkcon[k]=1
							elif spokesdata[i,jar[k]]==b and spokesdata[i,jar[k-1]]==(b+bound):
								checkcon[k]=2
					where1=np.where(checkcon==1)
					where2=np.where(checkcon==2)
					
					for k in range(min(len(where1[0]),len(where2[0]))):
						#print(k)
						if len(where1[0])==1:
							spokesdata[i,jar[where1[0][0]]:jar[where2[0][0]]]=b+bound
						else:
							spokesdata[i,jar[where1[0][k]]:jar[where2[0][k]]]=b+bound
							
	return spokesdata
			
# function to fill in boundaires w/o recursion
def getint_nr(spokesdata):
	# spokesdata: data that contains spokes/non-spokes information
	boundrange=range(bound+1,int(max(spokesdata.flatten())+1))
	for b in boundrange:
		getint_nr_s(spokesdata,b)
	return spokesdata	


# function to find boundaries
def findbound(spokesdata):
	m,n=spokesdata.shape
	b=0
	for i in range(m):
		# find verticle boundaries
		if i==0 or i==m-1:
			# find horizontal boundaries
			for j in range(n):
				if spokesdata[i,j]==spkcount:
					spokesdata[i,j]=bound
				
		else:
			# find horizontal boundaries
			for j in range(n):
				if j==0 or j==n-1:
					if spokesdata[i,j]==spkcount:
						spokesdata[i,j]=bound
				else:
					if spokesdata[i,j]==spkcount and spokesdata[i+1,j]!=spkcount and spokesdata[i+1,j]!=bound:
						spokesdata[i,j]=bound
					elif spokesdata[i,j]==spkcount and spokesdata[i-1,j]!=spkcount and spokesdata[i-1,j]!=bound:
						spokesdata[i,j]=bound
					elif spokesdata[i,j]==spkcount and spokesdata[i,j-1]!=spkcount and spokesdata[i,j-1]!=bound:
						spokesdata[i,j]=bound
					elif spokesdata[i,j]==spkcount and spokesdata[i,j+1]!=spkcount and spokesdata[i,j+1]!=bound:
						spokesdata[i,j]=bound		
	return spokesdata
	
from collections import Counter
# function to identify different spoke boundaries after finish identifying spokes boundary:
def findspoke_num(spokesdata,boundsiz,minrowsiz):
	# boundsiz: if the size of the boundary points are less than <boundsiz> then eliminate the spoke
	# minrowsiz: if the row size of the boundary points are less than <minrowsiz> then eliminate the spoke
	spokecount=1
	m,n=spokesdata.shape
	#orginal=len(np.where(spokesdata==bound)[0])
	while bound in spokesdata:
		wherebound=np.where(spokesdata==bound)
		starti=wherebound[0][0]
		startj=wherebound[1][0]

		boundnewzip=[(starti,startj)]
		boundtrack_o=boundnewzip
		
		newb=bound+spokecount # new number for new spoke's boundary
		inc=10
		spokesdata[starti,startj]=newb
		#print(len(wherebound[0]))
		while inc>0:
			inc=0
			boundtrack_n=[]
			for (si,sj) in boundtrack_o:
				if si!=m-1:
					if spokesdata[si+1,sj]==bound:
						spokesdata[si+1,sj]=newb
						boundnewzip.append((si+1,sj))
						boundtrack_n.append((si+1,sj))
						inc=inc+1
				if si!=0:
					if spokesdata[si-1,sj]==bound:
						spokesdata[si-1,sj]=newb
						boundnewzip.append((si-1,sj))
						boundtrack_n.append((si-1,sj))
						inc=inc+1
				if sj!=0:
					if spokesdata[si,sj-1]==bound:
						spokesdata[si,sj-1]=newb
						boundnewzip.append((si,sj-1))
						boundtrack_n.append((si,sj-1))
						inc=inc+1
				if sj!=n-1:
					if spokesdata[si,sj+1]==bound:
						spokesdata[si,sj+1]=newb
						boundnewzip.append((si,sj+1))
						boundtrack_n.append((si,sj+1))
						inc=inc+1
				if si!=m-1 and sj!=n-1:
					if spokesdata[si+1,sj+1]==bound:
						spokesdata[si+1,sj+1]=newb
						boundnewzip.append((si+1,sj+1))
						boundtrack_n.append((si+1,sj+1))
						inc=inc+1
				if si!=0 and sj!=0:
					if spokesdata[si-1,sj-1]==bound:
						spokesdata[si-1,sj-1]=newb
						boundnewzip.append((si-1,sj-1))
						boundtrack_n.append((si-1,sj-1))
						inc=inc+1
				if si!=0 and sj!=n-1:
					if spokesdata[si-1,sj+1]==bound:
						spokesdata[si-1,sj-1]=newb
						boundnewzip.append((si-1,sj+1))
						boundtrack_n.append((si-1,sj+1))
						inc=inc+1
				if si!=m-1 and sj!=0:
					if spokesdata[si+1,sj-1]==bound:
						spokesdata[si+1,sj-1]=newb
						boundnewzip.append((si+1,sj-1))
						boundtrack_n.append((si+1,sj-1))
						inc=inc+1
			boundtrack_o=boundtrack_n

		#indices for i and j for this spoke	
		jdel_a=[k[1] for k in boundnewzip]
		idel_a=[k[0] for k in boundnewzip]
		# find min and max for each row
		idel_a,jdel_a=zip(*sorted(zip(idel_a,jdel_a)))
		
		
		rows_s=list(Counter(idel_a).keys())
		colums_s=list(Counter(idel_a).values())
		# get how many columns are for each row
		for rowi in range(len(rows_s)):
			rownumbers=[int(ri) for ri in (np.where(idel_a==rows_s[rowi])[0])]
			#print(idel_a[min(rownumbers):max(rownumbers)+1])
			wherearerow=jdel_a[min(rownumbers):max(rownumbers)+1]
			colums_s[rowi]=(max(wherearerow)-min(wherearerow))+1
		if (len(boundnewzip)<boundsiz) or ((np.median(colums_s))/(len(rows_s)) > minrowsiz) or len(rows_s)<2:
			spokesdata[idel_a,jdel_a]=0 # need to change to non-spokes... just for visualization (revise)
			for i in range(min(idel_a),max(idel_a)+1):
				ja=[jdel_a[j] for j in range(len(jdel_a)) if (idel_a[j]==i)]
				spokesdata[i,min(ja):max(ja)]=0 # need to change to non-spokes (revise)
				
		else:
			spokecount=spokecount+1
			
	return spokesdata		
# this function sorts spoke numbers so any empty id number will be eliminated (for example spoke 5 existed but got cleaned out)
def sortspk(spokesdata,spknum_range):
	# spknum_range: spoke id range (need to be sorted)
	idint=0
	spkrealr_end=2*bound
	for i in spknum_range:
		wherei,wherej=np.where(spokesdata==i)
		if len(wherei)!=0:
			spkrealr_end=spkrealr_end+1
			spokesdata[wherei,wherej]=spkrealr_end
	return range(2*bound+1,spkrealr_end+1),spokesdata
	
import collections
# this function cleans the spokes on the edge
def cleanedge_spk(spokesdata,spknum_range,edgeper):
	# spknum_range: spoke id range
	# edgeper: 0.1 means the edge is at 10% the total pixel, get rid of any spokes that are mostly in that reagion 
	m,n=spokesdata.shape
	edgepix=n*edgeper
	for i in spknum_range:
		wherei,wherej=np.where(spokesdata==i)
		counter=collections.Counter(wherei)
		# get rid of right edges, seems to be the only ones that are causing problems
		#print(sum([j>edgepix for j in wherej]))
		#print(len(wherej))
		#print(np.var(counter.values()))
		maxj=max(wherej,default=0)
		#print(len(counterj))
		if sum([j>edgepix for j in wherej])>0.8*len(wherej) and (maxj==n-2):
			counterj=len(np.where(wherej==n-2)[0])
			if ((np.var(list(counter.values()))<1200) or (counterj)>0.5*m):
				spokesdata[wherei,wherej]=0
				sizbox=len(np.unique(wherei))*len(np.unique(wherej)) # size of box
				#print((float(sizbox)-float(len(wherei)))/float(sizbox))
# 				print('variance',np.var(list(counter.values())))
	spknum_range,spokesdata=sortspk(spokesdata,spknum_range)
	#print(spknum_range)
	return spknum_range,spokesdata
		
from scipy import signal
def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()

def blur_image(im, n, ny=None) :
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    g = gauss_kern(n, sizey=ny)
    improc = signal.convolve(im,g, mode='valid')
    return(improc)			
				
	
	
import numpy as np
from scipy.signal import butter, lfilter, freqz

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
   
# low pass filter for image
def reduce_highFn(datapoints,cutoff=0.05):
	order = 1
	fs = 1.0       # sample rate, Hz
	m,n=datapoints.shape
	artdat=90
	for i in range(m):
		findat=butter_lowpass_filter(np.append(np.ones(artdat)*datapoints[i,0],datapoints[i,:]), cutoff, fs, order)
		datapoints[i,:] = findat[artdat:len(findat)]
	for j in range(n):
		findat=butter_lowpass_filter(np.append(np.ones(artdat)*datapoints[0,j],datapoints[:,j]), cutoff, fs, order)
		datapoints[:,j] = findat[artdat:len(findat)]
	return datapoints



