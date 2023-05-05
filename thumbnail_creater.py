import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import scipy.io as io
import copy



def save_image(new_path, pixel_values):
	plt.figure()
	plt.axis('off')
	fig = plt.imshow(pixel_values,cmap = plt.get_cmap('gray'), origin="lower")
	plt.savefig(new_path,bbox_inches='tight',transparent=True, pad_inches=0, dpi=300)
	plt.close()

rpj_new = "/Users/willbyrne/Documents/work/code/hamilton/unet_spokes/data/2023_rpjb/all/"
thumb = "/Users/willbyrne/Documents/work/code/hamilton/unet_spokes/data/2023_imagery/thumbnails/all/"
folders = [folder+"/" for folder in glob.glob(rpj_new+"*")]
folder = ""

for filepath in sorted(glob.glob(rpj_new+"**/*.rpjb")):
    split = filepath.split("/")
    if split[-2] != folder:
        print(f"startng {split[-2]}")
    folder = split[-2]
    filename = filepath.split('/')[-1].split('_')[0]
    
    if not os.path.exists(thumb+folder+"/"):
        os.mkdir(thumb+folder+"/")

    idl = io.readsav((filepath))
    pixel_values = idl.rrpi
    pixel_values=copy.copy(pixel_values)

    new_thumb_path = thumb+folder+"/"+filename+".png"
    save_image(new_thumb_path, pixel_values)




    
