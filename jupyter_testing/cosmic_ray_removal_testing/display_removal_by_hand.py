import matplotlib.pyplot as plt
import numpy as np
import glob
import scipy.io as io
import copy
import preprocess_filter
import spoketools


def opusid_to_rpjb_path(opus_id):
    rpjb_path = None
    # print(glob.glob("hamilton/unet_spokes/data/2023_rpjb/good/*/"))
    temp_list = glob.glob(f"data/2023_rpjb/good/*/{opus_id}*.rpjb")
    if len(temp_list) == 1:
        rpjb_path = temp_list[0]
    elif len(temp_list) == 0:
        print("path doesn't exist")
    else:
        print("there were multiple!")

    return rpjb_path


plots = {'raw': False, 'cosmic_ray': False, 'outside_zero' : False, 'quant' : False, 'cropped' : False, 'lucy_median': False, 'forrier':False}

plots['raw'] = True
plots['cosmic_ray'] = True
# plots['cropped'] = True
plots['lucy_median'] = True
# plots['forrier'] = True


# odd circles being made right after the median filter - W1766581755
# What is that dark bit on the left side? - W1768358295
# looks like both of the above features persit through all of their respective folders

individual = []
individual.append("W1630677951")


for opus_id in individual:
    rpjb_path = opusid_to_rpjb_path(opus_id)
    print(rpjb_path)

    filename, pixel_values = preprocess_filter.apply_filters(rpjb_path, plots = plots)
    pixel_values = spoketools.fft2lpf(pixel_values, 0, 3)
    plt.imshow(pixel_values, cmap = "gray", origin = "lower")
    plt.show()
    pixel_values = preprocess_filter.buffer_image(pixel_values, 736, 160)

    plt.imshow(pixel_values, cmap = "gray", origin = "lower")
    plt.show()