import matplotlib.pyplot as plt
import numpy as np
import glob
import scipy.io as io
import copy

def remove_cosmic_rays(pixel_values):
    m,n=pixel_values.shape

    for i in range(0, m):
        top20 = np.sort(pixel_values[i])[-21:]
        for j in np.argsort(pixel_values[i])[-21:]:
            pixel_values[i,j] = top20[0]        

    return pixel_values

path_list = ['data/2023_rpjb/good/085_SPKMVLFLP/W1600549826_1_CALIB.rpjb',
 'data/2023_rpjb/good/085_SPKMVLFLP/W1600551910_1_CALIB.rpjb',
 'data/2023_rpjb/good/085_SPKMVLFLP/W1600547742_1_CALIB.rpjb']

for thumb_cosmic_ray in path_list:
    idl = io.readsav(thumb_cosmic_ray)
    pixel_values = idl.rrpi
    pixel_values=copy.copy(pixel_values)

    plt.imshow(pixel_values, cmap='gray', origin='lower')
    plt.show()
    
    pixel_values = remove_cosmic_rays(pixel_values)
    
    plt.imshow(pixel_values, cmap='gray', origin='lower')
    plt.show()