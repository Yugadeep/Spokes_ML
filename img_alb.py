#Spoke image albumentation code
#Created by Morgan Craver on 09/12/2022
#Based on code by Sreenivas Bhattiprolu

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage import io, img_as_ubyte
import random
import os
from scipy.ndimage import rotate
import albumentations as A
import math
from pathlib import Path
import re
import glob
#plugin='png'

images_to_generate = 30
image_paths = []
mask_paths = []

file_pattern = re.compile(r'.*?(\d+).*?')
def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

for img_path in sorted(glob.glob("../data/training/images/*.png"), key=get_order):        
    image_paths.append(img_path)

for msk_path in sorted(glob.glob("../data/training/masks/*.png"), key=get_order):    
    mask_paths.append(msk_path)


aug = A.Compose([
    A.VerticalFlip(p=0.5),              
    A.HorizontalFlip(p=0.5),
    A.GridDistortion(),
    A.RandomGamma(),
    A.RandomBrightnessContrast()
    ]
)



for img_path,mask_path in zip(image_paths, mask_paths):
    print(f"augmenting {img_path}")
    for i in range(1, images_to_generate+1):
        img_name = img_path.split("/")[-1][:-4]
        mask_name = mask_path.split("/")[-1][:-4]

        original_image = io.imread(img_path)
        original_mask = io.imread(mask_path)
        augmented = aug(image = original_image, mask = original_mask)
        transformed_image = augmented['image']
        transformed_mask = augmented['mask']
                                                            
        new_image_path = f"../data/training/images/{img_name}_aug{i}.png"
        new_mask_path = f"../data/training/masks/{mask_name}_aug{i}.png"
        io.imsave(new_image_path, transformed_image)
        io.imsave(new_mask_path, transformed_mask)
        txt = "Processing augment {}..."
        print(txt.format(i))
    

print("Done")

                                                                                                
