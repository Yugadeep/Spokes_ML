import albumentations as alb
import glob
from skimage import io
import re
from pathlib import Path
import matplotlib.pyplot as plt
from os.path import exists
import math
import numpy as np




#used for sorting files
file_pattern = re.compile(r'.*?(\d+).*?')
def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

#Albumentation function:
aug = alb.Compose([
    alb.ShiftScaleRotate(scale_limit=0, rotate_limit=0, shift_limit_x = 0, p=1)], 
    additional_targets={'image' : 'mask'}
)

def augment_semantic_set(X, Y, aug_num = 2):
    folder_path = "data/training/dark/augmentation/"

    for image, mask in zip(X,Y):
            for i in range(0, aug_num):
                augs = aug(image = image, mask = mask)
                X = np.append(X, [augs['image']], axis=0)
                Y = np.append(Y, [augs['mask']], axis=0)

    return X,Y
    



	