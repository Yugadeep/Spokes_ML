import albumentations as alb
import glob
from skimage import io
import re
from pathlib import Path
import matplotlib.pyplot as plt



#input
img_folder = "../data/training/images"
mask_folder = "../data/training/masks"

aug_num = 2




#used for sorting files
file_pattern = re.compile(r'.*?(\d+).*?')
def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

#Albumentation function:
aug = alb.Compose([
    alb.HorizontalFlip(),    #images and masks
    alb.RandomGamma(),    #images
    alb.RandomBrightnessContrast(),  #images
    alb.GridDistortion(),    #images and masks
    alb.RingingOvershoot(),   #images
    alb.GaussNoise()],    #images  
    additional_targets={'im' : 'msk'}
)




img_paths = glob.glob(f"{img_folder}/*.png")
mask_paths = glob.glob(f"{mask_folder}/*.png")
for img_path, mask_path in zip(sorted(img_paths, key = get_order),sorted(mask_paths, key = get_order)):

    img_name = img_path.split("/")[-1]
    mask_name = mask_path.split("/")[-1]

    image = io.imread(img_path)
    mask = io.imread(mask_path)
    print(f'Augementing {img_name} {aug_num} times ...')

    for i in range(0, aug_num):
        augs = aug(image = image, mask = mask)
        io.imsave(f"{img_folder}/aug{i}_{img_name}", augs['image'])
        io.imsave(f"{mask_folder}/aug{i}_{mask_name}",augs['mask'])




	