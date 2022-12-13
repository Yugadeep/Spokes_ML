import albumentations as alb
import glob
from skimage import io
import re
from pathlib import Path
import matplotlib.pyplot as plt
from os.path import exists



#input - training folder. Ex: ..../data/training
training_folder = "/Users/willbyrne/Documents/CODE/Hamilton/Unet_Spokes/data/training"
aug_num = 10




#used for sorting files
file_pattern = re.compile(r'.*?(\d+).*?')
def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

#Albumentation function:
aug = alb.Compose([
    alb.VerticalFlip(),
    alb.HorizontalFlip(),
    alb.PixelDropout(),
    alb.GridDistortion()], 
    additional_targets={'image' : 'mask'}
)


img_folder = f"{training_folder}/images"
mask_folder = f"{training_folder}/masks" 

img_paths = glob.glob(f"{img_folder}/*.png")
mask_paths = glob.glob(f"{mask_folder}/*.png")
for img_path, mask_path in zip(sorted(img_paths, key = get_order),sorted(mask_paths, key = get_order)):

    img_name = img_path.split("/")[-1]
    mask_name = mask_path.split("/")[-1]

    image = io.imread(img_path)
    mask = io.imread(mask_path)
    if "aug" not in img_name: 
        print(f'Augementing {img_name} {aug_num} times ...')

        for i in range(0, aug_num):
            augs = aug(image = image, mask = mask)
            if not exists(f"{img_folder}/aug{i}_{img_name}"):
                io.imsave(f"{img_folder}/aug{i}_{img_name}", augs['image'])
                io.imsave(f"{mask_folder}/aug{i}_{mask_name}",augs['mask'])
            else:
                print(f"aug{i}_{img_name} and mask Already exists")



	