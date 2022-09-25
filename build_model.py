
import tensorflow as tf
import segmentation_models as sm
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D
from keras.models import Model
import math
from pathlib import Path
import re

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)
sm.set_framework('tf.keras')


SIZE_X = 1504
SIZE_Y = 224

# capture training ifo as a list
train_images = []
train_masks = []

file_pattern = re.compile(r'.*?(\d+).*?')
def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])


for img_path in sorted(glob.glob("training/images/*.png"), key=get_order):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (SIZE_X, SIZE_Y))
    train_images.append(img)

for mask_path in sorted(glob.glob("training\masks\*.png"), key=get_order):
    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, (SIZE_X, SIZE_Y))
    train_masks.append(mask)


train_images = np.array(train_images)
train_masks = np.array(train_masks)


X = train_images
Y = train_masks

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X,Y, test_size = 0.2, random_state = 42)

# preprocess_input
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

N = 1 #number of channels

# define model 
model = sm.Unet(backbone_name='resnet34', encoder_weights=None, input_shape=(SIZE_Y, SIZE_X, N))

# model = Model(inp, out, name=base_model.name)
model.compile(optimizer =  'adam', loss = "binary_crossentropy", metrics = [sm.metrics.iou_score], )
#print(model.summary())

# fit model
history = model.fit(
   x=x_train,
   y=y_train,
   batch_size=5,
   epochs=100,
   verbose=1,
   validation_data=(x_val, y_val),
)


model.save('model/30_im-100b.h5')


# #########################################################
# from tensorflow import keras
# model = keras.models.load_model('../model/model.h5', compile=False)

# test_img = cv2.imread('../data/training/images/image1.png', cv2.IMREAD_GRAYSCALE)
# test_img = cv2.resize(test_img, (SIZE_Y, SIZE_X))
# test_img = np.expand_dims(test_img, axis=0)

# prediction = model.predict(test_img)

# #View and Save segmented image
# prediction_image = prediction.reshape(mask.shape)
# plt.imshow(prediction_image, cmap='gray')
# plt.imsave('test0_segmented.png', prediction_image, cmap='gray')
# #