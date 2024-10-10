#!/usr/bin/env -S python3

print("Usage: makem.py training_thumbnails_path masks_path model_savepath #epochs ")
import os
import sys
os.environ["SM_FRAMEWORK"] = "tf.keras"

# Model stuff
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D
from keras.models import Model
from keras.utils import normalize
import segmentation_models as sm
from sklearn.model_selection import train_test_split

#path sorting
import glob
import cv2
from pathlib import Path
import re

#math
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd


# in differnt folder
#sys.path.append('../src/')
import model_utility
import preprocess_filter as pf   # YK edit
from preprocess_filter import apply_filters  # Ensure this is uncommented

### That's it for importing libraries

backbone = "resnet34"

trainingset_path = f"{sys.argv[1]}"
masks_path = f"{sys.argv[2]}"
model_savepath = f"{sys.argv[3]}" + ".keras"
epoch_num = f"{sys.argv[4]}"

X, Y = [], []


X, Y = model_utility.data_gather_YK(X, Y, trainingset_path, masks_path, aug_flag = 1, aug_num = 5)

if X.shape != Y.shape:
    print("Error. Check shape")

X = normalize(np.array(X), axis=1)
Y = (np.array(Y))/255.

# train/test split test_size = .15 for light, .25 for dark(no agu)
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.15, random_state = 42)

SIZE_Y, SIZE_X = X.shape[1], X.shape[2]
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


model = sm.Unet(backbone_name="resnet34", encoder_weights = None, input_shape=(SIZE_Y,SIZE_X, 1))

#model.compile(optimizer = "Adam" , loss = "binary_crossentropy", metrics=['accuracy'] )
#model.compile(optimizer = "Adam" , loss = "binary_crossentropy", metrics = [sm.metrics.IOUScore()], )

model.compile(optimizer = "Adam" , loss = "binary_crossentropy", metrics=['BinaryIoU'] )

print(model.summary())



model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_savepath,
    monitor='BinaryIoU',
    mode='max',
    save_best_only=True, 
    verbose = True)


fit_model = model.fit(x_train, y_train, batch_size= 10, epochs = int(epoch_num) 
                      , verbose=1,  validation_split = .13 , callbacks = [model_checkpoint_callback])

model.save(str(model_savepath))















