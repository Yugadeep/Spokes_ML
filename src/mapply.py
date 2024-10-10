#!/usr/bin/env -S python3


# Model stuff
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"


import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D
from keras.models import Model
from keras.utils import normalize
import segmentation_models as sm
from sklearn.model_selection import train_test_split
import scipy.io as io


#our scripts
import model_utility
import spoketools
import preprocess_filter

#path sorting
import glob
import cv2
from pathlib import Path
import re
import sys

#math
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd


model_path = sys.argv[1]   # path to the ML model you want to use
model = keras.models.load_model(model_path, compile = False)

filename = sys.argv[2]  # apply the ML model to this file. Pass the path here.


datapoints = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


target_size = (736, 160)
resized_data = cv2.resize(datapoints, target_size)

final = normalize(resized_data, axis = 1 )

final = final.reshape(1, 160, 736, 1)  # Shape: batch dimension, height, width, number of color channels (gray = 1 )

prediction =  model.predict(final)
prediction = prediction.reshape((160, 736))

plt.figure(figsize = (15,7))
plt.imshow(prediction, cmap = "gray")
plt.show()






