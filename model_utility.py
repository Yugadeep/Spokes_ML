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

def data_gather(SIZE_X, SIZE_Y, training_path = "../training/"):

    # capture training ifo as a list
    X_train = []
    Y_train = []
    
    X_test = []
    Y_test = []
    
    
    #regex pattern I copied to sort filepaths in ascending order
    file_pattern = re.compile(r'.*?(\d+).*?')
    def get_order(file):
        match = file_pattern.match(Path(file).name)
        if not match:
            return math.inf
        return int(match.groups()[0])
    
    
    
    for img_path in sorted(glob.glob(training_path+"images/*.png"), key=get_order):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (SIZE_X, SIZE_Y))
        X_train.append(img)
        
        
    for mask_path in sorted(glob.glob(training_path+"masks/*.png"), key=get_order):
        mask = cv2.imread(mask_path, -1)
        mask = cv2.resize(mask, (SIZE_X, SIZE_Y))
        Y_train.append(mask)
    
    for test_img_path in glob.glob("/datasets/spoke_test_set/W*.png"):
        img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (SIZE_X, SIZE_Y))
        X_test.append(img)
    
    
    for test_mask_path in glob.glob("/datasets/spoke_test_set/mask*.png"):
        mask = cv2.imread(test_mask_path, -1)
        mask = cv2.resize(mask, (SIZE_X, SIZE_Y))
        Y_test.append(mask)
    
    
    X_train = normalize(np.array(X_train), axis=1)
    Y_train = (np.array(Y_train))/255.
    
    X_test = normalize(np.array(X_test), axis=1)
    Y_test = (np.array(Y_test))/255.
    
    return X_train,Y_train, X_test, Y_test


def train_model(model_path, model_params):
    SIZE_X = model_params['SIZE_X']
    SIZE_Y = model_params['SIZE_Y']
    backbone = model_params['backbone']
    #splits X and Y into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(model_params["X_train"],model_params["Y_train"], test_size = 0.17, random_state = 42)
    
    
    
    
    model = sm.Unet(backbone_name=backbone, encoder_weights = None, input_shape=(SIZE_Y,SIZE_X, 1))
    # model = Model(inp, out, name=base_model.name)
    model.compile(optimizer = "Adam" , loss = "binary_crossentropy", metrics = [sm.metrics.IOUScore()], )
    #print(model.summary())
    
    # fit model
    history = model.fit(
       x=x_train,
       y=y_train,
       batch_size= model_params['batches'],
       epochs= model_params['epoch_num'],
       verbose=1,
       validation_data=(x_val, y_val),
    )

    results = model.evaluate(model_params['X_test'], model_params['Y_test'])
    model.save(f"{model_path}spoke_{model_params['training_size']}im_{model_params['epoch_num']}e_{backbone}.h5")
    
    return results


def define_model(backbone = "vgg16"):   
    SIZE_X = 1504
    SIZE_Y =  224
    model = sm.Unet(backbone_name=backbone, encoder_weights = None, input_shape=(SIZE_Y*SIZE_X, 1))
    # model = Model(inp, out, name=base_model.name)
    model.compile(optimizer = "Adam" , loss = "binary_crossentropy", metrics = [sm.metrics.IOUScore()], )
    return model