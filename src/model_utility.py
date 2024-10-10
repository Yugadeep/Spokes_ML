import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D
from keras.models import Model
from keras.utils import normalize
import segmentation_models as sm
from sklearn.model_selection import train_test_split
import augment_trainingset



#path sorting
import glob
import cv2
from pathlib import Path
import re
import json

#math
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

file_pattern = re.compile(r'.*?(\d+).*?')
def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])



def display_results(results_path, ax = None):
    if ax == None:
        ax = plt.gca()

    model_path_no_ext = results_path.split(".")[0]
    results_path =  model_path_no_ext+".json"
    
    
    with open(results_path) as json_file:
        results = json.load(json_file)

    print("Which model is this? -",results_path.split("/")[-1])
    type = results_path.split("/")[-1].split("_")[0]

    iou_score = results['iou_score']
    val_iou_score = results['val_iou_score']
    loss = results['loss']
    val_loss = results['val_loss']

    epochs = range(1, len(iou_score) + 1)

    ax.plot(epochs, iou_score, 'bo', label='Training acc')
    ax.plot(epochs, val_iou_score, 'b', label='Validation acc')
    ax.legend()

    # plt.figure()
    # plt.yscale("log")
    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title(f'{type} Spoke Training and validation loss')
    # plt.legend()

    # plt.show()
    print("Last Train IOU Score: ",results['iou_score'][-1])
    print("Last Train Loss Score: ", results['loss'][-1])
    print("Last Validation IOU Score: ", results['val_iou_score'][-1])
    print("Last Validation Loss Score: ", results['val_loss'][-1])
    json_file.close()

import os

def data_gather_YK(X, Y, image_dir, mask_dir, aug_flag=0, aug_num=0):   # YK version

    # List all PNG files in the image and mask directories
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]

    # Sort files based on their numeric part in the filename
    def extract_number(filename):
        return int(''.join(filter(str.isdigit, filename)))
    
    image_files.sort(key=extract_number)
    mask_files.sort(key=extract_number)

    # Load images and masks
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        X.append(img)

    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask != 0] = 255
        Y.append(mask)

    # Apply augmentation if flag is set
    if aug_flag == 1:
        X, Y = augment_trainingset.augment_semantic_set(X, Y, aug_num=aug_num)

    print(len(X), len(Y))
    
    return X, Y


def data_gather(X, Y, image_type="light_spokes_training_images", mask_type="light_spokes_training_masks", training_path = "../training/", aug_flag = 0, aug_num = 0):

    training_path = "../datasets/"


    #regex pattern I copied to sort filepaths in ascending order
    file_pattern = re.compile(r'.*?(\d+).*?')
    def get_order(file):
        match = file_pattern.match(Path(file).name)
        if not match:
            return math.inf
        return int(match.groups()[0])
    

    for img_path in sorted(glob.glob(training_path+image_type+"/*.png"), key=get_order):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        X.append(img)

    for mask_path in sorted(glob.glob(training_path+mask_type+"/*.png"), key=get_order):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask != 0] = 255
        Y.append(mask)

    if aug_flag == 1:
        X, Y = augment_trainingset.augment_semantic_set(X, Y, aug_num = aug_num)
    print(len(X), len(Y))    

    
    
    return X, Y

def define_model(SIZE_Y, SIZE_X, backbone = "resnet34"):   

    model = sm.Unet(backbone_name="resnet34", encoder_weights = None, input_shape=(SIZE_Y,SIZE_X, 1))
    model.compile(optimizer = "Adam" , loss = "binary_crossentropy", metrics = [sm.metrics.IOUScore()], )
    print(model.summary())
    return model

def define_model_YK(SIZE_Y, SIZE_X, backbone="resnet34"):
    # Apply padding or adjust SIZE_Y and SIZE_X directly if needed
    padded_input_shape = (SIZE_Y + 1, SIZE_X + 1, 1)  # Adjust if you apply padding

    # Define the model
    model = sm.Unet(
        backbone_name=backbone,
        encoder_weights=None,
        input_shape=padded_input_shape
    )
    
    model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=[sm.metrics.IOUScore()])
    print(model.summary())
    return model



def fit_model(x_train, y_train, model, model_path,batch_size = 10,epochs = 300, validation_split = .15 ):
    #print(model.summary())
    
    # fit model
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    monitor='val_iou_score',
    mode='max',
    save_best_only=True, 
    verbose = True)
    
    history = model.fit(
       x = x_train,
       y = y_train,
       batch_size = batch_size,
       epochs = epochs,
       verbose = 1,
       validation_split = validation_split,
       shuffle = True,
       callbacks = [model_checkpoint_callback]
    )

    return history


def fit_model_YK(x_train, y_train, model, model_path, batch_size=10, epochs=300, validation_split=.15):   # YK edit
    print("Before fitting:")
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)

    # fit model
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    monitor='val_iou_score',
    mode='max',
    save_best_only=True, 
    verbose = True)
    
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=validation_split,
        shuffle=True,
        callbacks=[model_checkpoint_callback]
    )

    print("After fitting:")
    print("Model output shape:", model.output_shape)

    return history


import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model

class ResizeLayer(Layer):
    def __init__(self, target_size, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_size = target_size

    def call(self, inputs):
        return tf.image.resize(inputs, self.target_size, method='bilinear')

def unet_model(input_shape):
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(512, 3, activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    c5 = Conv2D(1024, 3, activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, 3, activation='relu', padding='same')(c5)
    
    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    c4_resized = ResizeLayer(target_size=tf.shape(u6)[1:3])(c4)  # Resize c4 to match u6
    u6 = concatenate([u6, c4_resized])
    c6 = Conv2D(512, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(512, 3, activation='relu', padding='same')(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    c3_resized = ResizeLayer(target_size=tf.shape(u7)[1:3])(c3)  # Resize c3 to match u7
    u7 = concatenate([u7, c3_resized])
    c7 = Conv2D(256, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(256, 3, activation='relu', padding='same')(c7)
    
    u8 = UpSampling2D((2, 2))(c7)
    c2_resized = ResizeLayer(target_size=tf.shape(u8)[1:3])(c2)  # Resize c2 to match u8
    u8 = concatenate([u8, c2_resized])
    c8 = Conv2D(128, 3, activation='relu', padding='same')(u8)
    c8 = Conv2D(128, 3, activation='relu', padding='same')(c8)
    
    u9 = UpSampling2D((2, 2))(c8)
    c1_resized = ResizeLayer(target_size=tf.shape(u9)[1:3])(c1)  # Resize c1 to match u9
    u9 = concatenate([u9, c1_resized])
    c9 = Conv2D(64, 3, activation='relu', padding='same')(u9)
    c9 = Conv2D(64, 3, activation='relu', padding='same')(c9)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model




def save_model_history(model_path, model, history, results):

    model_path_no_ext = model_path.split(".")[0]
    print(f"Which model is this:  {model_path_no_ext}")

    dump_dict = history.history
    dump_dict['eval_results'] = results

    with open(f"{model_path_no_ext}.json", 'w') as f:
        json.dump(dump_dict, f)
    f.close()



def model_testing(model, testing_folder, num_of_images):
    testing_folder = testing_folder+"/"
    remaining_dataset = sorted(glob.glob(f"../datasets/{testing_folder}*.png"), key=get_order)
    remaining_test = []
    filenames = []

    print(f"The {testing_folder} training set is made of {len(remaining_dataset)} images")

    for img_path in remaining_dataset:
        filenames.append(img_path.split("/")[-1])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        remaining_test.append(img)
    
    remaining_test = normalize(np.array(remaining_test), axis=1)

    # num_of_images problematic here
    for filename, img in zip(filenames[:num_of_images], remaining_test[:num_of_images]):
        print(filename, filenames.index(f"{filename}"))                                         
        plt.imshow(img, cmap="gray")
        plt.show()

        img = img.reshape((1, 160, 736))    
        prediction = model.predict(img)
        prediction = prediction.reshape((160, 736))

        plt.imshow(prediction, cmap='gray')
        plt.show()
    
    plt.close()
    return