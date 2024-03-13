import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D
from keras.models import Model
from keras.utils import normalize
import segmentation_models as sm
from sklearn.model_selection import train_test_split
import augment_trainingset

# Wills Code
import preprocess_filter
import spoketools

#path sorting
import glob
import cv2
from pathlib import Path
import re
import json
import sys

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

def display_results(results_path):
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

    plt.plot(epochs, iou_score, 'bo', label='Training acc')
    plt.plot(epochs, val_iou_score, 'b', label='Validation acc')
    plt.title(f'{type} Spoke Training and validation accuracy')
    plt.legend()

    plt.figure()
    plt.yscale("log")
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title(f'{type} Spoke Training and validation loss')
    plt.legend()

    plt.show()
    print("Last Train IOU Score: ",results['iou_score'][-1])
    print("Last Train Loss Score: ", results['loss'][-1])
    print("Last Validation IOU Score: ", results['val_iou_score'][-1])
    print("Last Validation Loss Score: ", results['val_loss'][-1])
    json_file.close()


# def data_gather(X, Y, image_type="light_spokes_training_images", mask_type="light_spokes_training_masks", training_path = "../training/", aug_num = 0):

#     training_path = "../datasets/"
    

#     for rpjb_filepath in sorted(glob.glob(training_path+image_type+"/*.rpjb"), key=get_order):
#             filename, pixel_values, coords = preprocess_filter.apply_filters(rpjb_filepath)
#             pixel_values = preprocess_filter.apply_lucy_median(pixel_values)
#             pixel_values = spoketools.fft2lpf(pixel_values, 0, 3)
#             pixel_values, coords = preprocess_filter.buffer_image(pixel_values, 736, 160, coords)

#             X.append(pixel_values)

#     for mask_path in sorted(glob.glob(training_path+mask_type+"/*.png"), key=get_order):
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         mask[mask != 0] = 1
#         Y.append(mask)

#     if aug_num:
#         X, Y = augment_trainingset.augment_semantic_set(X, Y, aug_num = aug_num)
#     print(len(X), len(Y))    
    
#     return X, Y

def data_gather(X, Y, rpjb_list_path, mask_list_path):
    with open(rpjb_list_path, "r") as f:
        rpjb_list = f.read()
        rpjb_list = rpjb_list.split("\n")[:-1]
    

    with open(mask_list_path, "r") as f:
        mask_list = f.read()
        mask_list = mask_list.split("\n")[:-1]


    for rpjb_filepath in rpjb_list:
            pixel_values = preprocess_filter.apply_filters(rpjb_filepath)
            X.append(pixel_values)

    for mask_path in mask_list:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask != 0] = 1
        Y.append(mask)


    return X, Y

def fit_model(x_train, y_train, model, model_path, batch_size = 10, epochs = 300, validation_split = .15 ):
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

def define_model(SIZE_Y, SIZE_X, backbone = "resnet34"):   
    model = sm.Unet(backbone_name="resnet34", encoder_weights = None, input_shape=(SIZE_Y,SIZE_X, 1))
    model.compile(optimizer = "Adam" , loss = "binary_crossentropy", metrics = [sm.metrics.IOUScore()])
    return model


def save_model_history(model_path, history, results):

    model_path_no_ext = str(Path(model_path).parent)
    dump_dict = history.history
    dump_dict['eval_results'] = results

    with open(f"{model_path_no_ext}.json", 'w') as f:
        json.dump(dump_dict, f)
    f.close()


def load_model(model_path):
    model = keras.models.load_model(model_path, compile = False)
    model.compile(optimizer = "Adam" , loss = "binary_crossentropy", metrics = [sm.metrics.IOUScore()], )
    return model

def model_testing(model, testing_folder, num_of_images):
    testing_folder = testing_folder+"/"
    remaining_dataset = sorted(glob.glob(f"../datasets/{testing_folder}*.rpjb"), key=get_order)

    print(f"The {testing_folder} training set is made of {len(remaining_dataset)} images")

    for rpjb_filepath in remaining_dataset:
        filename = rpjb_filepath.split("/")[-1]
        

        print(filename, remaining_dataset.index(rpjb_filepath))
        pixel_values = preprocess_filter.apply_filters(rpjb_filepath)


        plt.imshow(pixel_values, cmap="gray")
        plt.show()

        img = img.reshape((1, 160, 736))    
        prediction = model.predict(img)
        prediction = prediction.reshape((160, 736))

        plt.imshow(prediction, cmap='gray')
        plt.show()
    
    plt.close()
    return