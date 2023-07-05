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



def display_results(results_path):
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



def data_gather(X, Y, training_path = "../training/"):

    training_path = "../datasets/"    

    for img_path in sorted(glob.glob(training_path+"light_spokes_training_images/*.png"), key=get_order):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        X.append(img)

    for mask_path in sorted(glob.glob(training_path+"light_spokes_training_masks/*.png"), key=get_order):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask != 0] = 255
        Y.append(mask)

    print(len(X), len(Y))
    return X, Y



def fit_model(x_train, y_train, model, batch_size = 10,epochs = 300, validation_split = .15 ):
    #print(model.summary())
    
    # fit model
    history = model.fit(
       x = x_train,
       y = y_train,
       batch_size = batch_size,
       epochs = epochs,
       verbose = 1,
       validation_split = validation_split,
    )

    return history



def define_model(SIZE_X, SIZE_Y, backbone = "resnet34"):   

    model = sm.Unet(backbone_name=backbone, encoder_weights = None, input_shape=(SIZE_Y*SIZE_X, 1))
    # model = Model(inp, out, name=base_model.name)
    model.compile(optimizer = "Adam" , loss = "binary_crossentropy", metrics = [sm.metrics.IOUScore()], )
    return model



def save_model(model_path, model, history, results):
    model.save(model_path, save_format="h5")
    print(model_path)

    model_path_no_ext = model_path.split(".")[0]

    dump_dict = history.history
    dump_dict['eval_results'] = results

    with open(f"{model_path_no_ext}.json", 'w') as f:
        json.dump(dump_dict, f)
    f.close()



def model_testing(model, testing_folder, num_of_images):
    testing_folder = testing_folder+"_spokes_no_training/"
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
        print(filename, filenames.index(f"{filename}.png"))                                         
        plt.imshow(img, cmap="gray")
        plt.show()

        img = img.reshape((1, 160, 736))    
        prediction = model.predict(img)
        prediction = prediction.reshape((160, 736))

        plt.imshow(prediction, cmap='gray')
        plt.show()
    
    plt.close()
    return

