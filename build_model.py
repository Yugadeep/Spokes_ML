#Model creation
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D
from keras.models import Model
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


BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)
sm.set_framework('tf.keras')
training_size = 0



def data_gather(SIZE_X, SIZE_Y, training_path = "../training/"):

    # capture training ifo as a list
    train_images = []
    train_masks = []

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
        train_images.append(img)

    for mask_path in sorted(glob.glob(training_path+"masks/*.png"), key=get_order):
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (SIZE_X, SIZE_Y))
        train_masks.append(mask)


    train_images = np.array(train_images)
    train_masks = np.array(train_masks)

    training_size = len(train_images)
    X = train_images
    Y = train_masks
    return X,Y

def train_model(X, Y, SIZE_X, SIZE_Y, epoch_num, model_path, batch_size = 5):
    #splits X and Y into training and validation sets
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
       batch_size=batch_size,
       epochs=epoch_num,
       verbose=1,
       validation_data=(x_val, y_val),
    )


    model.save(f"{model_path}+{training_size}_im-{epoch_size}.h5")



def test_model(model_path, testim_path, SIZE_X, SIZE_Y, epoch_size, training_size): 
    model = keras.models.load_model(f"{model_path}{training_size}im_{epoch_size}b.h5", compile=False)

    test_img = cv2.imread(testim_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.resize(test_img, (SIZE_X, SIZE_Y))
    test_img = np.expand_dims(test_img, axis=0)

    prediction = model.predict(test_img)

    #View and Save segmented image
    prediction_image = prediction.reshape((test_img.shape[1], test_img.shape[2]))
    filename = testim_path.split('/')[3]
    plt.imshow(prediction_image, cmap='gray')
    plt.imsave(f'../training/validation/val_{filename}.png', prediction_image, cmap='gray')



#########################################################
#Main
SIZE_X = 1504
SIZE_Y =  224
X, Y = data_gather(SIZE_X, SIZE_Y, training_path = "../training/")
print(X.shape)
print(Y.shape)


#train_model(X, Y, SIZE_X, SIZE_Y, 10, model_path = "../model/", batch_size = 5)
for testim_path in glob.glob("../training/validation/*.png"):
    print(testim_path)
    test_model('../model/', testim_path, SIZE_X, SIZE_Y, 100, 30)