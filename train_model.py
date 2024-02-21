#!/usr/bin/env -S python3

# built-in imports
import argparse 
import sys
import glob
import re
import os
import numpy as np


# imports from installed packages
import tensorflow as tf
from keras.utils import normalize
from sklearn.model_selection import train_test_split
import segmentation_models as sm
os.environ["SM_FRAMEWORK"] = "tf.keras"

# imports from code written for project
import model_utility_rpjb
import preprocess_filter
import spoketools





class Formatter(argparse.RawDescriptionHelpFormatter): 
	pass


# helper functions

def main():

    parser = argparse.ArgumentParser(description="%(prog)s creates a model based on a list of rpjb's to train the model on") 
    parser.add_argument("rpjb_list_path", help = "filepath to list of rpjb files")
    parser.add_argument("mask_list_path", help = "filepath to list of mask files")
    parser.add_argument("epoch_num", help = "the number of epochs to train the model for")
    parser.add_argument("model_path", help = "filepath to where you would like the model to be saved to. Must save the File as a .h5 file")


    if len(sys.argv) < 4:
        print(f"{len(sys.argv)-1} parameters passed. At least 4 required. \nPlease make sure to include a path to the rpjbs, the masks, and to where you want to save the model.")
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    rpjb_list_path = args.rpjb_list_path
    mask_list_path = args.mask_list_path
    epoch_num =  args.epoch_num 
    model_path = args.model_path

    X, Y = [], []
    X, Y = model_utility_rpjb.data_gather(X, Y, rpjb_list_path, mask_list_path)

    X = normalize(np.array(X))
    Y = np.array(Y)
    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.15, random_state = 42)
    SIZE_Y, SIZE_X = X.shape[1], X.shape[2]

    model = model_utility_rpjb.define_model(SIZE_Y, SIZE_X, "resnet34")
    history = model_utility_rpjb.fit_model(x_train, y_train, model, model_path, batch_size = 10, epochs = int(epoch_num), validation_split = .13)
    results = model.evaluate(x_test, y_test)
    model_utility_rpjb.save_model_history(model_path, model, history, results)


    return


if __name__ == "__main__":
	 	main()



