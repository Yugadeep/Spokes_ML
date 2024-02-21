#!/usr/bin/env -S python3

import argparse
import sys
import numpy as np

import model_utility_rpjb
import preprocess_filter
import spoketools

class Formatter(argparse.RawDescriptionHelpFormatter):
	pass

def apply_model(model, filtered_image):

    filtered_image = filtered_image.reshape((1, 160, 736))    
    prediction = model.predict(filtered_image)
    prediction = prediction.reshape((160, 736))

    return prediction




def main():
    parser = argparse.ArgumentParser(description="%(prog)s applies a model to a given rpjb file and saves a numpy to a folder")
	
    parser.add_argument("model_path", help = "Absolute path to .h5 file of machine learning model")
    parser.add_argument("rpjb_list", help = "Absolute path to list of input rpjb files")
    parser.add_argument('save_folder_path', help="plot the cropped, unfiltered image")

    if len(sys.argv) < 2:
        print(f"{len(sys.argv)-1} parameters passed. At least 3 required.  A valid file list is needed if you want to apply.\n")
        parser.print_help()
        sys.exit(1)
	

    args = parser.parse_args()

    model_path = args.model_path
    rpjb_list = args.rpjb_list
    save_folder_path = args.save_folder_path

    print(f"Loading model {model_path.split('/')[-1]} ...")
    model = model_utility_rpjb.load_model(model_path)
    print("model loaded")

    print("applyin model to rpjb list ...")
    f = open(rpjb_list)
    text = f.read()
    rpjb_list  = text.split("\n")
    for rpjb_path in rpjb_list:
        filename = rpjb_path.split('/')[-1].split(".")[0].split("_")[0]
        filtered_image = preprocess_filter.apply_filters(rpjb_path)
        prediction = apply_model(model, filtered_image)
        np.savetxt(f"{save_folder_path}/{filename}.np", prediction, delimiter= ',')

    print("done")



if __name__ == "__main__":
 	main()