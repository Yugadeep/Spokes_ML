#!/usr/bin/env -S python3

import argparse
import sys
import numpy as np
import os
import model_utility_rpjb
import preprocess_filter
import spoketools

os.environ["SM_FRAMEWORK"] = "tf.keras"

def apply_model(model, filtered_image):

    filtered_image = filtered_image.reshape((1, 160, 736))
    prediction = model.predict(filtered_image)
    prediction = prediction.reshape((160, 736))

    return prediction


def main():
    parser = argparse.ArgumentParser(description="%(prog)s applies a model to a given rpjb file and saves a numpy to a folder")
	
    parser.add_argument("model_path", help = "Absolute path to .h5 file of machine learning model")
    parser.add_argument("rpjb_file", help = "Absolute path to the input data file")
    parser.add_argument('save_folder_path', help="plot the cropped, unfiltered image")

    if len(sys.argv) < 2:
        print(f"{len(sys.argv)-1} parameters passed. At least 2 required. An rpjb file and the variation of filters you want to apply.\n")
        parser.print_help()
        sys.exit(1)
	

    args = parser.parse_args()

    model_path = args.model_path
    rpjb_file = args.rpjb_file
    numpy_filename = rpjb_file.split("/")[-1].split(".")[0]
    save_folder_path = args.save_folder_path



    print(f"Loading model {model_path.split('/')[-1]} ...")
    model = model_utility_rpjb.load_model(model_path)
    print("model loaded")

    print(f"applying model to {rpjb_file.split('/')[-1]} ...")
    processed_rpjb = preprocess_filter.apply_filters(rpjb_file)
    prediction = apply_model(model, processed_rpjb)
    print("applyied")

    print("saving prediction ...")
    np.savetxt(f"{save_folder_path}/{numpy_filename}.np", prediction, delimiter = ',')
    print("saved")

    print("done")


if __name__ == "__main__":
 	main()