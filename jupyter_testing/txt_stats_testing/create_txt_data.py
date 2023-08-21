import pandas as pd
import re
import os
import glob
import numpy as np

def create_perSpoke_data(txt_path):
    path_split = txt_path.split("/")
    folder_path = "/".join(path_split[:-1])
    names = ["max_rad", "min_rad", "max_lon", "min_lon", "mean_intensity", "pixel_count"]


    filename = path_split[-1].split('_')[0]
    test_df = pd.read_csv(txt_path, delimiter="	")
    group_by = test_df.groupby("Spoke Number")

    min_rad = group_by["# Rad"].min()
    max_rad = group_by["# Rad"].max()
    min_lon = group_by["Long"].min()
    max_lon = group_by["Long"].max()
    mean_intensity =group_by["Intensity"].mean()
    count = group_by["Intensity"].count()

    per_spoke_df = pd.concat([min_rad, max_rad, min_lon, max_lon, mean_intensity, count],axis=1)
    per_spoke_df.columns = names
    per_spoke_df = per_spoke_df.rename_axis(index = "spoke_num")
    per_spoke_df.to_csv(f"{folder_path}/{filename}_spoke_ML.txt")

    
    print(f"{folder_path}/{filename}_spoke_ML.txt")
    return os.path.exists(f"{folder_path}/{filename}_spoke_ML.txt")

# How to use create_perSpoke_data
txt_list = glob.glob("../data/2023_imagery/filtered/**/*CALIB_ML.txt")

for txt_path in txt_list:
    if not create_perSpoke_data(txt_path):
        break



def bypixel_summary(txt_path):
    in_df = pd.read_csv(txt_path, delimiter = "	")
    filename = os.path.basename(txt_path)

    # what to do if the a image has no spokes in it
    if len(in_df) == 0:
        print('this file is empty')
        df_dict = {"filename": filename, "min_rad": [np.nan], "max_rad": [np.nan], "min_lon": [np.nan], "max_lon": [np.nan], "num_spokes": 0, "num_pix":0, "avg_pix_per_spoke":0}

        empty_df = pd.DataFrame(df_dict)
        empty_df = empty_df.set_index('filename')
        return empty_df

    

    min_rad = round(in_df['# Rad'].min(), 3)
    max_rad = round(in_df['# Rad'].max(), 3)
    min_lon = round(in_df['Long'].min(), 3)
    max_lon = round(in_df['Long'].max(), 3)

    num_spokes = in_df['Spoke Number'].unique()[-1]
    num_pix = len(in_df)
    avg_pix_per_spoke = round(in_df.groupby('Spoke Number')['Intensity'].count().mean(), 3)

    df_dict = {"filename": filename, "min_rad": min_rad, "max_rad": max_rad, "min_lon": min_lon, "max_lon": max_lon, "num_spokes": num_spokes, "num_pix":num_pix, "avg_pix_per_spoke":avg_pix_per_spoke}    
    full_df = pd.DataFrame(df_dict, index = [0])
    full_df = full_df.set_index("filename")

    return full_df

# how to use 
px_txt = glob.glob("../../../data/2023_imagery/filtered/074_SPKLFMOV/*CALIB_ML.txt")
frame_list = []

for path in px_txt:
    frame_list.append(bypixel_summary(path))
total_list = pd.concat(frame_list, axis = 0)

total_list.to_csv("path/.../filename")

