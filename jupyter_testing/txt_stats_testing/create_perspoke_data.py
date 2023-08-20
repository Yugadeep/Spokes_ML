import pandas as pd
import re
import os
import glob

def create_perSpoke_data(txt_path):
    path_split = txt_path.split("/")
    folder_path = "/".join(path_split[:-1])
    names = ["max_rad", "min_rad", "max_lon", "min_lon", "mean_intensity", "pixel_count"]


    filename = path_split[-1].split('_')[0]
    test_df = pd.read_csv(txt_path, delimiter="	")
    group_by = test_df.groupby("Spoke Number")

    max_rad = group_by["# Rad"].max()
    min_rad = group_by["# Rad"].min()
    max_lon = group_by["Long"].max()
    min_lon = group_by["Long"].min()
    mean_intensity =group_by["Intensity"].mean()
    count = group_by["Intensity"].count()

    per_spoke_df = pd.concat([max_rad, min_rad, max_lon, min_lon, mean_intensity, count],axis=1)
    per_spoke_df.columns = names
    per_spoke_df = per_spoke_df.rename_axis(index = "spoke_num")
    per_spoke_df.to_csv(f"{folder_path}/{filename}_spoke_ML.txt")

    
    print(f"{folder_path}/{filename}_spoke_ML.txt")
    return os.path.exists(f"{folder_path}/{filename}_spoke_ML.txt")


txt_list = glob.glob("../data/2023_imagery/filtered/**/*CALIB_ML.txt")

for txt_path in txt_list:
    if not create_perSpoke_data(txt_path):
        break