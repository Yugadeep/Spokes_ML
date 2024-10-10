#!/usr/bin/env -S python3


# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:27:36 2024

@author: yugad

JK's json file was wierd, in that the image names had a strange prefix seperating the actual thumbnail with a hyphen. 
This code will remove that prefix and the hyphen, and save the filename as desired.
"""

import os
import sys


exported_png_path = sys.argv[1]


for png_file in os.listdir(exported_png_path):
    if png_file.endswith(".png") :  # Match the Label Studio exported task file names
        new_name = png_file.split('-')[1]  # Extract the task ID from the file name
        
        new_file_name = f"{new_name}"  # Add an index to the file name

        old_path = os.path.join(exported_png_path, png_file)
        new_path = os.path.join(exported_png_path, new_file_name)

        os.rename(old_path, new_path)
        print(f"Renamed {old_path} to {new_path}")
