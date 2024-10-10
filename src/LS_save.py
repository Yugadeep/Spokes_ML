#!/usr/bin/env -S python3


import os
import json
import sys


json_path = sys.argv[1]  # path to the json file which you are supposed to download from label studio. it has information about which mask corresponds to which opus id image. This specific code expects you to have downloaded JSON file from label studio. Future version will need the JSON_min option as that's mroe relevant and easier to understand.
masks_dirpath = sys.argv[2]

# Load the JSON file with annotations
with open(json_path, 'r') as f:
    annotations = json.load(f)


# Path where the exported PNG files are located
exported_png_path = masks_dirpath

# Create a mapping of task IDs to original file names
task_to_filename = {}
for task in annotations:
    task_id = task['id']
    original_file_name = os.path.basename(task['data']['image'])  # Extract the original file name
    task_to_filename[task_id] = original_file_name

# Track how many times each task ID is used to handle duplicate file names
task_usage_count = {}

# Iterate over the PNG files in the export folder
for png_file in os.listdir(exported_png_path):
    if png_file.startswith("task"):  # Match the Label Studio exported task file names
        task_id = int(png_file.split('-')[1])  # Extract the task ID from the file name
        
        if task_id in task_to_filename:
            original_file_name = task_to_filename[task_id]
            base_name = os.path.splitext(original_file_name)[0]

            # Handle duplicate annotations by appending an index
            task_usage_count[task_id] = task_usage_count.get(task_id, 0) + 1
            new_file_name = f"{base_name}_mask_{task_usage_count[task_id]}.png"  # Add an index to the file name

            old_path = os.path.join(exported_png_path, png_file)
            new_path = os.path.join(exported_png_path, new_file_name)

            os.rename(old_path, new_path)
            print(f"Renamed {old_path} to {new_path}")
