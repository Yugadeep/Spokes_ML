#!/usr/bin/env -S python3

#created using ChatGPT

from PIL import Image
import os
import glob
import sys

def resize_and_convert_images(input_dir, output_dir, target_size=(736, 160)):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all image files in the input directory
    image_files = glob.glob(os.path.join(input_dir, '*.png'))  # Adjust the extension if needed

    for image_file in image_files:
        # Open the image
        img = Image.open(image_file)
        
        # Convert to grayscale
        img_gray = img.convert('L')  # 'L' mode is for grayscale images
        
        # Resize the image
        img_resized = img_gray.resize(target_size)
        
        # Get the base filename and create output path
        base_filename = os.path.basename(image_file)
        output_path = os.path.join(output_dir, base_filename)
        
        # Save the processed image
        img_resized.save(output_path)
        
        print(f"Processed and saved: {output_path}")

# Example usage
input_directory = sys.argv[1]   # Change this to your input directory
output_directory = sys.argv[2] # Change this to your output directory
resize_and_convert_images(input_directory, output_directory)
