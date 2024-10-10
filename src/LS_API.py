import json
import requests
import os

# Load JSON data
with open(r"C:\Users\yugad\Spokes_ML\comparing_ds_test\allmasksFlabel_studio.json") as f:
    data = json.load(f)

# Directory to save images
os.makedirs('images', exist_ok=True)

# Extract image URLs and download images
for task in data['tasks']:
    image_url = task['data']['image']
    image_name = os.path.basename(image_url)
    
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(os.path.join('images', image_name), 'wb') as img_file:
            img_file.write(response.content)


from PIL import Image
import os

input_directory = 'images'
output_directory = 'png_images'

os.makedirs(output_directory, exist_ok=True)

for file_name in os.listdir(input_directory):
    if file_name.endswith('.jpg'):  # or other formats
        with Image.open(os.path.join(input_directory, file_name)) as img:
            png_file_name = os.path.splitext(file_name)[0] + '.png'
            img.save(os.path.join(output_directory, png_file_name), 'PNG')
