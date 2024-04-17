import numpy as np
import os
from PIL import Image
import json

# Define the dataset root directory
dataset_root = r'C:\Users\Jekoc\Desktop\DS340W\DS\SEN12FLOOD'

# Load the JSON data
with open(r'C:\Users\Jekoc\Desktop\DS340W\DS\SEN12FLOOD\S1list.json') as json_file:
    data = json.load(json_file)

# Get a list of all the numbered folders inside the dataset root directory
numbered_folders = [folder for folder in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, folder))]

mean_images = []
labels = []

target_size = (224, 224)

# Loop through each numbered folder
for folder in numbered_folders:
    folder_path = os.path.join(dataset_root, folder)
    
    # Get a list of all files in the current folder
    file_names = os.listdir(folder_path)
    
    # Initialize a dictionary to store pairs of images
    image_pairs = {}
    
    # Loop through each file in the folder
    for file_name in file_names:
        # Check if the file starts with 'S2' and ends with '.tif'
        if file_name.startswith('S2') and file_name.endswith('.tif'):
            # Extract the date from the file name
            date = file_name.split('_')[1]
            
            # Append the file name to the list of files for the corresponding date
            if date in image_pairs:
                image_pairs[date].append(file_name)
            else:
                image_pairs[date] = [file_name]
    
    # Loop through each date and its corresponding images
    for date, image_list in image_pairs.items():
        images = []
        # Load and preprocess each image in the pair
        for file_name in image_list:
            image_path = os.path.join(folder_path, file_name)
            try:
                image = Image.open(image_path)
                resized_image = image.resize(target_size)
                image_array = np.array(resized_image)
                images.append(image_array)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        # Compute the mean image for the pair
        if len(images) == 12:  # Check if there are exactly 12 images for each date
            mean_image = np.mean(images, axis=0)
            mean_images.append(mean_image)
            
            # Get the label from the JSON data based on the date
            label = 1 if data[folder][date]['FLOODING'] else 0
            labels.append(label)

# Convert lists to numpy arrays
mean_images = np.array(mean_images)
labels = np.array(labels)

print("Mean images shape:", mean_images.shape)
print("Labels shape:", labels.shape)

np.save(r'C:\Users\Jekoc\Desktop\DS340W\DS\MSmean_images.npy', mean_images)
np.save(r'C:\Users\Jekoc\Desktop\DS340W\DS\MSlabels.npy', labels)