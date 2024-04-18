import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.ensemble import RandomForestClassifier
import os
from PIL import Image
import json
import pickle
from sklearn.model_selection import train_test_split

#root directory
dataset_root = r'C:\Users\Jekoc\Desktop\DS340W\DS\SEN12FLOOD'

# JSON data
with open(r'C:\Users\Jekoc\Desktop\DS340W\DS\SEN12FLOOD\S1list.json') as json_file:
    data = json.load(json_file)
    
numbered_folders = [folder for folder in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, folder))]

mean_images = []
labels = []

target_size = (224, 224)

for folder in numbered_folders:
    folder_path = os.path.join(dataset_root, folder)
    
    file_names = os.listdir(folder_path)
    image_pairs = {}
    
    
    for file_name in file_names:
        if file_name.endswith('_corrected_VH.tif') or file_name.endswith('_corrected_VV.tif'):
            # Extract the prefix of the image pair
            prefix = file_name.split('_')[8]
            print("Prefix:", prefix)
            
            # add to the list of files for the corresponding prefix
            if prefix in image_pairs:
                image_pairs[prefix].append(file_name)
            else:
                image_pairs[prefix] = [file_name]
    
    for prefix, pair in image_pairs.items():
        images = []
        for file_name in pair:
            image_path = os.path.join(folder_path, file_name)
            #resize and append images
            try:
                image = Image.open(image_path)
                resized_image = image.resize(target_size)
                image_array = np.array(resized_image)
                images.append(image_array)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        # find mean image for the pair
        if len(images) == 2:
            mean_image = np.mean(images, axis=0)
            mean_image = np.stack((mean_image,) * 3, axis=-1)
            
            # Get the label
            json_filenames = [data[folder][key]['filename'] for key in data[folder] if key.isdigit()]
            json_prefixes = [json_filename.split('_')[-1][-4:] for json_filename in json_filenames]
            
            if prefix in json_prefixes:
                label_index = json_prefixes.index(prefix)
                label = 1 if data[folder][str(label_index + 1)]['FLOODING'] else 0
                mean_images.append(mean_image)
                labels.append(label)
            else:
                print(f"Prefix {prefix} not found in JSON data for folder {folder}")


mean_images = np.array(mean_images)
labels = np.array(labels)

print("Mean images shape:", mean_images.shape)
print("Labels shape:", labels.shape)
print(labels)

np.save(r'C:\Users\Jekoc\Desktop\DS340W\DS\SARmean_images.npy', mean_images)
np.save(r'C:\Users\Jekoc\Desktop\DS340W\DS\SARlabels.npy', labels)
