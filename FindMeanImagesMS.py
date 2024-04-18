import numpy as np
import os
from PIL import Image
import json

dataset_root = r'C:\Users\Jekoc\Desktop\DS340W\DS\SEN12FLOOD'
with open(r'C:\Users\Jekoc\Desktop\DS340W\DS\SEN12FLOOD\S2list.json') as json_file:
    data = json.load(json_file)
numbered_folders = [folder for folder in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, folder))]

mean_images = []
labels = []


target_size = (224, 224)

for folder in numbered_folders:
    folder_path = os.path.join(dataset_root, folder)
    if not os.path.isdir(folder_path):
        continue
    
    # Initialize a dictionary to store images for each date
    date_images = {}
    for file_name in os.listdir(folder_path):
        if not file_name.endswith('.tif'):
            continue
        
        if file_name.startswith('S2'):
            # extract date from the file name
            date = file_name.split('_')[1][:10]  # Extract the date from the file name
            
            # add the image to the date group
            if date in date_images:
                date_images[date].append(file_name)
            else:
                date_images[date] = [file_name]
    
    for date, image_list in date_images.items():
        if len(image_list) == 12:
            images = []
            for image_name in image_list:
                image_path = os.path.join(folder_path, image_name)
                try:
                    image = Image.open(image_path)
                    image = image.resize(target_size)  # Resize the image to a common size
                    images.append(np.array(image))
                except:
                    print(f"Error opening image: {image_path}")
            
            # Check if any images were loaded successfully
            if len(images) == 12:
                # find the mean image
                mean_image = np.mean(images, axis=0)
                mean_images.append(mean_image)
                folder_data = data.get(folder, {})
                
                flooding_label = False
                for entry in folder_data.values():
                    entry_date = entry.get('date')
                    entry_filename = entry.get('filename', '')
                    print("Entry Date:", entry_date)
                    print("Entry Filename:", entry_filename)

                    if entry.get('date') == date or entry.get('filename', '').startswith('S2_' + date):
                        flooding_label = entry.get('FLOODING', False)
                        print("Flooding Label:", flooding_label)
                        break
                
                labels.append(int(flooding_label))
            else:
                print(f"Skipping date {date} in folder {folder} due to inability to open some images")


mean_images = np.array(mean_images)
labels = np.array(labels)

print("Mean images shape:", mean_images.shape)
print("Labels shape:", labels.shape)
print(labels)

# save the mean images and labels
np.save(r'C:\Users\Jekoc\Desktop\DS340W\DS\MSmean_images.npy', mean_images)
np.save(r'C:\Users\Jekoc\Desktop\DS340W\DS\MSlabels.npy', labels)
