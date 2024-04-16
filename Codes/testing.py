import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the pre-trained MobileNetV2 model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False # Freeze the base model

# Build the segmentation model with additional layers
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = UpSampling2D(size=(4, 4))(x)
x = Conv2D(256, (3, 3), padding='same')(x) # Additional convolution layer
x = BatchNormalization()(x)
x = Activation('relu')(x)
outputs = Conv2D(2, (1, 1), activation='softmax')(x) # Adjust the filter count if you have more classes
model = Model(inputs=inputs, outputs=outputs)

# Define the dataset paths
dataset_paths = [
    '/path/to/dataset1',
    '/path/to/dataset2',
    '/path/to/dataset3'
]

# Initialize variables to track accuracy
total_images = 0
correct_predictions = 0

# Loop through the dataset paths
for dataset_path in dataset_paths:
    for filename in os.listdir(dataset_path):
        if filename.endswith('.png'):
            image_path = os.path.join(dataset_path, filename)
            image = load_img(image_path, target_size=(224, 224)) # Load the image using Keras preprocessing
            image = img_to_array(image) # Convert the image to an array
            image = preprocess_input(image) # Use the corresponding preprocess_input function
            image = np.expand_dims(image, axis=0) # Add batch dimension

            # Make predictions
            predictions = model.predict(image)
            predictions = np.argmax(predictions, axis=-1)

            # Check for flood or no flood
            if np.any(predictions == 1):
                print(f'FLOOD: {filename}')
            else:
                print(f'NOFLOOD: {filename}')

            # Update accuracy tracking
            total_images += 1
            if (np.any(predictions == 1) and dataset_path[-1] == '1') or (not np.any(predictions == 1) and dataset_path[-1] == '0'):
                correct_predictions += 1

# Print the overall accuracy
accuracy = (correct_predictions / total_images) * 100
print(f'Accuracy: {accuracy:.2f}%')
