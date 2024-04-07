import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the pre-trained MobileNetV2 model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model

# Build the binary classification model with additional layers
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)  # Single output for binary classification
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Set the path to the validation folder
val_dir = '/Users/marvin/Desktop/2023 Classes/DS 340W/FloodNet/val/val-org-img'

# Load and preprocess the validation images
val_images = []
image_filenames = []

for img_name in os.listdir(val_dir):
    img_path = os.path.join(val_dir, img_name)
    if os.path.isfile(img_path):
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        val_images.append(image)
        image_filenames.append(img_name)

# Convert list to numpy array
val_images = np.array(val_images)

# Make predictions
predictions = model.predict(val_images)

# Threshold predictions
predicted_labels = np.where(predictions > 0.5, 'FLOOD', 'NOFLOOD')

# Print predictions for each file
for filename, prediction in zip(image_filenames, predicted_labels):
    print(f"{filename}, {prediction}")
