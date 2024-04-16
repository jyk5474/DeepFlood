import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

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

# Initialize variables to track performance
total_images = 0
y_true = []
y_pred = []

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

            # Track true and predicted labels
            if dataset_path[-1] == '1':
                y_true.append(1) # Flood
            else:
                y_true.append(0) # No Flood
            y_pred.append(predictions[0])

            total_images += 1

# Compute performance metrics
accuracy = np.mean(np.array(y_true) == np.array(y_pred))
confusion_mat = confusion_matrix(y_true, y_pred)
classification_rep = classification_report(y_true, y_pred)

print(f'Accuracy: {accuracy:.2f}%')
print('Confusion Matrix:')
print(confusion_mat)
print('Classification Report:')
print(classification_rep)

# Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
