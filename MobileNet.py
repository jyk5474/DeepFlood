import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2


# Load the mean images and labels from the saved files
mean_images = np.load(r'C:\Users\Jekoc\Desktop\DS340W\DS\mean_images.npy')
labels = np.load(r'C:\Users\Jekoc\Desktop\DS340W\DS\labels.npy')

print("Mean images shape:", mean_images.shape)
print("Labels shape:", labels.shape)

X_train, X_test, y_train, y_test = train_test_split(mean_images, labels, test_size=0.2, random_state=42)

base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Extract features using the pre-trained MobileNetV2 model
train_features = base_model.predict(X_train)
test_features = base_model.predict(X_test)


# Reshape the features to flatten them
train_features_flat = X_train.reshape(X_train.shape[0], -1)
test_features_flat = X_test.reshape(X_test.shape[0], -1)

# Train a Random Forest classifier
clf = RandomForestClassifier()
clf.fit(train_features_flat, y_train)

# Predict labels using the trained classifier
y_pred_train = clf.predict(train_features_flat)
y_pred_test = clf.predict(test_features_flat)

# Calculate accuracy
accuracy_train = np.mean(y_pred_train == y_train)
accuracy_test = np.mean(y_pred_test == y_test)
print("Training Accuracy:", accuracy_train)
print("Testing Accuracy:", accuracy_test)