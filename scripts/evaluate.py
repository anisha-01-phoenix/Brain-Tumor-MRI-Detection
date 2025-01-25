from tensorflow.keras.models import load_model
from load_data import load_train_test_data
import numpy as np
import cv2
import os

data_dir = '../data'  
model_dir = '../models/model.h5'
autoencoder_model_path = '../models/autoencoder.h5'
test_data_path = '../data/testing/no_tumor/'

# Load the model
print(f"Loading the model from {model_dir}...")
model = load_model(model_dir)

# Load the test data
print(f"Loading test data from {data_dir}...")
_, X_test, _, y_test, _ = load_train_test_data(data_dir)

# Evaluate the model on the test set
print(f"Evaluating the model on the test data...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy:.4f}')

# ===========================================================
# AUTOENCODER EVALUATION FOR ANOMALY DETECTION
# ===========================================================

# Load the pre-trained autoencoder model
print(f"Loading the autoencoder model from {autoencoder_model_path}...")
autoencoder = load_model(autoencoder_model_path)

# lload and preprocess the "No Tumor" test images
def load_no_tumor_images(directory, target_size=(128, 128)):
    print(f"[INFO] Loading test images from directory: {directory}")
    
    images = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Read the image as grayscale
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image_resized = cv2.resize(image, target_size)
            images.append(image_resized)
        else:
            print(f"[WARN] Failed to load image: {file_path}")
    
    print(f"[INFO] Loaded {len(images)} test images from {directory}")
    return np.array(images)

# Load and preprocess the test images
print("[INFO] Loading 'No Tumor' test images for autoencoder evaluation...")
X_test_no_tumor = load_no_tumor_images(test_data_path)

# Normalize and reshape the images to match the autoencoder input
print("[INFO] Preprocessing the test images for autoencoder...")
X_test_no_tumor = X_test_no_tumor.astype('float32') / 255.
X_test_no_tumor = np.reshape(X_test_no_tumor, (len(X_test_no_tumor), 128, 128, 1))
print(f"[INFO] Test dataset shape for autoencoder: {X_test_no_tumor.shape}")

# Use the autoencoder to reconstruct the test images
print("[INFO] Reconstructing the 'No Tumor' test images using the autoencoder...")
reconstructed = autoencoder.predict(X_test_no_tumor)

# Calculate the reconstruction error (Mean Squared Error - MSE) for each image
print("[INFO] Calculating reconstruction errors...")
mse = np.mean(np.power(X_test_no_tumor - reconstructed, 2), axis=(1, 2, 3))

# Set a threshold for anomaly detection (e.g., 95th percentile of the MSE from training)
threshold = np.percentile(mse, 95)
print(f"[INFO] Anomaly detection threshold set to: {threshold}")

# Predict anomalies: Any image with reconstruction error above the threshold is considered an anomaly
y_pred = (mse > threshold).astype(int)  # 1 = anomaly, 0 = normal

# Display the MSE and anomaly detection results
print(f"Mean Squared Error (MSE) for each image:\n{mse}")
print(f"Detected anomalies (1 = anomaly, 0 = normal):\n{y_pred}")

