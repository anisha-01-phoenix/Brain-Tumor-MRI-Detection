from fastapi import APIRouter, UploadFile, File
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

router = APIRouter()

# Load the autoencoder model with debugging
AUTOENCODER_MODEL_PATH = "models/autoencoder.h5"
print(f"[INFO] Loading autoencoder model from: {AUTOENCODER_MODEL_PATH}")
try:
    autoencoder = load_model(AUTOENCODER_MODEL_PATH)
    print(f"[INFO] Autoencoder model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load autoencoder model: {str(e)}")

# Preprocess the MRI image (resize and normalize)
def preprocess_image(image, target_size=(128, 128)):
    try:
        print(f"[INFO] Preprocessing image... original shape: {image.shape}")
        image = cv2.resize(image, target_size)
        print(f"[INFO] Image resized to: {image.shape}")
        
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        print(f"[INFO] Added channel dimension: {image.shape}")
        
        image = image.astype("float32") / 255.0  # Normalize the image
        print(f"[INFO] Image normalized. Shape: {image.shape}")
        
        image = np.expand_dims(image, axis=0)    # Add batch dimension
        print(f"[INFO] Added batch dimension: {image.shape}")
        
        return image
    except Exception as e:
        print(f"[ERROR] Failed to preprocess image: {str(e)}")
        return None

# Function to compute the anomaly score
def compute_anomaly_score(image):
    try:
        print(f"[INFO] Computing anomaly score... image shape: {image.shape}")
        
        # Reconstruct the image using the autoencoder
        reconstructed = autoencoder.predict(image)
        print(f"[INFO] Image reconstructed by autoencoder. Reconstructed shape: {reconstructed.shape}")
        
        # Compute the mean squared error between the original and reconstructed image
        mse = np.mean(np.power(image - reconstructed, 2))
        print(f"[INFO] Computed MSE (anomaly score): {mse}")
        
        return mse
    except Exception as e:
        print(f"[ERROR] Failed to compute anomaly score: {str(e)}")
        return None

# Define the anomaly detection endpoint
@router.post("/anomaly_detection")
async def detect_anomaly(mri_image: UploadFile = File(...)):
    print("[INFO] Anomaly detection API called.")
    
    try:
        # Read the uploaded image
        image_data = await mri_image.read()
        np_arr = np.frombuffer(image_data, np.uint8)
        print(f"[INFO] Image uploaded. Size of data: {len(np_arr)} bytes.")
        
        image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Invalid image format")
        
        print(f"[INFO] Image decoded. Shape: {image.shape}")
        
        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        if preprocessed_image is None:
            raise ValueError("Preprocessing failed")
        
        # Compute the anomaly score
        anomaly_score = compute_anomaly_score(preprocessed_image)
        if anomaly_score is None:
            raise ValueError("Anomaly score computation failed")
        
        # Define a threshold to classify as anomaly (you can tweak this value)
        threshold = 0.01
        is_anomaly = anomaly_score > threshold
        print(f"[INFO] Anomaly detection completed. Anomaly score: {anomaly_score}, Is anomaly: {is_anomaly}")
        
        # Return the anomaly score and whether it is an anomaly
        return {
            "anomaly_score": str(anomaly_score),
            "is_anomaly": str(is_anomaly)
        }
    except Exception as e:
        print(f"[ERROR] Failed in anomaly detection: {str(e)}")
        return {"error": str(e)}
