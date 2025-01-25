from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import os

data_dir = '../data'  
autoencoder_model_path = '../models/autoencoder.h5'
train_data_path = '../data/training/no_tumor/'

# Build an autoencoder model with debug statements
def build_autoencoder(input_shape):
    print(f"[INFO] Building autoencoder model with input shape: {input_shape}")
    
    input_img = Input(shape=input_shape)
    
    # Encoder
    print(f"[INFO] Adding encoder layers...")
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    print(f"[INFO] Encoder layer 1 completed.")
    
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    print(f"[INFO] Encoder layer 2 completed.")
    
    # Decoder
    print(f"[INFO] Adding decoder layers...")
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    print(f"[INFO] Decoder layer 1 completed.")
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    print(f"[INFO] Decoder layer 2 completed.")
    
    autoencoder = Model(input_img, decoded)
    print(f"[INFO] Autoencoder model built successfully.")
    return autoencoder

# Preprocess the MRI images and prepare the dataset
def load_no_tumor_images(directory, target_size=(128, 128)):
    print(f"[INFO] Loading images from directory: {directory}")
    
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
    
    print(f"[INFO] Loaded {len(images)} images from {directory}")
    return np.array(images)

# Load and preprocess the images
print("[INFO] Starting to load 'No Tumor' MRI images for training...")
X_train = load_no_tumor_images(train_data_path)

# Normalize and reshape the images
print("[INFO] Preprocessing images...")
X_train = X_train.astype('float32') / 255.
X_train = np.reshape(X_train, (len(X_train), 128, 128, 1))
print(f"[INFO] Dataset shape after preprocessing: {X_train.shape}")

# Build the autoencoder model
input_shape = (128, 128, 1)
autoencoder = build_autoencoder(input_shape)

# Compile the model
print("[INFO] Compiling the autoencoder model...")
autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')
print("[INFO] Autoencoder model compiled successfully.")

# Train the autoencoder on the "No Tumor" images
print(f"[INFO] Starting training of the autoencoder model on {X_train.shape[0]} images...")
autoencoder.fit(X_train, X_train, epochs=50, batch_size=64, shuffle=True)
print(f"[INFO] Training completed successfully.")

# Save the trained model
print(f"[INFO] Saving the trained model to: {autoencoder_model_path}")
autoencoder.save(autoencoder_model_path)
print(f"[INFO] Model saved successfully.")
