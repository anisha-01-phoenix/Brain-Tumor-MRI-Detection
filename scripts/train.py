# scripts/train.py
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from load_data import load_train_test_data
from model import build_model

# Directories
data_dir = '../data'  
model_dir = '../models/model.h5'
log_dir = '../logs/tensorboard_logs'

# Load data
print(f"Loading data from {data_dir}...")
X_train, X_test, y_train, y_test, class_names = load_train_test_data(data_dir)

# Build the model
print("Building the model...")
input_shape = X_train.shape[1:]  # (128, 128, 3)
model = build_model(input_shape, num_classes=len(class_names))

# Callbacks
print("Setting up callbacks...")
checkpoint = ModelCheckpoint(model_dir, monitor='val_accuracy', save_best_only=True, mode='max')
tensorboard = TensorBoard(log_dir=log_dir)

# Train the model
print(f"Training the model with {len(X_train)} training samples and {len(X_test)} testing samples...")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[checkpoint, tensorboard])

# Save the final model
print(f"Saving the trained model to {model_dir}...")
model.save(model_dir)
