import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

def load_images_from_directory(data_dir, img_size=(128, 128), augment=False):
    print(f"Loading images from {data_dir}...")
    images = []
    labels = []
    class_names = os.listdir(data_dir)
    print(f"Found classes: {class_names}")

    datagen = None
    if augment:
        print("Applying data augmentation...")
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        print(f"Processing class: {class_name}")
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"Skipping invalid image: {img_path}")
                        continue
                    image = cv2.resize(image, img_size)
                    images.append(image)
                    labels.append(class_names.index(class_name))

                    # Apply augmentation if enabled
                    if augment and datagen:
                        image = np.expand_dims(image, 0)  # Expand dims for single image
                        for aug_img in datagen.flow(image, batch_size=1):
                            aug_img = np.squeeze(aug_img)  # Remove batch dimension
                            images.append(aug_img)
                            labels.append(class_names.index(class_name))
                            break  # Use one augmentation per image
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
            else:
                print(f"Skipping non-image file: {img_file}")
    
    print(f"Loaded {len(images)} images.")
    images = np.array(images, dtype='float32') / 255.0  # Normalize to [0, 1]
    labels = np.array(labels)
    print(f"Image array shape: {images.shape}, Labels array shape: {labels.shape}")
    return images, labels, class_names

def load_train_test_data(data_dir, img_size=(128, 128), augment=False):
    print(f"Loading training and testing data from {data_dir}...")
    
    # Load training data with augmentation
    train_dir = os.path.join(data_dir, 'training')
    print(f"Loading training data from {train_dir}...")
    X_train, y_train, class_names = load_images_from_directory(train_dir, img_size, augment=augment)
    
    # Load testing data without augmentation
    test_dir = os.path.join(data_dir, 'testing')
    print(f"Loading testing data from {test_dir}...")
    X_test, y_test, _ = load_images_from_directory(test_dir, img_size, augment=False)
    
    # Convert labels to one-hot encoding
    print(f"Converting labels to one-hot encoding...")
    y_train = to_categorical(y_train, num_classes=len(class_names))
    y_test = to_categorical(y_test, num_classes=len(class_names))
    
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, class_names
