import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.nasnet import preprocess_input
from tensorflow.keras.models import load_model
import tensorflow as tf
import os

# Load the model
def load_classification_model(model_path='models/model.h5'):
    print("[INFO] Loading model from:", model_path)
    model = load_model(model_path)
    print("[INFO] Model loaded successfully")
    return model

# Preprocess the MRI image
def preprocess_image(image, target_size=(128, 128)):
    try:
        print("[INFO] Preprocessing image...")
        image = cv2.resize(image, target_size)
        image = img_to_array(image)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)
        print("Image shape after preprocessing:", image.shape)
        print("[INFO] Image preprocessed successfully")
        return image
    except Exception as e:
        print("[ERROR] Failed to preprocess image:", str(e))
        return None

# Generate Grad-CAM heatmap
def generate_gradcam(image, layer_name="separable_conv_2_bn_normal_left5_12"):
    model = load_classification_model()
    try:
        print("[INFO] Generating Grad-CAM heatmap...")

        # if image.ndim == 3:
        #     image = np.expand_dims(image, axis=0)

        grad_model = tf.keras.models.Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])

        # Convert the image to a tensor
        image = tf.convert_to_tensor(image, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            loss = predictions[:, np.argmax(predictions[0])]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = np.mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        output_list = []

        for i in range(conv_outputs.shape[-1]):
            modified_output = conv_outputs[:, :, i] * pooled_grads[i] 
            output_list.append(modified_output)
        conv_outputs = tf.stack(output_list, axis=-1)
        
        heatmap = np.mean(conv_outputs, axis=-1)

        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))

        print("[INFO] Grad-CAM heatmap generated successfully")
        return heatmap
    except Exception as e:
        print("[ERROR] Failed to generate Grad-CAM:", str(e))
        return None

# Save the heatmap
def overlay_heatmap_on_image(original_image, heatmap):
    try:
        print("[INFO] Overlaying heatmap on original image...")
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlayed_image = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        print("[INFO] Heatmap overlay successful")
        return overlayed_image
    except Exception as e:
        print("[ERROR] Failed to overlay heatmap:", str(e))
        return None

# Save image to file
def save_image(image, path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] Created missing directory: {directory}")
    try:
        print(f"[INFO] Saving image to {path}...")
        cv2.imwrite(path, image)
        print(f"[INFO] Image saved at {path}")
    except Exception as e:
        print("[ERROR] Failed to save image:", str(e))
