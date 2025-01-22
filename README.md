# **Brain Tumor Classification from MRI**

## **Project Overview**
This project involves building a Convolutional Neural Network (CNN) model using **transfer learning** with **VGG16** to classify MRI brain scans into four categories: **Glioma Tumor**, **Meningioma Tumor**, **Pituitary Tumor**, and **No Tumor**. Additionally, a **Flask API** is created to expose the trained model as a web service for inference. The project also utilizes **TensorBoard** for visualizing the training process.

## **Table of Contents**
1. [Folder Structure](#folder-structure)
2. [Setup and Installation](#setup-and-installation)
3. [Dataset Structure](#dataset-structure)
4. [Training the Model](#training-the-model)
5. [Running the Flask API](#running-the-flask-api)
6. [Running TensorBoard](#running-tensorboard)
7. [Model Evaluation](#model-evaluation)
8. [Conclusion](#conclusion)

---

## **Folder Structure**

```
brain-tumor-classification/
│
├── data/
│   ├── training/
│   │   ├── glioma_tumor/
│   │   ├── meningioma_tumor/
│   │   ├── pituitary_tumor/
│   │   └── no_tumor/
│   ├── testing/
│   │   ├── glioma_tumor/
│   │   ├── meningioma_tumor/
│   │   ├── pituitary_tumor/
│   │   └── no_tumor/
│
├── models/
│   └── model.h5     # Saved trained model
│
├── logs/
│   └── tensorboard_logs/  # TensorBoard log files
│
├── scripts/
│   ├── load_data.py       # Data loading and preprocessing
│   ├── model.py           # Model architecture and compilation
│   ├── train.py           # Script to train the model
│   ├── evaluate.py        # Script to evaluate the model
│   └── api.py             # Flask API to serve the model
│
├── venv/                  # Virtual environment
│
├── .gitignore             # Files and folders to ignore in version control
├── requirements.txt       # Python dependencies
└── README.md              # Project instructions
```

---

## **Setup and Installation**

### **1. Clone the repository**

```bash
git clone https://github.com/your-repo/brain-tumor-classification.git
cd brain-tumor-classification
```

### **2. Create and activate a virtual environment**

#### On Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### **3. Install the required dependencies**

```bash
pip install -r requirements.txt
```

---

## **Dataset Structure**

Ensure your dataset is structured as follows inside the `data/` directory:

```
data/
├── training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── pituitary_tumor/
│   └── no_tumor/
├── testing/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── pituitary_tumor/
│   └── no_tumor/
```

Each folder should contain images that correspond to the respective tumor type or `no_tumor` (healthy MRI scans).

---

## **Training the Model**

To train the model on your dataset, run the following command:

```bash
python scripts/train.py
```

This script will:
- Load the training and testing data from the `data/` directory.
- Build the CNN model using **VGG16** transfer learning.
- Train the model for **50 epochs** with a **batch size of 32**.
- Save the best model to `models/model.h5`.
- Log training and validation metrics to the `logs/` directory for **TensorBoard** visualization.

---

## **Running the Flask API**

After training the model, you can expose it as an API for prediction using **Flask**.

### **1. Start the Flask server:**

```bash
python scripts/api.py
```

### **2. API Endpoints**

- **POST /predict**  
  **Description**: Allows you to upload an MRI image, and the model will classify the image into one of the tumor categories.  
  **Example Request (using `curl`)**:
  ```bash
  curl -X POST -F "file=@path/to/mri_image.jpg" http://127.0.0.1:5000/predict
  ```

  **Response**:
  ```json
  {
    "prediction": "glioma_tumor",
    "confidence": 0.89
  }
  ```

### **3. Integration**

The Flask API can be integrated with any mobile or web application to send MRI images for predictions and display the results to users.

---

## **Running TensorBoard**

You can visualize the model's training and validation metrics using **TensorBoard**.

1. Run the following command in your terminal:

   ```bash
   tensorboard --logdir=logs/tensorboard_logs
   ```

2. Open a web browser and navigate to `http://localhost:6006` to see the visualizations.

---

## **Model Evaluation**

After training, you can evaluate the model on the test dataset to check its performance:

```bash
python scripts/evaluate.py
```

This will print the model's accuracy on the test dataset.

Example Output:
```
Test accuracy: 0.9456
```

---

## **Conclusion**

This project demonstrates the use of **transfer learning** to classify brain tumors using MRI scans. With the **Flask API**, you can deploy the model as a web service, making it available for use in applications. The **TensorBoard** integration helps visualize the model's training progress, while the well-structured code makes it easy to extend or integrate into larger projects.

If you encounter any issues, feel free to check the **TensorBoard** logs for insights or consult the **README** for troubleshooting tips.
