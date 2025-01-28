# **Brain Tumor MRI Detection**

This project uses a **Convolutional Neural Network (CNN)** model, leveraging **transfer learning** with **NasNetMobile**, to classify MRI brain scans into four categories: **Glioma Tumor**, **Meningioma Tumor**, **Pituitary Tumor**, and **No Tumor**. Additionally, the project implements a **FastAPI** service to expose the trained model as a web service for inference, with support for **Grad-CAM** visualizations and **Anomaly Detection**. Training visualizations are handled by **TensorBoard** for monitoring and insights into the model's performance.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [Setup and Installation](#setup-and-installation)
4. [Dataset Structure](#dataset-structure)
5. [ML Models](#ml-models)
6. [Training the Model](#training-the-model)
7. [Model Evaluation](#model-evaluation)
8. [Running TensorBoard](#running-tensorboard)
9. [Running the FastAPI](#running-the-fastapi)
10. [API Endpoints](#api-endpoints)
11. [Conclusion](#conclusion)

---

## **Project Overview**

This project involves building a robust CNN model using **NasNetMobile** with transfer learning for classifying MRI brain scans into different tumor categories. It also includes an anomaly detection system using an autoencoder model to flag unusual MRI images. The trained models are made accessible through a **FastAPI** web service, and the training process is visualized using **TensorBoard**.

The API is hosted on **Render** and can be used to classify images, generate Grad-CAM visualizations, and perform anomaly detection.

---

## **Folder Structure**

```bash
brain-tumor-classification/
│
├── app/
│   ├── __init__.py             # Initializes the FastAPI and registers the routers 
│   ├── anomaly_detection.py     # Contains logic for anomaly detection
│   ├── classification.py        # Contains logic for MRI classification
│   ├── grad_cam.py              # Generates Grad-CAM visualizations
│   └── utils.py                 # Utility functions for image preprocessing
│
├── data/
│   ├── training/
│   │   ├── glioma_tumor/        # Glioma tumor MRI images
│   │   ├── meningioma_tumor/    # Meningioma tumor MRI images
│   │   ├── pituitary_tumor/     # Pituitary tumor MRI images
│   │   └── no_tumor/            # Healthy MRI images (No tumor)
│   ├── testing/
│   │   ├── glioma_tumor/
│   │   ├── meningioma_tumor/
│   │   ├── pituitary_tumor/
│   │   └── no_tumor/
│
├── models/
│   └── model.h5                 # Saved trained classification model
│   └── autoencoder.h5           # Saved anomaly detection autoencoder model
│
├── logs/
│   └── tensorboard_logs/        # TensorBoard log files for monitoring training progress
│
├── static/heatmap/              # Stores generated GradCAM heatmaps
|
├── scripts/
│   ├── load_data.py             # Data loading and preprocessing
│   ├── model.py                 # Model architecture and compilation
│   ├── train.py                 # Script to train the classification model
│   ├── evaluate.py              # Script to evaluate the models
│   └── autoencoder.py           # Script to train the autoencoder model
│
├── venv/                        # Virtual environment
│
├── main.py                      # Entry point of the FastAPI server
├── .gitignore                   # Files and folders to ignore in version control
├── requirements.txt             # Python dependencies
└── README.md                    # Project instructions
```

---

## **Setup and Installation**

### **1. Clone the repository**

```bash
git clone https://github.com/anisha-01-phoenix/Brain-Tumor-MRI-Detection.git
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

```bash
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

Each folder should contain images corresponding to the respective tumor type or `no_tumor` (healthy MRI scans).

---

## **ML Models**

- **NasNetMobile**: A convolutional neural network pre-trained on ImageNet and fine-tuned for brain tumor classification.
- **Autoencoder**: A neural network used for anomaly detection by reconstructing MRI images and identifying abnormal patterns.

---

## **Training the Model**

To train the model on your dataset, run the following commands:

```bash
python scripts/train.py
python scripts/autoencoder.py
```

This will:
- Load the training and testing data from the `data/` directory.
- Build the CNN model using **NasNETMobile** and the autoencoder model.
- Train the models for **50 epochs** with a **batch size of 32**.
- Save the trained classification model to `models/model.h5`.
- Save the autoencoder model to `models/autoencoder.h5`.

---

## **Model Evaluation**

To evaluate the performance of the trained models, use the following command:

```bash
python scripts/evaluate.py
```

The evaluation results will be printed and saved in the logs.

---

## **Running TensorBoard**

To visualize the training process and logs in **TensorBoard**, run the following command:

```bash
tensorboard --logdir=logs/tensorboard_logs
```

Open a web browser and navigate to `http://localhost:6006` to see the visualizations.

---

## **Running the FastAPI**

### **1. Run the FastAPI server locally**

To start the FastAPI server on your machine, run:

```bash
uvicorn main:app --reload
```

This will start the server, and you can access the API at `http://127.0.0.1:8000`.

### **2. Render Deployment**

The API is also hosted on Render. You can access the deployed version of the FastAPI [here](https://brain-tumor-mri-classification.onrender.com/).

---

## **API Endpoints**

### **1. Tumor Classification**

- **Endpoint**: `/classify_mri`
- **Method**: `POST`
- **Description**: Upload an MRI image, and the API returns the predicted tumor type.
- **Request Body**: Form data with the image file.
- **Example of API Response** :
```json
{
    "classification": "Meningioma Tumor",
    "confidence_scores": {
        "Glioma Tumor": 0.00011530885967658833,
        "Meningioma Tumor": 0.7650470733642578,
        "Pituitary Tumor": 0.13927949965000153,
        "No Tumor": 0.09555811434984207
    }
}
```

### **2. Grad-CAM Visualization**

- **Endpoint**: `/grad_cam_visualization`
- **Method**: `POST`
- **Description**: Upload an MRI image, and the API returns the Grad-CAM visualization, highlighting regions of interest.
- **Request Body**: Form data with the image file.
- **Example of API Response** :
```json

```

### **3. Anomaly Detection**

- **Endpoint**: `/anomaly_detection`
- **Method**: `POST`
- **Description**: Upload an MRI image, and the API checks for any anomalies.
- **Request Body**: Form data with the image file.
- **Example of API Response** :
```json

```

---

## **Conclusion**

This project demonstrates the effective use of **transfer learning** for classifying brain tumors using MRI scans. The **FastAPI** interface allows easy integration with other applications and services. Furthermore, **TensorBoard** enables real-time visualization of the training process, making it easier to tune hyperparameters and monitor model performance.

With the **Grad-CAM** functionality, the model's predictions can be made more interpretable by highlighting regions in the MRI scans that are most relevant to the classification. The autoencoder-based **anomaly detection** adds an extra layer of insight by detecting unusual patterns in the scans.

