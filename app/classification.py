from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from app.utils import preprocess_image, load_classification_model
import numpy as np
import cv2

router = APIRouter()

# Load model on app startup
model = load_classification_model()

# Class names (Glioma Tumor, Meningioma Tumor, Pituitary Tumor, No Tumor)
class_names = ['Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor', 'No Tumor']

@router.post("/classify_mri")
async def classify_mri(mri_image: UploadFile = File(...)):
    try:
        print("[INFO] Received MRI image for classification")
        # Read the uploaded file
        file_bytes = np.fromstring(await mri_image.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        if preprocessed_image is None:
            return JSONResponse({"error": "Failed to preprocess image"}, status_code=400)

        # Make prediction
        prediction = model.predict(preprocessed_image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence_scores = {class_name: float(conf) for class_name, conf in zip(class_names, prediction[0])}

        print("[INFO] Classification successful:", predicted_class)
        return JSONResponse({"classification": predicted_class, "confidence_scores": confidence_scores})

    except Exception as e:
        print("[ERROR] Classification failed:", str(e))
        return JSONResponse({"error": "Classification failed"}, status_code=500)
