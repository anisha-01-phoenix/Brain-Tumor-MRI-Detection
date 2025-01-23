from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from app.utils import preprocess_image, generate_gradcam, overlay_heatmap_on_image, save_image
import numpy as np
import cv2
import os

router = APIRouter()

@router.post("/grad_cam_visualization")
async def grad_cam_visualization(mri_image: UploadFile = File(...)):
    try:
        print("[INFO] Received MRI image for Grad-CAM visualization")
        # Read the uploaded file
        file_bytes = np.frombuffer(await mri_image.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        if preprocessed_image is None:
            return JSONResponse({"error": "Failed to preprocess image"}, status_code=400)

        # Generate Grad-CAM heatmap
        heatmap = generate_gradcam(preprocessed_image)

        if heatmap is None:
            return JSONResponse({"error": "Failed to generate Grad-CAM"}, status_code=500)

        # Overlay heatmap on original image
        overlayed_image = overlay_heatmap_on_image(image, heatmap)

        if overlayed_image is None:
            return JSONResponse({"error": "Failed to overlay heatmap"}, status_code=500)

        # Save the result as an image
        heatmap_path = f'static/heatmaps/{mri_image.filename}_heatmap.jpg'
        save_image(overlayed_image, heatmap_path)

        return FileResponse(heatmap_path)

    except Exception as e:
        print("[ERROR] Grad-CAM visualization failed:", str(e))
        return JSONResponse({"error": "Grad-CAM visualization failed"}, status_code=500)
