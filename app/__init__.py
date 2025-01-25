from fastapi import FastAPI

# Initialize the FastAPI app
def create_app():
    app = FastAPI()

    # Import all the routes
    from app.classification import router as classification_router
    from app.grad_cam import router as grad_cam_router
    from app.anomaly_detection import router as anomaly_router
    # from app.tumor_growth import router as growth_router
    # from app.treatment_recommendation import router as treatment_router

    # Register routers
    app.include_router(classification_router)
    app.include_router(grad_cam_router)
    app.include_router(anomaly_router)
    # app.include_router(growth_router)
    # app.include_router(treatment_router)

    return app
