from app import create_app
import uvicorn

# Create the app instance
app = create_app()

# Entry point to run the FastAPI server using Uvicorn
if __name__ == "__main__":
    print("[INFO] Starting FastAPI server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
