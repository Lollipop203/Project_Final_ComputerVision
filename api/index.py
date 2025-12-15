from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import shutil
import os
import cv2
import numpy as np
from pathlib import Path
import tempfile

# Standard imports assuming flattened structure
from api.detector import TrafficLightDetector
from api.utils import load_config

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants with fallback to temp directory for Vercel
TEMP_DIR = Path(tempfile.gettempdir()) / "traffic_lights"
TEMP_DIR.mkdir(exist_ok=True, parents=True)

# Initialize Detector
# Try to find model
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'trained' / 'epoch80_stripped.pt'

if not MODEL_PATH.exists():
    # Fallback to older name if stripped not found (user might have reverted)
    MODEL_PATH = PROJECT_ROOT / 'models' / 'trained' / 'epoch80.pt'

print(f"Loading model from: {MODEL_PATH}")
detector = TrafficLightDetector(model_path=str(MODEL_PATH))

# Mount static files
app.mount("/static", StaticFiles(directory="api/static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse('api/static/index.html')

@app.get("/status")
def get_status():
    return {"status": "online", "model": str(MODEL_PATH.name)}

@app.get("/files/{filename}")
async def get_file(filename: str):
    file_path = TEMP_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@app.post("/detect")
async def detect_object(
    file: UploadFile = File(...),
    conf: float = 0.25,
    iou: float = 0.45,
    save_image: bool = True
):
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
             raise HTTPException(status_code=400, detail="Invalid image file")

        # Detect
        results = detector.detect_image(image, conf=conf, iou=iou)
        
        output_filename = None
        if save_image:
            # Create a unique filename in temp
            filename = f"detected_{Path(file.filename).stem}.jpg"
            output_path = TEMP_DIR / filename
            
            # Save logic is inside detector but we can do it here to simpler
            # Or use detector's save method if updated to handle raw input?
            # Creating a dummy 'results' dict that matches detector.save_results expectation
            detector.save_results(results, image, output_path, save_json=False)
            output_filename = f"/files/{filename}"

        return {
            "success": True,
            "filename": file.filename,
            "results": {
                "count": results['num_detections'],
                "detections": results['detections'],
                "output_image": output_filename 
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
