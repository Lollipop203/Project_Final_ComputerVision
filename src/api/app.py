"""
FastAPI Application for Traffic Light Detection
RESTful API Server
"""

import os
import sys
import io
import shutil
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import cv2
import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.detector import TrafficLightDetector
from src.agent.utils import load_config


# Initialize FastAPI app
app = FastAPI(
    title="Traffic Light Detection API",
    description="API for detecting traffic lights in images and videos using YOLOv8",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
config_path = PROJECT_ROOT / 'configs' / 'config.yaml'
if config_path.exists():
    config = load_config(str(config_path))
else:
    config = {
        'api': {'max_upload_size': 10485760},
        'paths': {'output_dir': 'output'}
    }

# Create output directory
OUTPUT_DIR = Path(config['paths'].get('output_dir', 'output'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create static directory if it doesn't exist
STATIC_DIR = Path(__file__).parent / 'static'
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/results", StaticFiles(directory=str(OUTPUT_DIR)), name="results")

# Global detector instance
detector = None


def get_detector():
    """Get or initialize the detector instance"""
    global detector
    if detector is None:
        # Check if trained model path is specified in config
        model_path = config.get('model', {}).get('trained_model')
        if model_path:
            detector = TrafficLightDetector(model_path=model_path)
        else:
            detector = TrafficLightDetector()
    return detector


# Pydantic models
class DetectionResponse(BaseModel):
    """Response model for detection endpoint"""
    success: bool
    num_detections: int
    detections: List[dict]
    image_shape: List[int]
    image_url: Optional[str] = None
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    version: str


@app.on_event("startup")
async def startup_event():
    """Initialize detector on startup"""
    print("Starting Traffic Light Detection API...")
    get_detector()
    print("✓ API ready!")


@app.get("/", tags=["Root"])
async def root():
    """Serve the web app"""
    index_path = STATIC_DIR / 'index.html'
    if index_path.exists():
        return FileResponse(index_path)
    else:
        return {
            "message": "Traffic Light Detection API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health"
        }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check API health status
    
    Returns:
        Health status including model loading state
    """
    det = get_detector()
    return HealthResponse(
        status="healthy",
        model_loaded=det is not None and det.model is not None,
        version="1.0.0"
    )


@app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_image(
    file: UploadFile = File(...),
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    save_image: bool = True
):
    """
    Detect traffic lights in an uploaded image
    
    Args:
        file: Image file (jpg, png, etc.)
        conf: Confidence threshold (0-1)
        iou: IOU threshold (0-1)
        save_image: Whether to save annotated image
        
    Returns:
        Detection results with bounding boxes and classes
    """
    
    # Check file size
    max_size = config['api'].get('max_upload_size', 10485760)
    contents = await file.read()
    if len(contents) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {max_size} bytes"
        )
    
    # Check file extension
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_extensions}"
        )
    
    try:
        # Read image
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Run detection
        det = get_detector()
        results = det.detect_image(
            image_path=image_np,
            conf=conf,
            iou=iou
        )
        
        # Save annotated image if requested
        image_url = None
        if save_image:
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            output_filename = f"detection_{timestamp}_{unique_id}.jpg"
            output_path = OUTPUT_DIR / output_filename
            
            # Save results
            det.save_results(
                detection_results=results,
                image_path=image_np,
                output_path=output_path,
                save_json=True
            )
            
            image_url = f"/results/{output_filename}"
        
        return DetectionResponse(
            success=True,
            num_detections=results['num_detections'],
            detections=results['detections'],
            image_shape=results['image_shape'],
            image_url=image_url,
            message="Detection successful"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )


@app.post("/detect-video", tags=["Detection"])
async def detect_video(
    file: UploadFile = File(...),
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    background_tasks: BackgroundTasks = None
):
    """
    Detect traffic lights in an uploaded video
    
    Args:
        file: Video file (mp4, avi, etc.)
        conf: Confidence threshold
        iou: IOU threshold
        background_tasks: Background task handler
        
    Returns:
        Processing status and result location
    """
    
    # Check file extension
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_extensions}"
        )
    
    try:
        # Save uploaded video temporarily
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        temp_video_path = OUTPUT_DIR / f"temp_{timestamp}_{unique_id}{file_ext}"
        output_video_path = OUTPUT_DIR / f"output_{timestamp}_{unique_id}.mp4"
        
        with open(temp_video_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        
        # Run detection
        det = get_detector()
        results = det.detect_video(
            video_path=temp_video_path,
            output_path=output_video_path,
            show=False,
            conf=conf,
            iou=iou
        )
        
        # Clean up temp file
        if temp_video_path.exists():
            temp_video_path.unlink()
        
        return JSONResponse(content={
            "success": True,
            "message": "Video processing complete",
            "total_frames": len(results),
            "video_url": f"/results/{output_video_path.name}",
            "summary": {
                "frames_with_detections": sum(1 for r in results if r['detections']['num_detections'] > 0)
            }
        })
    
    except Exception as e:
        # Clean up on error
        if temp_video_path.exists():
            temp_video_path.unlink()
        
        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {str(e)}"
        )


@app.get("/results/{filename}", tags=["Results"])
async def get_result_file(filename: str):
    """
    Get a result file (image or video)
    
    Args:
        filename: Name of the result file
        
    Returns:
        File response
    """
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=filename
    )


@app.delete("/results/{filename}", tags=["Results"])
async def delete_result_file(filename: str):
    """
    Delete a result file
    
    Args:
        filename: Name of the result file
        
    Returns:
        Deletion status
    """
    file_path = OUTPUT_DIR / filename
    json_path = file_path.with_suffix('.json')
    
    deleted_files = []
    
    if file_path.exists():
        file_path.unlink()
        deleted_files.append(filename)
    
    if json_path.exists():
        json_path.unlink()
        deleted_files.append(json_path.name)
    
    if not deleted_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    return JSONResponse(content={
        "success": True,
        "message": f"Deleted {len(deleted_files)} file(s)",
        "files": deleted_files
    })


@app.get("/list-results", tags=["Results"])
async def list_results():
    """
    List all result files
    
    Returns:
        List of result files
    """
    files = []
    
    for file_path in OUTPUT_DIR.glob("*"):
        if file_path.is_file() and not file_path.name.startswith('temp_'):
            files.append({
                "filename": file_path.name,
                "size": file_path.stat().st_size,
                "created": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                "url": f"/results/{file_path.name}"
            })
    
    return JSONResponse(content={
        "success": True,
        "count": len(files),
        "files": sorted(files, key=lambda x: x['created'], reverse=True)
    })


def main():
    """Run the API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Traffic Light Detection API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host address')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port number')
    parser.add_argument('--reload', action='store_true',
                       help='Enable auto-reload')
    
    args = parser.parse_args()
    
    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
