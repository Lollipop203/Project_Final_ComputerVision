# Quick Start Guide

## Installation

### 1. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac  
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- YOLOv8 (ultralytics)
- FastAPI & Uvicorn
- OpenCV
- And all other required packages

## Quick Usage

### Option 1: Command Line Interface

#### Detect from Image
```bash
python -m src.agent.detector --source path/to/image.jpg --output output/
```

#### Detect from Video
```bash
python -m src.agent.detector --source path/to/video.mp4 --output output/ --show
```

#### Detect from Webcam
```bash
python -m src.agent.detector --source 0
```

### Option 2: Python API

```python
from src.agent.detector import TrafficLightDetector

# Initialize detector
detector = TrafficLightDetector()

# Detect from image
results = detector.detect_image('path/to/image.jpg')
print(f"Found {results['num_detections']} traffic lights")

# Save results
detector.save_results(results, 'path/to/image.jpg', 'output/result.jpg')
```

### Option 3: REST API

#### Start Server
```bash
python src/api/app.py
# or
uvicorn src.api.app:app --reload
```

#### Test API
```bash
# Health check
curl http://localhost:8000/health

# Detect from image
curl -X POST "http://localhost:8000/detect" -F "file=@image.jpg"

# View API docs
# Open browser: http://localhost:8000/docs
```

## Training Your Own Model

### 1. Prepare Dataset

Organize your data in YOLO format:
```
data/processed/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

### 2. Run Training

```bash
# Using config file
python src/training/train.py --config configs/train_config.yaml

# Custom parameters
python src/training/train.py --data data/dataset.yaml --epochs 100 --batch 16
```

### 3. Evaluate Model

```bash
python src/training/evaluate.py --weights models/trained/best.pt
```

## Next Steps

1. **Add your data** to `data/raw/` or `data/processed/`
2. **Try examples** in `examples/basic_usage.py`
3. **Explore notebooks** in `notebooks/exploration.ipynb`
4. **Train custom model** with your dataset
5. **Deploy API** for production use

## Troubleshooting

### No GPU/CUDA
If you don't have CUDA, use `--device cpu`:
```bash
python -m src.agent.detector --source image.jpg --device cpu
```

### Missing Model
First run will download YOLOv8 pretrained weights automatically.

### Import Errors
Make sure you're in the project root and virtual environment is activated.

## Documentation

- Full API docs: `http://localhost:8000/docs` (when server is running)
- Dataset format: See `data/README.md`
- Main README: See root `README.md`
