# Data Directory

This directory contains the datasets for traffic light detection.

## Structure

```
data/
├── raw/                  # Original, unprocessed data
├── processed/            # Processed data ready for training
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── annotations/          # Annotation files
└── dataset.yaml          # YOLO dataset configuration
```

## Dataset Format

The project uses YOLO format for annotations:
- Each image has a corresponding `.txt` file with the same name
- Each line in the `.txt` file represents one object
- Format: `class_id center_x center_y width height` (all normalized 0-1)

## Classes

- 0: red
- 1: yellow
- 2: green
- 3: off

## Getting Data

### Option 1: Download from Roboflow
```bash
# Example: Download from Roboflow
# pip install roboflow
# from roboflow import Roboflow
# rf = Roboflow(api_key="YOUR_API_KEY")
# project = rf.workspace().project("traffic-lights")
# dataset = project.version(1).download("yolov8")
```

### Option 2: Prepare Your Own Data

1. Collect images containing traffic lights
2. Annotate using tools like:
   - [LabelImg](https://github.com/tzutalin/labelImg)
   - [CVAT](https://github.com/opencv/cvat)
   - [Roboflow](https://roboflow.com/)
3. Export in YOLO format
4. Organize into train/val/test splits (e.g., 70/20/10)

### Option 3: Use Public Datasets

- [LISA Traffic Light Dataset](https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset)
- [Bosch Small Traffic Lights Dataset](https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset)
- Search "traffic light detection" on [Roboflow Universe](https://universe.roboflow.com/)

## Data Augmentation

The project includes automated data augmentation through:
- `src/preprocessing/augmentation.py` for custom augmentation
- YOLOv8 built-in augmentation during training

## Notes

- Ensure balanced class distribution
- Recommended minimum: 500+ images per class
- Include diverse lighting conditions (day/night, rain, etc.)
- Various traffic light types (vertical, horizontal, arrow lights)
