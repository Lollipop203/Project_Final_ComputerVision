# LISA Dataset Conversion Summary

## Dataset Information

**Source**: LISA Traffic Light Dataset  
**Format**: Converted to YOLO format  
**Total Annotated Images**: 36,265

## Dataset Split

- **Training Set**: 25,385 images (70%)
- **Validation Set**: 7,253 images (20%)  
- **Test Set**: 3,627 images (10%)

## Classes

| Class ID | Name | LISA Mappings |
|----------|------|---------------|
| 0 | red | stop, stopLeft |
| 1 | yellow | warning, warningLeft |
| 2 | green | go, goLeft, goForward |
| 3 | off | (not mapped from LISA) |

## Conversion Process

### Status: IN PROGRESS ✓

The conversion script `scripts/convert_simple.py` is currently processing the dataset:

1. **Annotation Parsing**: Completed - Found 24 BOX CSV annotation files
2. **Image Processing**: In progress - Converting and copying images with YOLO format labels
3. **Estimated Time**: ~30-40 minutes for full dataset

### Output Structure

```
data/processed/
├── dataset.yaml          # YOLO training configuration
├── train/
│   ├── images/          # Training images
│   └── labels/          # Training labels (YOLO format)
├── val/
│   ├── images/          # Validation images
│   └── labels/          # Validation labels
└── test/
    ├── images/          # Test images
    └── labels/          # Test labels
```

## YOLO Label Format

Each label file contains one line per object:
```
<class_id> <center_x> <center_y> <width> <height>
```

All coordinates are normalized to [0-1] range.

## Next Steps

After conversion completes:

1. **Verify Dataset**:
   ```bash
   # Check image counts
   python scripts/verify_dataset.py
   ```

2. **Start Training**:
   ```bash
   python src/training/train.py --data data/processed/dataset.yaml --epochs 100
   ```

3. **Monitor Training**:
   - Check `runs/train/` for training logs
   - Use Tensorboard: `tensorboard --logdir runs/train`

## Notes

- LISA BOX annotations contain bounding box coordinates
- Images are located in nested directory structure (dayTrain/dayTrain/dayClip*/frames/)
- CSV files use semicolon (;) delimiter
- Some LISA classes (goLeft, goForward, etc.) are mapped to single "green" class for simplification
