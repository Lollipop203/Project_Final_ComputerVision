"""
Convert LISA Traffic Light Dataset to YOLO Format

LISA dataset structure (typical):
- Annotations/: CSV files with bounding boxes
- dayTrain/: Training images
- dayTest/: Test images
- nightTrain/: Night training images
- nightTest/: Night test images

YOLO format:
- Each image has a .txt file with same name
- Format: class_id center_x center_y width height (normalized 0-1)
"""

import os
import sys
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from tqdm import tqdm
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class LISAToYOLOConverter:
    """Convert LISA Traffic Light Dataset to YOLO format"""
    
    # LISA class mapping to our classes
    LISA_TO_YOLO_CLASSES = {
        'go': 2,           # green
        'goLeft': 2,       # green (left arrow)
        'goForward': 2,    # green (forward arrow)
        'stop': 0,         # red
        'stopLeft': 0,     # red (left arrow)
        'warning': 1,      # yellow
        'warningLeft': 1,  # yellow (left arrow)
    }
    
    YOLO_CLASSES = ['red', 'yellow', 'green', 'off']
    
    def __init__(
        self,
        lisa_root: str,
        output_root: str,
        train_split: float = 0.7,
        val_split: float = 0.2,
        test_split: float = 0.1
    ):
        """
        Initialize converter
        
        Args:
            lisa_root: Root directory of LISA dataset
            output_root: Output directory for YOLO format
            train_split: Training split ratio
            val_split: Validation split ratio
            test_split: Test split ratio
        """
        self.lisa_root = Path(lisa_root)
        self.output_root = Path(output_root)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        # Create output directories
        self.setup_output_dirs()
        
    def setup_output_dirs(self):
        """Create output directory structure"""
        for split in ['train', 'val', 'test']:
            (self.output_root / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_root / split / 'labels').mkdir(parents=True, exist_ok=True)
        print(f"Created output directories in {self.output_root}")
    
    def find_annotation_files(self) -> List[Path]:
        """Find all annotation CSV files"""
        annotation_files = []
        
        # Common locations for LISA annotations
        possible_paths = [
            self.lisa_root / 'Annotations',
            self.lisa_root / 'annotations',
            self.lisa_root / 'labels',
        ]
        
        for path in possible_paths:
            if path.exists():
                csv_files = list(path.glob('*.csv'))
                annotation_files.extend(csv_files)
        
        # Also search in root
        annotation_files.extend(list(self.lisa_root.glob('*.csv')))
        
        return list(set(annotation_files))  # Remove duplicates
    
    def parse_lisa_annotation(self, csv_path: Path) -> Dict:
        """
        Parse LISA annotation CSV file
        
        Returns:
            Dictionary mapping image paths to list of annotations
        """
        annotations = {}
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')
                
                for row in reader:
                    # LISA CSV format: Filename;Annotation tag;Upper left corner X;Upper left corner Y;Lower right corner X;Lower right corner Y;Origin file;Origin frame number;Origin track;Origin track frame number
                    filename = row.get('Filename', '').strip()
                    if not filename:
                        continue
                    
                    tag = row.get('Annotation tag', '').strip()
                    
                    # Get coordinates
                    try:
                        x1 = int(row.get('Upper left corner X', 0))
                        y1 = int(row.get('Upper left corner Y', 0))
                        x2 = int(row.get('Lower right corner X', 0))
                        y2 = int(row.get('Lower right corner Y', 0))
                    except (ValueError, KeyError) as e:
                        print(f"Warning: Failed to parse coordinates in {csv_path}: {e}")
                        continue
                    
                    # Map LISA class to YOLO class
                    if tag not in self.LISA_TO_YOLO_CLASSES:
                        # Skip unknown classes
                        continue
                    
                    class_id = self.LISA_TO_YOLO_CLASSES[tag]
                    
                    # Store annotation
                    if filename not in annotations:
                        annotations[filename] = []
                    
                    annotations[filename].append({
                        'class_id': class_id,
                        'bbox': [x1, y1, x2, y2]
                    })
        
        except Exception as e:
            print(f"Warning: Failed to parse {csv_path}: {e}")
        
        return annotations
    
    def convert_bbox_to_yolo(
        self,
        bbox: List[int],
        img_width: int,
        img_height: int
    ) -> Tuple[float, float, float, float]:
        """
        Convert bounding box to YOLO format
        
        Args:
            bbox: [x1, y1, x2, y2]
            img_width: Image width
            img_height: Image height
            
        Returns:
            (center_x, center_y, width, height) normalized to 0-1
        """
        x1, y1, x2, y2 = bbox
        
        # Calculate center and dimensions
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        # Normalize
        center_x /= img_width
        center_y /= img_height
        width /= img_width
        height /= img_height
        
        # Clip to valid range
        center_x = max(0, min(1, center_x))
        center_y = max(0, min(1, center_y))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        return center_x, center_y, width, height
    
    def find_image_file(self, filename: str) -> Path:
        """Find image file in LISA dataset"""
        # Extract just the filename without path
        base_filename = Path(filename).name
        
        # Search recursively for the image
        for img_path in self.lisa_root.rglob(base_filename):
            if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                return img_path
        
        # If not found, try alternative path structures
        # LISA CSVs reference "dayTraining" but actual dir is "dayTrain/dayTrain"
        # Try to extract clip and build path
        parts = Path(filename).parts
        if len(parts) >= 2:
            # Pattern: dayTraining/dayClip1/frames/image.jpg
            # Actual: dayTrain/dayTrain/dayClip1/frames/image.jpg
            if 'dayTraining' in parts:
                # Replace dayTraining with dayTrain/dayTrain
                new_parts = ['dayTrain', 'dayTrain'] + list(parts[1:])
                test_path = self.lisa_root
                for part in new_parts:
                    test_path = test_path / part
                if test_path.exists():
                    return test_path
            
            if 'nightTraining' in parts:
                # Replace nightTraining with nightTrain/nightTrain  
                new_parts = ['nightTrain', 'nightTrain'] + list(parts[1:])
                test_path = self.lisa_root
                for part in new_parts:
                    test_path = test_path / part
                if test_path.exists():
                    return test_path
        
        return None
    
    def convert_dataset(self):
        """Convert entire LISA dataset to YOLO format"""
        print("=" * 70)
        print("Converting LISA Traffic Light Dataset to YOLO Format")
        print("=" * 70)
        
        # Find annotation files
        print("\n1. Finding annotation files...")
        annotation_files = self.find_annotation_files()
        
        # Filter to only BOX annotations (not BULB)
        annotation_files = [f for f in annotation_files if 'BOX' in f.name.upper()]
        print(f"   Found {len(annotation_files)} BOX annotation file(s)")
        
        if not annotation_files:
            print("❌ No annotation files found!")
            print("Please check LISA dataset structure.")
            return
        
        # Parse all annotations
        print("\n2. Parsing annotations...")
        all_annotations = {}
        for csv_path in annotation_files:
            print(f"   Processing: {csv_path.relative_to(self.lisa_root)}")
            annotations = self.parse_lisa_annotation(csv_path)
            print(f"      → {len(annotations)} images with annotations")
            all_annotations.update(annotations)
        
        print(f"\n   Total unique images with annotations: {len(all_annotations)}")
        
        if len(all_annotations) == 0:
            print("❌ No valid annotations found!")
            return
        
        # Convert and split dataset
        print("\n3. Converting to YOLO format and splitting dataset...")
        
        # Get all image filenames
        image_filenames = list(all_annotations.keys())
        np.random.shuffle(image_filenames)
        
        # Calculate split indices
        n_total = len(image_filenames)
        n_train = int(n_total * self.train_split)
        n_val = int(n_total * self.val_split)
        
        splits = {
            'train': image_filenames[:n_train],
            'val': image_filenames[n_train:n_train + n_val],
            'test': image_filenames[n_train + n_val:]
        }
        
        # Process each split
        stats = {'train': 0, 'val': 0, 'test': 0}
        
        for split_name, filenames in splits.items():
            print(f"\n   Processing {split_name} split ({len(filenames)} images)...")
            
            for filename in tqdm(filenames, desc=f"   {split_name}"):
                # Find image file
                img_path = self.find_image_file(filename)
                if img_path is None:
                    continue
                
                # Read image to get dimensions
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                img_height, img_width = img.shape[:2]
                
                # Copy image
                output_img_path = self.output_root / split_name / 'images' / img_path.name
                shutil.copy2(img_path, output_img_path)
                
                # Create YOLO label file
                label_path = self.output_root / split_name / 'labels' / f"{img_path.stem}.txt"
                
                with open(label_path, 'w') as f:
                    for annotation in all_annotations[filename]:
                        class_id = annotation['class_id']
                        bbox = annotation['bbox']
                        
                        # Convert to YOLO format
                        center_x, center_y, width, height = self.convert_bbox_to_yolo(
                            bbox, img_width, img_height
                        )
                        
                        # Write to file
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                
                stats[split_name] += 1
        
        # Print summary
        print("\n" + "=" * 70)
        print("Conversion Complete!")
        print("=" * 70)
        print(f"\nDataset split:")
        print(f"  Training:   {stats['train']} images")
        print(f"  Validation: {stats['val']} images")
        print(f"  Test:       {stats['test']} images")
        print(f"  Total:      {sum(stats.values())} images")
        print(f"\nOutput directory: {self.output_root}")
    
    def create_dataset_yaml(self):
        """Create dataset.yaml file for YOLO training"""
        yaml_content = f"""# LISA Traffic Light Dataset - YOLO Format
# Auto-generated by convert_lisa_to_yolo.py

path: {self.output_root.absolute()}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images  # val images (relative to 'path')
test: test/images  # test images (optional)

# Classes
names:
  0: red
  1: yellow
  2: green
  3: off

# Number of classes
nc: 4
"""
        
        yaml_path = self.output_root / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n✓ Created dataset.yaml: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert LISA Dataset to YOLO Format')
    parser.add_argument('--lisa-root', type=str, required=True,
                       help='Root directory of LISA dataset')
    parser.add_argument('--output', type=str, default='data/processed',
                       help='Output directory for YOLO format')
    parser.add_argument('--train-split', type=float, default=0.7,
                       help='Training split ratio')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--test-split', type=float, default=0.1,
                       help='Test split ratio')
    
    args = parser.parse_args()
    
    # Validate splits
    total = args.train_split + args.val_split + args.test_split
    if abs(total - 1.0) > 0.01:
        print(f"❌ Splits must sum to 1.0 (got {total})")
        return
    
    # Create converter
    converter = LISAToYOLOConverter(
        lisa_root=args.lisa_root,
        output_root=args.output,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split
    )
    
    # Convert dataset
    converter.convert_dataset()
    
    # Create dataset.yaml
    converter.create_dataset_yaml()
    
    print("\n✓ All done! You can now train with:")
    print(f"  python src/training/train.py --data {args.output}/dataset.yaml")


if __name__ == "__main__":
    main()
