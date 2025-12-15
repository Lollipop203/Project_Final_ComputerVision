# -*- coding: utf-8 -*-
"""
Simple LISA to YOLO converter
"""

import os
import sys
import csv
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Class mapping
LISA_CLASSES = {
    'go': 2, 'goLeft': 2, 'goForward': 2,  # green
    'stop': 0, 'stopLeft': 0,                # red  
    'warning': 1, 'warningLeft': 1           # yellow
}

def convert_lisa(lisa_root, output_root):
    """Convert LISA dataset to YOLO format"""
    lisa_root = Path(lisa_root)
    output_root = Path(output_root)
    
    print("Converting LISA Traffic Light Dataset to YOLO Format")
    print("="*70)
    
    # Create output dirs
    for split in ['train', 'val', 'test']:
        (output_root / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_root / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Find BOX CSV files
    print("\nStep 1: Finding annotation files...")
    csv_files = [f for f in lisa_root.rglob("*BOX.csv")]
    print(f"Found {len(csv_files)} BOX CSV files")
    
    if not csv_files:
        print("ERROR: No CSV files found!")
        return
    
    # Parse all CSVs
    print("\nStep 2: Parsing annotations...")
    all_annotations = {}
    
    for csv_path in csv_files:
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')
                
                for row in reader:
                    filename = row.get('Filename', '').strip()
                    tag = row.get('Annotation tag', '').strip()
                    
                    if not filename or tag not in LISA_CLASSES:
                        continue
                    
                    try:
                        x1 = int(row.get('Upper left corner X', 0))
                        y1 = int(row.get('Upper left corner Y', 0))
                        x2 = int(row.get('Lower right corner X', 0))
                        y2 = int(row.get('Lower right corner Y', 0))
                    except:
                        continue
                    
                    class_id = LISA_CLASSES[tag]
                   
                    if filename not in all_annotations:
                        all_annotations[filename] = []
                    
                    all_annotations[filename].append({
                        'class_id': class_id,
                        'bbox': [x1, y1, x2, y2]
                    })
        except Exception as e:
            print(f"Warning: Failed to parse {csv_path}: {e}")
    
    print(f"Total images with annotations: {len(all_annotations)}")
    
    if not all_annotations:
        print("ERROR: No annotations found!")
        return
    
    # Split dataset
    image_list = list(all_annotations.keys())
    np.random.shuffle(image_list)
    
    n_total = len(image_list)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.2)
    
    splits = {
        'train': image_list[:n_train],
        'val': image_list[n_train:n_train+n_val],
        'test': image_list[n_train+n_val:]
    }
    
    print(f"\nDataset split: train={n_train}, val={n_val}, test={n_total-n_train-n_val}")
    
    # Convert each split
    print("\nStep 3: Converting images and labels...")
    stats = {'train': 0, 'val': 0, 'test': 0}
    
    for split_name, filenames in splits.items():
        print(f"\nProcessing {split_name} split ({len(filenames)} images)...")
        
        for filename in tqdm(filenames, desc=split_name):
            # Find image
            base_name = Path(filename).name
            img_path = None
            
            for candidate in lisa_root.rglob(base_name):
                if candidate.is_file():
                    img_path = candidate
                    break
            
            if img_path is None:
                continue
            
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # Copy image
            out_img = output_root / split_name / 'images' / img_path.name
            shutil.copy2(img_path, out_img)
            
            # Create label
            label_path = output_root / split_name / 'labels' / f"{img_path.stem}.txt"
            
            with open(label_path, 'w') as f:
                for ann in all_annotations[filename]:
                    cls_id = ann['class_id']
                    x1, y1, x2, y2 = ann['bbox']
                    
                    # Convert to YOLO format
                    cx = (x1 + x2) / 2.0 / w
                    cy = (y1 + y2) / 2.0 / h
                    bw = (x2 - x1) / float(w)
                    bh = (y2 - y1) / float(h)
                    
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            
            stats[split_name] += 1
    
    print("\n" + "="*70)
    print("Conversion Complete!")
    print(f"Train: {stats['train']} images")
    print(f"Val: {stats['val']} images") 
    print(f"Test: {stats['test']} images")
    print(f"Total: {sum(stats.values())} images")
    
    # Create dataset.yaml
    yaml_content = f"""# LISA Traffic Light Dataset - YOLO Format

path: {output_root.absolute()}
train: train/images
val: val/images
test: test/images

names:
  0: red
  1: yellow
  2: green
  3: off

nc: 4
"""
    
    yaml_path = output_root / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nCreated dataset.yaml: {yaml_path}")
    print("Done!")

if __name__ == "__main__":
    lisa_root = sys.argv[1] if len(sys.argv) > 1 else "data/raw/archive"
    output_root = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
    
    convert_lisa(lisa_root, output_root)
