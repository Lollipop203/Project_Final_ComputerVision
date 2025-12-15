"""
Merge two traffic light datasets:
- Vietnam Traffic Light (Green=0, Red=1)
- Traffic Light 3 (Yellow currently labeled as 0, needs to be 2)

Output: Unified dataset with 3 classes [Green, Red, Yellow]
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Paths
BASE_DIR = Path(r"c:\ComputerVision\TrafficLights\data\processed")
DATASET1 = BASE_DIR / "Vietnam Traffic Light.v1i.yolov8"  # Green + Red
DATASET2 = BASE_DIR / "Traffic Light 3.v3i.yolov8"  # Yellow (needs relabeling)
OUTPUT_DIR = BASE_DIR / "merged_traffic_light"

# Class mapping
CLASSES = ['Green', 'Red', 'Yellow']

def create_output_structure():
    """Create output directory structure"""
    for split in ['train', 'valid', 'test']:
        (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)
    print(f"✓ Created output directory: {OUTPUT_DIR}")

def copy_dataset1():
    """Copy Vietnam Traffic Light dataset (Green + Red) as-is"""
    print("\n📦 Copying Vietnam Traffic Light dataset (Green + Red)...")
    
    for split in ['train', 'valid', 'test']:
        src_images = DATASET1 / split / 'images'
        src_labels = DATASET1 / split / 'labels'
        dst_images = OUTPUT_DIR / split / 'images'
        dst_labels = OUTPUT_DIR / split / 'labels'
        
        if not src_images.exists():
            print(f"  ⚠ Skipping {split} (not found)")
            continue
            
        # Copy images
        image_files = list(src_images.glob('*'))
        for img_file in tqdm(image_files, desc=f"  {split}/images"):
            shutil.copy2(img_file, dst_images / img_file.name)
        
        # Copy labels (no modification needed - already Green=0, Red=1)
        if src_labels.exists():
            label_files = list(src_labels.glob('*.txt'))
            for lbl_file in tqdm(label_files, desc=f"  {split}/labels"):
                shutil.copy2(lbl_file, dst_labels / lbl_file.name)
        
        print(f"  ✓ Copied {len(image_files)} images from {split}")

def copy_and_relabel_dataset2():
    """Copy Traffic Light 3 dataset (Yellow) and relabel from 0 to 2"""
    print("\n📦 Copying Traffic Light 3 dataset (Yellow) with relabeling...")
    
    # Only has train split
    src_images = DATASET2 / 'train' / 'images'
    src_labels = DATASET2 / 'train' / 'labels'
    dst_images = OUTPUT_DIR / 'train' / 'images'
    dst_labels = OUTPUT_DIR / 'train' / 'labels'
    
    if not src_images.exists():
        print("  ⚠ Yellow dataset train folder not found!")
        return
    
    # Copy images
    image_files = list(src_images.glob('*'))
    for img_file in tqdm(image_files, desc="  train/images"):
        # Rename to avoid conflicts
        new_name = f"yellow_{img_file.name}"
        shutil.copy2(img_file, dst_images / new_name)
    
    # Copy and relabel labels (0 -> 2)
    if src_labels.exists():
        label_files = list(src_labels.glob('*.txt'))
        relabeled_count = 0
        
        for lbl_file in tqdm(label_files, desc="  train/labels (relabeling)"):
            new_name = f"yellow_{lbl_file.name}"
            dst_file = dst_labels / new_name
            
            # Read, relabel, and write
            with open(lbl_file, 'r') as f:
                lines = f.readlines()
            
            with open(dst_file, 'w') as f:
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Change class from 0 to 2
                        parts[0] = '2'
                        relabeled_count += 1
                        f.write(' '.join(parts) + '\n')
                    else:
                        f.write(line)
        
        print(f"  ✓ Copied {len(image_files)} images and relabeled {relabeled_count} annotations")

def create_data_yaml():
    """Create unified data.yaml file"""
    yaml_content = f"""# Merged Traffic Light Dataset
# Classes: Green, Red, Yellow

path: {OUTPUT_DIR.as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 3
names: {CLASSES}
"""
    
    yaml_path = OUTPUT_DIR / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✓ Created data.yaml at {yaml_path}")

def verify_dataset():
    """Verify the merged dataset"""
    print("\n🔍 Verifying merged dataset...")
    
    for split in ['train', 'valid', 'test']:
        img_dir = OUTPUT_DIR / split / 'images'
        lbl_dir = OUTPUT_DIR / split / 'labels'
        
        if img_dir.exists():
            img_count = len(list(img_dir.glob('*')))
            lbl_count = len(list(lbl_dir.glob('*.txt'))) if lbl_dir.exists() else 0
            print(f"  {split:6s}: {img_count:4d} images, {lbl_count:4d} labels")
    
    # Sample a few labels to verify class distribution
    print("\n📊 Sample label verification:")
    train_labels = list((OUTPUT_DIR / 'train' / 'labels').glob('*.txt'))[:10]
    class_counts = {0: 0, 1: 0, 2: 0}
    
    for lbl_file in train_labels:
        with open(lbl_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    if cls in class_counts:
                        class_counts[cls] += 1
    
    print(f"  Green (0): {class_counts[0]} annotations")
    print(f"  Red   (1): {class_counts[1]} annotations")
    print(f"  Yellow(2): {class_counts[2]} annotations")

def main():
    print("=" * 60)
    print("🚦 Traffic Light Dataset Merger")
    print("=" * 60)
    
    # Create output structure
    create_output_structure()
    
    # Copy datasets
    copy_dataset1()
    copy_and_relabel_dataset2()
    
    # Create data.yaml
    create_data_yaml()
    
    # Verify
    verify_dataset()
    
    print("\n" + "=" * 60)
    print("✅ Dataset merge completed successfully!")
    print(f"📁 Output location: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
