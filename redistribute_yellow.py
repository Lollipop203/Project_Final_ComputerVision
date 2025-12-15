"""
Redistribute yellow images from train to valid and test sets
to ensure all splits contain all 3 traffic light classes.

Target split:
- Train: 70% (~130 images)
- Valid: 19% (~35 images)
- Test: 11% (~21 images)
"""

import os
import shutil
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Paths
MERGED_DIR = Path(r"c:\ComputerVision\TrafficLights\data\processed\merged_traffic_light")

# Split ratios
TRAIN_RATIO = 0.70
VALID_RATIO = 0.19
TEST_RATIO = 0.11

def get_yellow_files():
    """Get all yellow image files from train set"""
    train_images = MERGED_DIR / 'train' / 'images'
    yellow_images = list(train_images.glob('yellow_*.jpg')) + \
                   list(train_images.glob('yellow_*.png')) + \
                   list(train_images.glob('yellow_*.jpeg'))
    return yellow_images

def move_files(image_files, target_split):
    """Move image and corresponding label files to target split"""
    moved_count = 0
    
    for img_file in image_files:
        # Get corresponding label file
        label_name = img_file.stem + '.txt'
        label_file = MERGED_DIR / 'train' / 'labels' / label_name
        
        # Target directories
        target_img_dir = MERGED_DIR / target_split / 'images'
        target_lbl_dir = MERGED_DIR / target_split / 'labels'
        
        # Move image
        if img_file.exists():
            shutil.move(str(img_file), str(target_img_dir / img_file.name))
            moved_count += 1
        
        # Move label
        if label_file.exists():
            shutil.move(str(label_file), str(target_lbl_dir / label_name))
    
    return moved_count

def verify_distribution():
    """Verify the new distribution of images"""
    print("\n" + "=" * 60)
    print("📊 Dataset Distribution After Redistribution")
    print("=" * 60)
    
    for split in ['train', 'valid', 'test']:
        img_dir = MERGED_DIR / split / 'images'
        
        # Count total images
        total_images = len(list(img_dir.glob('*')))
        
        # Count yellow images
        yellow_images = len(list(img_dir.glob('yellow_*')))
        
        # Count other images (Green/Red)
        other_images = total_images - yellow_images
        
        print(f"\n{split.upper()}:")
        print(f"  Total images:  {total_images:3d}")
        print(f"  Yellow images: {yellow_images:3d} ({yellow_images/total_images*100:.1f}%)")
        print(f"  Green/Red:     {other_images:3d} ({other_images/total_images*100:.1f}%)")

def count_class_annotations(split):
    """Count annotations per class in a split"""
    lbl_dir = MERGED_DIR / split / 'labels'
    class_counts = {0: 0, 1: 0, 2: 0}
    
    for lbl_file in lbl_dir.glob('*.txt'):
        with open(lbl_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    if cls in class_counts:
                        class_counts[cls] += 1
    
    return class_counts

def verify_class_distribution():
    """Verify class distribution in each split"""
    print("\n" + "=" * 60)
    print("📈 Class Annotation Distribution")
    print("=" * 60)
    
    for split in ['train', 'valid', 'test']:
        counts = count_class_annotations(split)
        total = sum(counts.values())
        
        print(f"\n{split.upper()}:")
        print(f"  Green (0):  {counts[0]:3d} ({counts[0]/total*100:.1f}%)")
        print(f"  Red (1):    {counts[1]:3d} ({counts[1]/total*100:.1f}%)")
        print(f"  Yellow (2): {counts[2]:3d} ({counts[2]/total*100:.1f}%)")
        print(f"  Total:      {total:3d}")

def main():
    print("=" * 60)
    print("🔄 Redistributing Yellow Images")
    print("=" * 60)
    
    # Get all yellow images from train
    yellow_images = get_yellow_files()
    total_yellow = len(yellow_images)
    
    print(f"\n📦 Found {total_yellow} yellow images in train set")
    
    # Calculate split sizes
    valid_size = int(total_yellow * VALID_RATIO)
    test_size = int(total_yellow * TEST_RATIO)
    train_size = total_yellow - valid_size - test_size
    
    print(f"\n📊 Target distribution:")
    print(f"  Train: {train_size} images ({train_size/total_yellow*100:.1f}%)")
    print(f"  Valid: {valid_size} images ({valid_size/total_yellow*100:.1f}%)")
    print(f"  Test:  {test_size} images ({test_size/total_yellow*100:.1f}%)")
    
    # Shuffle images randomly
    random.shuffle(yellow_images)
    
    # Split images
    valid_images = yellow_images[:valid_size]
    test_images = yellow_images[valid_size:valid_size + test_size]
    # Remaining stay in train
    
    print(f"\n🔄 Moving images...")
    
    # Move to valid
    valid_moved = move_files(valid_images, 'valid')
    print(f"  ✓ Moved {valid_moved} images to valid")
    
    # Move to test
    test_moved = move_files(test_images, 'test')
    print(f"  ✓ Moved {test_moved} images to test")
    
    # Verify distribution
    verify_distribution()
    verify_class_distribution()
    
    print("\n" + "=" * 60)
    print("✅ Redistribution completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
