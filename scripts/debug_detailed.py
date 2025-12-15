"""
Detailed debugging script for LISA dataset conversion
"""
from pathlib import Path
import csv

project_root = Path("c:/ComputerVision/TrafficLights")
lisa_root = project_root / "data/raw/archive"

print("="*70)
print("LISA Dataset Debugging")
print("="*70)

# Step 1: Find annotation files
print("\n1. Finding BOX annotation CSV files...")
annotation_files = []
for csv_path in lisa_root.rglob("*BOX.csv"):
    annotation_files.append(csv_path)
    
print(f"   Found {len(annotation_files)} BOX CSVs")

# Step 2: Parse one CSV in detail
if annotation_files:
    csv_path = annotation_files[0]
    print(f"\n2. Parsing first CSV: {csv_path.relative_to(lisa_root)}")
    
    annotations = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        
        for i, row in enumerate(reader):
            if i >= 3:  # Just check first 3
                break
                
            filename = row.get('Filename', '').strip()
            tag = row.get('Annotation tag', '').strip()
            
            print(f"\n   Row {i+1}:")
            print(f"     Filename: {filename}")
            print(f"     Tag: {tag}")
            
            # Try to find image
            base_filename = Path(filename).name
            print(f"     Base filename: {base_filename}")
            
            # Method 1: Recursive glob
            found_by_glob = False
            for img_path in lisa_root.rglob(base_filename):
                if img_path.is_file():
                    print(f"     FOUND (glob): {img_path.relative_to(lisa_root)}")
                    found_by_glob = True
                    break
            
            if not found_by_glob:
                print(f"     NOT FOUND by glob")
            
            # Method 2: Path manipulation
            parts = Path(filename).parts
            print(f"     Path parts: {parts}")
            
            if 'dayTraining' in parts:
                new_parts = ['dayTrain', 'dayTrain'] + list(parts[1:])
                test_path = lisa_root
                for part in new_parts:
                    test_path = test_path / part
                print(f"     Test path: {test_path}")
                print(f"     Exists: {test_path.exists()}")

# Step 3: Count total available JPGs
print(f"\n3. Counting JPG files in dataset...")
all_jpgs = list(lisa_root.rglob("*.jpg"))
print(f"   Total JPG files: {len(all_jpgs)}")
if all_jpgs:
    print(f"   First 3 examples:")
    for jpg in all_jpgs[:3]:
        print(f"     - {jpg.relative_to(lisa_root)}")
