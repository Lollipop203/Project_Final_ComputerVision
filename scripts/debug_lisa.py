"""
Debug script to check LISA dataset structure
"""
import csv
from pathlib import Path

lisa_root = Path("data/raw/archive")

# Find CSV file
csv_path = lisa_root / "Annotations" / "Annotations" / "dayTrain" / "dayClip1" / "frameAnnotationsBOX.csv"

print(f"Checking CSV: {csv_path}")
print(f"Exists: {csv_path.exists()}")
print()

if csv_path.exists():
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        
        print("CSV Headers:")
        print(reader.fieldnames)
        print()
        
        print("First 5 rows:")
        for i, row in enumerate(reader):
            if i >= 5:
                break
            print(f"\nRow {i+1}:")
            for key, value in row.items():
                print(f"  {key}: {value}")
            
            # Try to find this image
            filename = row.get('Filename', '').strip()
            if filename:
                print(f"\n  Searching for: {filename}")
                base_filename = Path(filename).name
                print(f"  Base filename: {base_filename}")
                
                # Search for image
                found = False
                for img_path in lisa_root.rglob(base_filename):
                    if img_path.is_file():
                        print(f"  FOUND: {img_path.relative_to(lisa_root)}")
                        found = True
                        break
                
                if not found:
                    print(f"  NOT FOUND!")
