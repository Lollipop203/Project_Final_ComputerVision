"""
Data Augmentation Module for Traffic Light Detection
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import yaml
from pathlib import Path


class TrafficLightAugmentation:
    """Data augmentation pipeline for traffic light images"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize augmentation pipeline
        
        Args:
            config: Augmentation configuration dictionary
        """
        if config is None:
            config = self.get_default_config()
        
        self.config = config
        self.train_transform = self._create_train_transform()
        self.val_transform = self._create_val_transform()
    
    @staticmethod
    def get_default_config() -> Dict:
        """Get default augmentation configuration"""
        return {
            'image_size': 640,
            'brightness_limit': 0.2,
            'contrast_limit': 0.2,
            'hue_shift_limit': 20,
            'sat_shift_limit': 30,
            'val_shift_limit': 20,
            'blur_limit': 3,
            'noise_var_limit': (10.0, 50.0),
            'rotate_limit': 15,
            'scale_limit': 0.1,
            'flip_prob': 0.5
        }
    
    def _create_train_transform(self) -> A.Compose:
        """
        Create training augmentation pipeline
        
        Returns:
            Albumentations composition
        """
        return A.Compose([
            # Geometric transformations
            A.HorizontalFlip(p=self.config.get('flip_prob', 0.5)),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=self.config.get('scale_limit', 0.1),
                rotate_limit=self.config.get('rotate_limit', 15),
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT
            ),
            
            # Color augmentations
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=self.config.get('brightness_limit', 0.2),
                    contrast_limit=self.config.get('contrast_limit', 0.2),
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=self.config.get('hue_shift_limit', 20),
                    sat_shift_limit=self.config.get('sat_shift_limit', 30),
                    val_shift_limit=self.config.get('val_shift_limit', 20),
                    p=1.0
                ),
            ], p=0.8),
            
            # Blur and noise
            A.OneOf([
                A.MotionBlur(blur_limit=self.config.get('blur_limit', 3), p=1.0),
                A.GaussianBlur(blur_limit=self.config.get('blur_limit', 3), p=1.0),
                A.MedianBlur(blur_limit=self.config.get('blur_limit', 3), p=1.0),
            ], p=0.3),
            
            A.OneOf([
                A.GaussNoise(var_limit=self.config.get('noise_var_limit', (10.0, 50.0)), p=1.0),
                A.ISONoise(p=1.0),
            ], p=0.2),
            
            # Weather effects (useful for traffic light scenarios)
            A.OneOf([
                A.RandomRain(p=1.0),
                A.RandomFog(p=1.0),
                A.RandomShadow(p=1.0),
            ], p=0.1),
            
            # Final adjustments
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
    
    def _create_val_transform(self) -> A.Compose:
        """
        Create validation augmentation pipeline (minimal)
        
        Returns:
            Albumentations composition
        """
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
    
    def augment_train(
        self,
        image: np.ndarray,
        bboxes: list,
        class_labels: list
    ) -> Tuple[np.ndarray, list, list]:
        """
        Apply training augmentation
        
        Args:
            image: Input image
            bboxes: List of bounding boxes in YOLO format
            class_labels: List of class labels
            
        Returns:
            Augmented image, bboxes, and labels
        """
        transformed = self.train_transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        
        return (
            transformed['image'],
            transformed['bboxes'],
            transformed['class_labels']
        )
    
    def augment_val(
        self,
        image: np.ndarray,
        bboxes: list,
        class_labels: list
    ) -> Tuple[np.ndarray, list, list]:
        """
        Apply validation augmentation
        
        Args:
            image: Input image
            bboxes: List of bounding boxes in YOLO format
            class_labels: List of class labels
            
        Returns:
            Augmented image, bboxes, and labels
        """
        transformed = self.val_transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        
        return (
            transformed['image'],
            transformed['bboxes'],
            transformed['class_labels']
        )


class YOLOAugmentation:
    """
    Simple augmentation wrapper compatible with YOLOv8 training
    This uses YOLO's built-in augmentation during training
    """
    
    @staticmethod
    def get_yolo_augmentation_config() -> Dict:
        """
        Get augmentation config for YOLOv8 training
        
        Returns:
            Dictionary with YOLO augmentation parameters
        """
        return {
            # HSV augmentation
            'hsv_h': 0.015,  # HSV-Hue augmentation (fraction)
            'hsv_s': 0.7,    # HSV-Saturation augmentation (fraction)
            'hsv_v': 0.4,    # HSV-Value augmentation (fraction)
            
            # Geometric augmentation
            'degrees': 0.0,      # Rotation (+/- deg)
            'translate': 0.1,    # Translation (+/- fraction)
            'scale': 0.5,        # Scale (+/- gain)
            'shear': 0.0,        # Shear (+/- deg)
            'perspective': 0.0,  # Perspective (+/- fraction)
            'flipud': 0.0,       # Vertical flip probability
            'fliplr': 0.5,       # Horizontal flip probability
            
            # Advanced augmentation
            'mosaic': 1.0,       # Mosaic augmentation probability
            'mixup': 0.0,        # Mixup augmentation probability
            'copy_paste': 0.0,   # Copy-paste augmentation probability
        }


def augment_single_image(
    image_path: str,
    output_path: str,
    num_variations: int = 5
) -> None:
    """
    Create multiple augmented versions of a single image
    
    Args:
        image_path: Path to input image
        output_path: Directory to save augmented images
        num_variations: Number of augmented versions to create
    """
    augmentor = TrafficLightAugmentation()
    
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original
    original_name = Path(image_path).stem
    cv2.imwrite(
        str(output_dir / f"{original_name}_original.jpg"),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )
    
    # Create augmented versions
    for i in range(num_variations):
        # Apply augmentation (without bboxes for visualization)
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.8),
            A.HueSaturationValue(p=0.8),
            A.GaussianBlur(p=0.3),
            A.GaussNoise(p=0.2),
        ])
        
        augmented = transform(image=image)['image']
        
        # Save augmented image
        cv2.imwrite(
            str(output_dir / f"{original_name}_aug_{i+1}.jpg"),
            cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
        )
    
    print(f"Created {num_variations} augmented versions in {output_dir}")


if __name__ == "__main__":
    # Example usage
    print("Traffic Light Augmentation Module")
    print("=" * 50)
    
    # Get default config
    augmentor = TrafficLightAugmentation()
    print("\nDefault Augmentation Config:")
    for key, value in augmentor.config.items():
        print(f"  {key}: {value}")
    
    # Get YOLO augmentation config
    print("\nYOLO Augmentation Config:")
    yolo_config = YOLOAugmentation.get_yolo_augmentation_config()
    for key, value in yolo_config.items():
        print(f"  {key}: {value}")
