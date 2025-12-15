"""
Utility functions for Traffic Light Detection System
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import yaml
import cv2
import numpy as np
from datetime import datetime


def setup_logging(log_file: str = None, level: str = 'INFO') -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger('TrafficLightDetector')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict:
    """
    Load YAML configuration file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, config_path: str) -> None:
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config
    """
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)


def format_detection_results(result, class_names: List[str]) -> Dict:
    """
    Format YOLO detection results into structured dictionary
    
    Args:
        result: YOLO result object
        class_names: List of class names
        
    Returns:
        Formatted detection results
    """
    detections = []
    
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.cpu().numpy()
        
        for box in boxes:
            # Extract box data
            xyxy = box.xyxy[0].tolist()  # Bounding box coordinates
            conf = float(box.conf[0])     # Confidence
            cls_id = int(box.cls[0])      # Class ID
            
            # Get class name
            class_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            
            detection = {
                'class': class_name,
                'class_id': cls_id,
                'confidence': round(conf, 3),
                'bbox': [round(coord, 2) for coord in xyxy],  # [x1, y1, x2, y2]
                'bbox_center': [
                    round((xyxy[0] + xyxy[2]) / 2, 2),
                    round((xyxy[1] + xyxy[3]) / 2, 2)
                ],
                'bbox_area': round((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]), 2)
            }
            detections.append(detection)
    
    return {
        'num_detections': len(detections),
        'detections': detections,
        'image_shape': result.orig_shape,
        'timestamp': datetime.now().isoformat()
    }


def draw_detections(
    image: np.ndarray,
    detection_results: Dict,
    viz_config: Dict
) -> np.ndarray:
    """
    Draw detection results on image
    
    Args:
        image: Input image
        detection_results: Detection results dictionary
        viz_config: Visualization configuration
        
    Returns:
        Annotated image
    """
    annotated = image.copy()
    
    # Get visualization settings
    colors = viz_config.get('colors', {})
    show_labels = viz_config.get('show_labels', True)
    show_confidence = viz_config.get('show_confidence', True)
    
    # Default color map
    default_colors = {
        'red': (0, 0, 255),
        'yellow': (0, 255, 255),
        'green': (0, 255, 0),
        'off': (128, 128, 128)
    }
    
    for detection in detection_results.get('detections', []):
        # Get bounding box
        bbox = detection['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get class and confidence
        class_name = detection['class']
        confidence = detection['confidence']
        
        # Get color (BGR format for OpenCV)
        color_rgb = colors.get(class_name, default_colors.get(class_name, [255, 255, 255]))
        if isinstance(color_rgb, list):
            color = tuple(reversed(color_rgb))  # Convert RGB to BGR
        else:
            color = (255, 255, 255)
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        if show_labels:
            label = f"{class_name}"
            if show_confidence:
                label += f" {confidence:.2f}"
            
            # Calculate label size
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
    
    # Draw summary info
    summary_text = f"Detections: {detection_results['num_detections']}"
    cv2.putText(
        annotated,
        summary_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )
    
    return annotated


def save_detection_json(detection_results: Dict, output_path: str) -> None:
    """
    Save detection results to JSON file
    
    Args:
        detection_results: Detection results dictionary
        output_path: Path to save JSON file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(detection_results, f, indent=2, ensure_ascii=False)


def load_detection_json(json_path: str) -> Dict:
    """
    Load detection results from JSON file
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Detection results dictionary
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    # Calculate intersection coordinates
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Calculate intersection area
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    # Calculate IoU
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou


def filter_detections_by_confidence(
    detection_results: Dict,
    min_confidence: float
) -> Dict:
    """
    Filter detections by minimum confidence threshold
    
    Args:
        detection_results: Detection results dictionary
        min_confidence: Minimum confidence threshold
        
    Returns:
        Filtered detection results
    """
    filtered_detections = [
        det for det in detection_results['detections']
        if det['confidence'] >= min_confidence
    ]
    
    return {
        **detection_results,
        'detections': filtered_detections,
        'num_detections': len(filtered_detections)
    }


def get_dominant_traffic_light(detection_results: Dict) -> str:
    """
    Get the dominant traffic light state (highest confidence)
    
    Args:
        detection_results: Detection results dictionary
        
    Returns:
        Dominant class name or 'none' if no detections
    """
    detections = detection_results.get('detections', [])
    
    if not detections:
        return 'none'
    
    # Find detection with highest confidence
    dominant = max(detections, key=lambda x: x['confidence'])
    
    return dominant['class']


def create_output_directory(base_dir: str = 'output') -> Path:
    """
    Create timestamped output directory
    
    Args:
        base_dir: Base output directory
        
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir
