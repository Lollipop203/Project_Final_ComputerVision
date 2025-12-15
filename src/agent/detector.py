"""
Traffic Light Detector Module
Main AI Agent for detecting traffic lights using YOLOv8
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
import yaml
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.utils import (
    load_config,
    format_detection_results,
    draw_detections,
    save_detection_json,
    setup_logging
)


class TrafficLightDetector:
    """Traffic Light Detection Agent using YOLOv8"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = 'cuda'
    ):
        """
        Initialize the Traffic Light Detector
        
        Args:
            model_path: Path to trained model weights
            config_path: Path to configuration file
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.logger = setup_logging()
        self.logger.info("Initializing Traffic Light Detector...")
        
        # Load configuration
        if config_path is None:
            config_path = PROJECT_ROOT / 'configs' / 'config.yaml'
        self.config = load_config(config_path)
        
        # Set device
        self.device = device if device else self.config['model'].get('device', 'cpu')
        
        # Load model
        if model_path is None:
            # Try to load trained model, fallback to pretrained
            trained_path = PROJECT_ROOT / 'models' / 'trained' / 'best.pt'
            if trained_path.exists():
                model_path = str(trained_path)
            else:
                model_path = self.config['model']['architecture'] + '.pt'
                self.logger.warning(f"No trained model found. Using pretrained: {model_path}")
        
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.logger.info(f"Model loaded: {model_path} on {self.device}")
        
        # Detection parameters
        self.conf_threshold = self.config['model']['confidence_threshold']
        self.iou_threshold = self.config['model']['iou_threshold']
        self.class_names = self.config['classes']['names']
        
    def detect_image(
        self,
        image_path: Union[str, Path, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None
    ) -> Dict:
        """
        Detect traffic lights in an image
        
        Args:
            image_path: Path to image or numpy array
            conf: Confidence threshold (optional)
            iou: IOU threshold (optional)
            
        Returns:
            Dictionary containing detection results
        """
        conf = conf if conf is not None else self.conf_threshold
        iou = iou if iou is not None else self.iou_threshold
        
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False
        )
        
        # Format results
        detection_results = format_detection_results(results[0], self.class_names)
        
        return detection_results
    
    def detect_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        show: bool = False,
        conf: Optional[float] = None,
        iou: Optional[float] = None
    ) -> List[Dict]:
        """
        Detect traffic lights in a video
        
        Args:
            video_path: Path to video file
            output_path: Path to save output video
            show: Whether to display video while processing
            conf: Confidence threshold
            iou: IOU threshold
            
        Returns:
            List of detection results for each frame
        """
        conf = conf if conf is not None else self.conf_threshold
        iou = iou if iou is not None else self.iou_threshold
        
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        all_results = []
        frame_count = 0
        
        self.logger.info(f"Processing video: {video_path}")
        self.logger.info(f"Total frames: {total_frames}, FPS: {fps}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect in frame
            results = self.model.predict(
                source=frame,
                conf=conf,
                iou=iou,
                device=self.device,
                verbose=False
            )
            
            # Format and store results
            detection_results = format_detection_results(results[0], self.class_names)
            all_results.append({
                'frame': frame_count,
                'detections': detection_results
            })
            
            # Draw detections
            annotated_frame = draw_detections(
                frame.copy(),
                detection_results,
                self.config['visualization']
            )
            
            # Write to output
            if writer:
                writer.write(annotated_frame)
            
            # Display
            if show:
                cv2.imshow('Traffic Light Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            if frame_count % 30 == 0:
                self.logger.info(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        
        self.logger.info(f"Video processing complete. Processed {frame_count} frames.")
        
        return all_results
    
    def detect_webcam(
        self,
        camera_id: int = 0,
        conf: Optional[float] = None,
        iou: Optional[float] = None
    ) -> None:
        """
        Real-time detection from webcam
        
        Args:
            camera_id: Camera device ID
            conf: Confidence threshold
            iou: IOU threshold
        """
        conf = conf if conf is not None else self.conf_threshold
        iou = iou if iou is not None else self.iou_threshold
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            self.logger.error(f"Failed to open camera {camera_id}")
            return
        
        self.logger.info(f"Starting webcam detection (camera {camera_id}). Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                self.logger.error("Failed to read frame from camera")
                break
            
            # Run detection
            results = self.model.predict(
                source=frame,
                conf=conf,
                iou=iou,
                device=self.device,
                verbose=False,
                stream=True
            )
            
            # Process first result
            for result in results:
                detection_results = format_detection_results(result, self.class_names)
                annotated_frame = draw_detections(
                    frame.copy(),
                    detection_results,
                    self.config['visualization']
                )
                
                # Display FPS
                fps_text = f"FPS: {int(cap.get(cv2.CAP_PROP_FPS))}"
                cv2.putText(
                    annotated_frame,
                    fps_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                cv2.imshow('Traffic Light Detection - Webcam', annotated_frame)
                break
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.logger.info("Webcam detection stopped.")
    
    def save_results(
        self,
        detection_results: Dict,
        image_path: Union[str, Path],
        output_path: Union[str, Path],
        save_json: bool = True
    ) -> None:
        """
        Save detection results with annotations
        
        Args:
            detection_results: Detection results dictionary
            image_path: Original image path
            output_path: Output path for annotated image
            save_json: Whether to save JSON results
        """
        # Load original image
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
        else:
            image = image_path
        
        # Draw detections
        annotated_image = draw_detections(
            image,
            detection_results,
            self.config['visualization']
        )
        
        # Save annotated image
        cv2.imwrite(str(output_path), annotated_image)
        self.logger.info(f"Saved annotated image to {output_path}")
        
        # Save JSON results
        if save_json:
            json_path = Path(output_path).with_suffix('.json')
            save_detection_json(detection_results, json_path)
            self.logger.info(f"Saved JSON results to {json_path}")


def main():
    """CLI interface for Traffic Light Detector"""
    parser = argparse.ArgumentParser(description='Traffic Light Detection System')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image, video, or camera ID (0, 1, etc.)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model weights')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--conf', type=float, default=None,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=None,
                       help='IOU threshold')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on (cuda or cpu)')
    parser.add_argument('--show', action='store_true',
                       help='Display results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    detector = TrafficLightDetector(
        model_path=args.model,
        device=args.device
    )
    
    # Determine source type
    source = args.source
    
    # Check if source is a camera ID
    if source.isdigit():
        detector.detect_webcam(
            camera_id=int(source),
            conf=args.conf,
            iou=args.iou
        )
    # Check if source is a video
    elif Path(source).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        output_path = output_dir / f"output_{Path(source).stem}.mp4"
        detector.detect_video(
            video_path=source,
            output_path=output_path,
            show=args.show,
            conf=args.conf,
            iou=args.iou
        )
    # Assume source is an image
    else:
        results = detector.detect_image(
            image_path=source,
            conf=args.conf,
            iou=args.iou
        )
        output_path = output_dir / f"output_{Path(source).name}"
        detector.save_results(results, source, output_path)
        
        if args.show:
            img = cv2.imread(str(output_path))
            cv2.imshow('Detection Result', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    print(f"✓ Detection complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
