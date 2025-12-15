import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json

class TrafficLightDetector:
    def __init__(self, model_path='models/trained/epoch80_stripped.pt'):
        self.model_path = model_path
        self.model = YOLO(model_path)
        
    def detect_image(self, image_path, conf=0.25, iou=0.45):
        """
        Detect traffic lights in image
        image_path: numpy array (BGR) or path string
        """
        results = self.model(image_path, conf=conf, iou=iou)[0]
        
        detections = []
        for box in results.boxes:
            b = box.xyxy[0].tolist()
            c = int(box.cls)
            conf_score = float(box.conf)
            label = self.model.names[c]
            
            detections.append({
                "bbox": [int(x) for x in b],
                "class": label,
                "confidence": conf_score
            })
            
        return {
            "num_detections": len(detections),
            "detections": detections,
            "image_shape": list(results.orig_shape)
        }

    def save_results(self, detection_results, image_path, output_path, save_json=True):
        """
        Draw bounding boxes and save result
        """
        # If image_path is numpy, use it. If str, load it.
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
        else:
            img = image_path.copy()
            
        for det in detection_results['detections']:
            bbox = det['bbox']
            label = det['class']
            conf = det['confidence']
            
            # Color map
            colors = {
                'red': (0, 0, 255),
                'green': (0, 255, 0),
                'yellow': (0, 255, 255),
                'off': (128, 128, 128)
            }
            color = colors.get(label.lower(), (255, 0, 0))
            
            # Draw box
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            text = f"{label} {conf:.2f}"
            cv2.putText(img, text, (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                       
        # Save image
        cv2.imwrite(str(output_path), img)
        
        if save_json:
            json_path = Path(output_path).with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(detection_results, f, indent=4)

    def detect_video(self, video_path, output_path, conf=0.25, iou=0.45, show=False):
        """
        Process video frame by frame
        """
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Output writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_results = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect
            res = self.detect_image(frame, conf=conf, iou=iou)
            frame_results.append(res)
            
            # Draw (reuse save logic logic essentially, but in-memory)
            for det in res['detections']:
                bbox = det['bbox']
                label = det['class']
                color = (0, 255, 0) # Default
                if 'red' in label: color = (0, 0, 255)
                elif 'yellow' in label: color = (0, 255, 255)
                
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            out.write(frame)
            
        cap.release()
        out.release()
        
        return frame_results
