"""
Unit Tests for Traffic Light Detector
"""

import os
import sys
import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.detector import TrafficLightDetector
from src.agent.utils import (
    load_config,
    format_detection_results,
    calculate_iou,
    filter_detections_by_confidence,
    get_dominant_traffic_light
)


@pytest.fixture
def sample_image():
    """Create a sample test image"""
    # Create a simple RGB image
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    # Draw a red circle (simulating traffic light)
    cv2.circle(image, (320, 200), 50, (255, 0, 0), -1)
    return image


@pytest.fixture
def sample_detection_results():
    """Create sample detection results"""
    return {
        'num_detections': 2,
        'detections': [
            {
                'class': 'red',
                'class_id': 0,
                'confidence': 0.95,
                'bbox': [100, 150, 200, 300],
                'bbox_center': [150, 225],
                'bbox_area': 15000
            },
            {
                'class': 'green',
                'class_id': 2,
                'confidence': 0.87,
                'bbox': [400, 200, 500, 350],
                'bbox_center': [450, 275],
                'bbox_area': 15000
            }
        ],
        'image_shape': [640, 640],
        'timestamp': '2024-01-01T12:00:00'
    }


class TestDetector:
    """Test cases for TrafficLightDetector"""
    
    @patch('src.agent.detector.YOLO')
    def test_detector_initialization(self, mock_yolo):
        """Test detector initialization"""
        # Mock YOLO model
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        
        # Initialize detector
        detector = TrafficLightDetector(device='cpu')
        
        # Assertions
        assert detector is not None
        assert detector.device == 'cpu'
        assert detector.model is not None
        mock_yolo.assert_called()
    
    @patch('src.agent.detector.YOLO')
    def test_detect_image(self, mock_yolo, sample_image):
        """Test image detection"""
        # Mock YOLO model and results
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.cpu.return_value.numpy.return_value = []
        mock_result.orig_shape = (640, 640)
        
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        # Initialize detector
        detector = TrafficLightDetector(device='cpu')
        
        # Run detection
        results = detector.detect_image(sample_image)
        
        # Assertions
        assert isinstance(results, dict)
        assert 'num_detections' in results
        assert 'detections' in results
        assert 'image_shape' in results
        mock_model.predict.assert_called_once()
    
    @patch('src.agent.detector.YOLO')
    def test_save_results(self, mock_yolo, sample_image, sample_detection_results, tmp_path):
        """Test saving detection results"""
        # Mock YOLO
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        
        # Initialize detector
        detector = TrafficLightDetector(device='cpu')
        
        # Save results
        output_path = tmp_path / "output.jpg"
        detector.save_results(
            detection_results=sample_detection_results,
            image_path=sample_image,
            output_path=output_path,
            save_json=True
        )
        
        # Assertions
        assert output_path.exists()
        json_path = output_path.with_suffix('.json')
        assert json_path.exists()


class TestUtils:
    """Test cases for utility functions"""
    
    def test_load_config(self, tmp_path):
        """Test config loading"""
        # Create temporary config file
        config_content = """
model:
  architecture: yolov8n
  confidence_threshold: 0.5

classes:
  names: ['red', 'yellow', 'green', 'off']
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        
        # Load config
        config = load_config(str(config_file))
        
        # Assertions
        assert config is not None
        assert 'model' in config
        assert config['model']['architecture'] == 'yolov8n'
        assert config['model']['confidence_threshold'] == 0.5
    
    def test_calculate_iou(self):
        """Test IoU calculation"""
        box1 = [0, 0, 100, 100]
        box2 = [50, 50, 150, 150]
        
        iou = calculate_iou(box1, box2)
        
        # Expected IoU: intersection = 2500, union = 17500, iou = 0.1428...
        assert 0.14 < iou < 0.15
        
        # Test identical boxes
        iou_identical = calculate_iou(box1, box1)
        assert iou_identical == 1.0
        
        # Test non-overlapping boxes
        box3 = [200, 200, 300, 300]
        iou_zero = calculate_iou(box1, box3)
        assert iou_zero == 0.0
    
    def test_filter_detections_by_confidence(self, sample_detection_results):
        """Test confidence filtering"""
        # Filter with threshold 0.9
        filtered = filter_detections_by_confidence(
            sample_detection_results,
            min_confidence=0.9
        )
        
        # Should only keep the detection with confidence 0.95
        assert filtered['num_detections'] == 1
        assert filtered['detections'][0]['confidence'] >= 0.9
        
        # Filter with threshold 0.8
        filtered_low = filter_detections_by_confidence(
            sample_detection_results,
            min_confidence=0.8
        )
        
        # Should keep both detections
        assert filtered_low['num_detections'] == 2
    
    def test_get_dominant_traffic_light(self, sample_detection_results):
        """Test dominant traffic light detection"""
        dominant = get_dominant_traffic_light(sample_detection_results)
        
        # Should be 'red' (highest confidence 0.95)
        assert dominant == 'red'
        
        # Test empty detections
        empty_results = {'detections': []}
        dominant_empty = get_dominant_traffic_light(empty_results)
        assert dominant_empty == 'none'


class TestFormatDetectionResults:
    """Test cases for result formatting"""
    
    def test_format_empty_results(self):
        """Test formatting when no detections"""
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_result.orig_shape = (640, 640)
        
        class_names = ['red', 'yellow', 'green', 'off']
        formatted = format_detection_results(mock_result, class_names)
        
        assert formatted['num_detections'] == 0
        assert len(formatted['detections']) == 0
        assert formatted['image_shape'] == (640, 640)


@pytest.mark.integration
class TestIntegration:
    """Integration tests (require actual model)"""
    
    @pytest.mark.skip(reason="Requires trained model weights")
    def test_full_detection_pipeline(self, sample_image, tmp_path):
        """Test full detection pipeline with real model"""
        # This test would run with an actual model
        detector = TrafficLightDetector(device='cpu')
        
        # Detect
        results = detector.detect_image(sample_image)
        
        # Save
        output_path = tmp_path / "test_output.jpg"
        detector.save_results(results, sample_image, output_path)
        
        # Verify
        assert output_path.exists()
        assert results is not None


def test_module_imports():
    """Test that all modules can be imported"""
    try:
        from src.agent import detector, utils
        from src.preprocessing import augmentation
        from src.training import train, evaluate
        from src.api import app
        assert True
    except ImportError as e:
        pytest.fail(f"Module import failed: {e}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
