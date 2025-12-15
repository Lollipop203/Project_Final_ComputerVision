"""
Training Script for Traffic Light Detection Model
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.augmentation import YOLOAugmentation


def train_model(
    data_yaml: str,
    model_name: str = 'yolov8n.pt',
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    config_path: str = None,
    device: str = None,
    project: str = 'runs/train',
    name: str = 'traffic_light_exp',
    resume: bool = False,
    pretrained: bool = True
):
    """
    Train YOLOv8 model for traffic light detection
    
    Args:
        data_yaml: Path to dataset YAML file
        model_name: Model architecture (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Input image size
        config_path: Path to training configuration file
        device: Device to train on (cuda:0, cpu, etc.)
        project: Project directory
        name: Experiment name
        resume: Resume training from last checkpoint
        pretrained: Use pretrained weights
    """
    
    print("=" * 70)
    print("Traffic Light Detection Model Training")
    print("=" * 70)
    
    # Load config if provided
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override parameters from config
        epochs = config['training'].get('epochs', epochs)
        batch_size = config['training'].get('batch_size', batch_size)
        img_size = config['training'].get('image_size', img_size)
        # model_name = config['model'].get('pretrained', model_name)  # Don't override from config
        
        # Get augmentation parameters
        aug_config = YOLOAugmentation.get_yolo_augmentation_config()
        if 'augmentation' in config:
            aug_config.update(config['augmentation'])
        
        # Get training hyperparameters
        lr = config['training'].get('learning_rate', 0.01)
        optimizer = config['training'].get('optimizer', 'AdamW')
        momentum = config['training'].get('momentum', 0.937)
        weight_decay = config['training'].get('weight_decay', 0.0005)
    else:
        aug_config = YOLOAugmentation.get_yolo_augmentation_config()
        lr = 0.01
        optimizer = 'AdamW'
        momentum = 0.937
        weight_decay = 0.0005
    
    # Check if CUDA is available
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nTraining Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Image Size: {img_size}")
    print(f"  Device: {device}")
    print(f"  Learning Rate: {lr}")
    print(f"  Optimizer: {optimizer}")
    print(f"  Project: {project}")
    print(f"  Name: {name}")
    print()
    
    # Initialize model
    if resume:
        # Resume from last checkpoint
        checkpoint_path = Path(project) / name / 'weights' / 'last.pt'
        if checkpoint_path.exists():
            print(f"Resuming from checkpoint: {checkpoint_path}")
            model = YOLO(str(checkpoint_path))
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            print("Starting from pretrained model")
            model = YOLO(model_name)
    else:
        # Load pretrained or create new model
        if pretrained:
            print(f"Loading pretrained model: {model_name}")
            model = YOLO(model_name)
        else:
            print(f"Creating new model: {model_name}")
            model = YOLO(model_name.replace('.pt', '.yaml'))
    
    # Train the model
    print("\nStarting training...")
    print("-" * 70)
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        pretrained=pretrained,
        optimizer=optimizer,
        lr0=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Augmentation parameters
        **aug_config,
        # Additional settings
        val=True,
        save=True,
        save_period=10,
        cache=False,
        workers=8,
        patience=50,
        plots=True,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    # Print final metrics
    print("\nFinal Metrics:")
    if hasattr(results, 'results_dict'):
        for key, value in results.results_dict.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
    
    # Print model save location
    weights_dir = Path(project) / name / 'weights'
    print(f"\nModel weights saved to:")
    print(f"  Best: {weights_dir / 'best.pt'}")
    print(f"  Last: {weights_dir / 'last.pt'}")
    
    return results


def main():
    """CLI interface for training"""
    parser = argparse.ArgumentParser(description='Train Traffic Light Detection Model')
    
    parser.add_argument('--data', type=str, 
                       default='data/dataset.yaml',
                       help='Path to dataset YAML file')
    parser.add_argument('--model', type=str,
                       default='yolov8n.pt',
                       help='Model architecture (yolov8n, yolov8s, yolov8m, etc.)')
    parser.add_argument('--weights', type=str,
                       default=None,
                       help='Path to pretrained weights (optional)')
    parser.add_argument('--epochs', type=int,
                       default=100,
                       help='Number of epochs')
    parser.add_argument('--batch', type=int,
                       default=16,
                       help='Batch size')
    parser.add_argument('--img-size', type=int,
                       default=640,
                       help='Input image size')
    parser.add_argument('--config', type=str,
                       default=None,
                       help='Path to training config YAML')
    parser.add_argument('--device', type=str,
                       default=None,
                       help='Device (cuda:0, cpu, etc.)')
    parser.add_argument('--project', type=str,
                       default='runs/train',
                       help='Project directory')
    parser.add_argument('--name', type=str,
                       default='traffic_light_exp',
                       help='Experiment name')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Train from scratch (no pretrained weights)')
    
    args = parser.parse_args()
    
    # Use config file if provided
    if args.config:
        config_path = args.config
    else:
        # Try default config path
        default_config = PROJECT_ROOT / 'configs' / 'train_config.yaml'
        config_path = str(default_config) if default_config.exists() else None
    
    # Override model with weights if provided
    model_name = args.weights if args.weights else args.model
    
    # Train model
    train_model(
        data_yaml=args.data,
        model_name=model_name,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        config_path=config_path,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        pretrained=not args.no_pretrained
    )


if __name__ == "__main__":
    main()
