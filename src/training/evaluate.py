"""
Model Evaluation Script for Traffic Light Detection
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def evaluate_model(
    model_path: str,
    data_yaml: str,
    device: str = None,
    save_dir: str = 'runs/evaluate',
    img_size: int = 640,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> Dict:
    """
    Evaluate trained model on test/validation dataset
    
    Args:
        model_path: Path to trained model weights
        data_yaml: Path to dataset YAML file
        device: Device to run evaluation on
        save_dir: Directory to save evaluation results
        img_size: Input image size
        conf_threshold: Confidence threshold
        iou_threshold: IOU threshold
        
    Returns:
        Dictionary containing evaluation metrics
    """
    
    print("=" * 70)
    print("Traffic Light Detection Model Evaluation")
    print("=" * 70)
    
    # Check device
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nEvaluation Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Dataset: {data_yaml}")
    print(f"  Device: {device}")
    print(f"  Image Size: {img_size}")
    print(f"  Confidence Threshold: {conf_threshold}")
    print(f"  IOU Threshold: {iou_threshold}")
    print(f"  Save Directory: {save_dir}")
    print()
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    model.to(device)
    
    # Run validation
    print("\nRunning validation...")
    print("-" * 70)
    
    results = model.val(
        data=data_yaml,
        imgsz=img_size,
        batch=16,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        save_json=True,
        save_hybrid=False,
        plots=True,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)
    
    # Extract metrics
    metrics = {
        'mAP50': float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
        'mAP50-95': float(results.box.map) if hasattr(results.box, 'map') else 0.0,
        'precision': float(results.box.p.mean()) if hasattr(results.box, 'p') else 0.0,
        'recall': float(results.box.r.mean()) if hasattr(results.box, 'r') else 0.0,
        'f1_score': 0.0,
    }
    
    # Calculate F1 score
    if metrics['precision'] > 0 and metrics['recall'] > 0:
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                              (metrics['precision'] + metrics['recall'])
    
    # Print metrics
    print("\nOverall Metrics:")
    print(f"  mAP@0.5: {metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    
    # Per-class metrics
    if hasattr(results.box, 'ap_class_index'):
        print("\nPer-Class Metrics:")
        class_names = ['red', 'yellow', 'green', 'off']  # Default classes
        
        # Load class names from dataset yaml if available
        try:
            with open(data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
                if 'names' in data_config:
                    class_names = list(data_config['names'].values()) if isinstance(data_config['names'], dict) \
                                 else data_config['names']
        except:
            pass
        
        per_class_metrics = {}
        for idx in range(len(class_names)):
            class_name = class_names[idx]
            per_class_metrics[class_name] = {
                'precision': float(results.box.p[idx]) if idx < len(results.box.p) else 0.0,
                'recall': float(results.box.r[idx]) if idx < len(results.box.r) else 0.0,
                'mAP50': float(results.box.ap50[idx]) if idx < len(results.box.ap50) else 0.0,
            }
            print(f"  {class_name}:")
            print(f"    Precision: {per_class_metrics[class_name]['precision']:.4f}")
            print(f"    Recall: {per_class_metrics[class_name]['recall']:.4f}")
            print(f"    mAP@0.5: {per_class_metrics[class_name]['mAP50']:.4f}")
        
        metrics['per_class'] = per_class_metrics
    
    # Save metrics to JSON
    metrics_file = save_path / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_file}")
    
    # Generate additional plots
    generate_evaluation_plots(metrics, save_path)
    
    return metrics


def generate_evaluation_plots(metrics: Dict, save_dir: Path):
    """
    Generate additional evaluation plots
    
    Args:
        metrics: Evaluation metrics dictionary
        save_dir: Directory to save plots
    """
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Overall metrics bar plot
    if 'per_class' not in metrics:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    class_names = list(metrics['per_class'].keys())
    
    # Precision plot
    precisions = [metrics['per_class'][cls]['precision'] for cls in class_names]
    axes[0].bar(class_names, precisions, color='skyblue')
    axes[0].set_title('Precision by Class')
    axes[0].set_ylabel('Precision')
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Recall plot
    recalls = [metrics['per_class'][cls]['recall'] for cls in class_names]
    axes[1].bar(class_names, recalls, color='lightgreen')
    axes[1].set_title('Recall by Class')
    axes[1].set_ylabel('Recall')
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis='y', alpha=0.3)
    
    # mAP plot
    maps = [metrics['per_class'][cls]['mAP50'] for cls in class_names]
    axes[2].bar(class_names, maps, color='salmon')
    axes[2].set_title('mAP@0.5 by Class')
    axes[2].set_ylabel('mAP@0.5')
    axes[2].set_ylim([0, 1])
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'class_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation plots saved to {save_dir}")


def main():
    """CLI interface for evaluation"""
    parser = argparse.ArgumentParser(description='Evaluate Traffic Light Detection Model')
    
    parser.add_argument('--weights', type=str,
                       required=True,
                       help='Path to trained model weights')
    parser.add_argument('--data', type=str,
                       default='data/dataset.yaml',
                       help='Path to dataset YAML file')
    parser.add_argument('--device', type=str,
                       default=None,
                       help='Device (cuda:0, cpu, etc.)')
    parser.add_argument('--save-dir', type=str,
                       default='runs/evaluate',
                       help='Directory to save evaluation results')
    parser.add_argument('--img-size', type=int,
                       default=640,
                       help='Input image size')
    parser.add_argument('--conf', type=float,
                       default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float,
                       default=0.45,
                       help='IOU threshold')
    
    args = parser.parse_args()
    
    # Run evaluation
    metrics = evaluate_model(
        model_path=args.weights,
        data_yaml=args.data,
        device=args.device,
        save_dir=args.save_dir,
        img_size=args.img_size,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
