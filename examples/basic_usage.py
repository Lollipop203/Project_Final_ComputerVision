"""
Example script demonstrating Traffic Light Detection
"""

from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.detector import TrafficLightDetector


def example_image_detection():
    """Example: Detect traffic lights in an image"""
    print("=" * 70)
    print("Example: Image Detection")
    print("=" * 70)
    
    # Initialize detector
    detector = TrafficLightDetector(device='cpu')
    
    # Path to image (replace with your image)
    image_path = "data/raw/sample.jpg"
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        print("Please add a sample image to data/raw/sample.jpg")
        return
    
    # Run detection
    print(f"\nDetecting traffic lights in {image_path}...")
    results = detector.detect_image(image_path, conf=0.5)
    
    # Print results
    print(f"\n✓ Detection complete!")
    print(f"Found {results['num_detections']} traffic light(s)")
    
    for i, det in enumerate(results['detections'], 1):
        print(f"\n{i}. Traffic Light:")
        print(f"   State: {det['class'].upper()}")
        print(f"   Confidence: {det['confidence']:.2%}")
        print(f"   Location: {det['bbox']}")
    
    # Save results
    output_path = "output/example_detection.jpg"
    detector.save_results(results, image_path, output_path)
    print(f"\n✓ Annotated image saved to: {output_path}")


def example_video_detection():
    """Example: Detect traffic lights in a video"""
    print("\n" + "=" * 70)
    print("Example: Video Detection")
    print("=" * 70)
    
    # Initialize detector
    detector = TrafficLightDetector(device='cpu')
    
    # Path to video (replace with your video)
    video_path = "data/raw/sample.mp4"
    
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        print("Please add a sample video to data/raw/sample.mp4")
        return
    
    # Run detection
    print(f"\nProcessing video: {video_path}...")
    output_path = "output/example_video_result.mp4"
    
    results = detector.detect_video(
        video_path=video_path,
        output_path=output_path,
        show=False
    )
    
    # Print summary
    frames_with_detections = sum(
        1 for r in results 
        if r['detections']['num_detections'] > 0
    )
    
    print(f"\n✓ Video processing complete!")
    print(f"Total frames: {len(results)}")
    print(f"Frames with detections: {frames_with_detections}")
    print(f"✓ Output video saved to: {output_path}")


def example_webcam_detection():
    """Example: Real-time detection from webcam"""
    print("\n" + "=" * 70)
    print("Example: Webcam Detection")
    print("=" * 70)
    
    # Initialize detector
    detector = TrafficLightDetector(device='cpu')
    
    print("\nStarting webcam detection...")
    print("Press 'q' to quit")
    
    # Run webcam detection
    detector.detect_webcam(camera_id=0)
    
    print("\n✓ Webcam detection stopped")


def main():
    """Run all examples"""
    print("\n🚦 Traffic Light Detection Examples")
    print("=" * 70)
    
    # 1. Image detection
    example_image_detection()
    
    # 2. Video detection
    # example_video_detection()  # Uncomment if you have a video
    
    # 3. Webcam detection
    # example_webcam_detection()  # Uncomment to test webcam
    
    print("\n" + "=" * 70)
    print("✓ Examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
