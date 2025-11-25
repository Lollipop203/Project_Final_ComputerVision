# Traffic Light Detection System

Hệ thống phát hiện đèn giao thông sử dụng YOLOv8 và Deep Learning.

## 📋 Mục lục
- [Tính năng](#tính-năng)
- [Cài đặt](#cài-đặt)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Sử dụng](#sử-dụng)
- [Huấn luyện mô hình](#huấn-luyện-mô-hình)
- [API Documentation](#api-documentation)

## 🎯 Tính năng

- Phát hiện đèn giao thông trong ảnh và video
- Phân loại trạng thái: Đỏ, Vàng, Xanh
- Hỗ trợ real-time detection qua webcam
- RESTful API để tích hợp vào các hệ thống khác
- Giao diện web đơn giản để demo
- Xuất kết quả dưới dạng JSON và hình ảnh đã được annotate

## 🚀 Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- CUDA (tùy chọn, để sử dụng GPU)

### Cài đặt dependencies

```bash
# Clone repository
git clone https://github.com/yourusername/traffic-light-detection.git
cd traffic-light-detection

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt các thư viện
pip install -r requirements.txt
```

## 📁 Cấu trúc thư mục

```
traffic-light-detection/
├── data/
│   ├── raw/                 # Dữ liệu gốc
│   ├── processed/           # Dữ liệu đã xử lý
│   ├── annotations/         # File annotations
│   └── dataset.yaml         # Config dataset cho YOLO
├── models/
│   ├── pretrained/          # Mô hình pretrained
│   └── trained/             # Mô hình đã train
├── src/
│   ├── agent/
│   │   ├── detector.py      # AI Agent chính
│   │   └── utils.py         # Utility functions
│   ├── preprocessing/
│   │   └── augmentation.py  # Data augmentation
│   ├── training/
│   │   └── train.py         # Script training
│   └── api/
│       └── app.py           # FastAPI application
├── notebooks/
│   └── exploration.ipynb    # Jupyter notebook để thử nghiệm
├── tests/
│   └── test_detector.py     # Unit tests
├── configs/
│   └── config.yaml          # File config chung
├── requirements.txt
├── README.md
└── setup.py
```

## 💻 Sử dụng

### 1. Phát hiện từ ảnh

```python
from src.agent.detector import TrafficLightDetector

# Khởi tạo detector
detector = TrafficLightDetector(model_path='models/trained/best.pt')

# Phát hiện từ ảnh
results = detector.detect_image('path/to/image.jpg')

# Lưu kết quả
detector.save_results(results, 'output/result.jpg')
```

### 2. Phát hiện từ video

```python
detector.detect_video(
    video_path='path/to/video.mp4',
    output_path='output/result.mp4',
    show=True
)
```

### 3. Real-time detection từ webcam

```python
detector.detect_webcam(camera_id=0)
```

### 4. Chạy API Server

```bash
# Khởi động server
python src/api/app.py

# Server sẽ chạy tại http://localhost:8000
# Truy cập API docs tại http://localhost:8000/docs
```

### 5. Sử dụng CLI

```bash
# Phát hiện từ ảnh
python -m src.agent.detector --source image.jpg --output output/

# Phát hiện từ video
python -m src.agent.detector --source video.mp4 --output output/

# Phát hiện từ webcam
python -m src.agent.detector --source 0
```

## 🎓 Huấn luyện mô hình

### Chuẩn bị dữ liệu

```bash
# Download dataset (ví dụ từ Roboflow)
python scripts/download_data.py

# Hoặc chuẩn bị dữ liệu riêng theo format YOLO
# Xem hướng dẫn chi tiết trong data/README.md
```

### Training

```bash
# Train từ đầu
python src/training/train.py --epochs 100 --batch 16

# Fine-tune từ pretrained model
python src/training/train.py --weights models/pretrained/yolov8n.pt --epochs 50

# Train với config file
python src/training/train.py --config configs/train_config.yaml
```

### Đánh giá mô hình

```bash
python src/training/evaluate.py --weights models/trained/best.pt
```

## 🔌 API Documentation

### Endpoints

#### POST /detect
Phát hiện đèn giao thông từ ảnh upload

**Request:**
```bash
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

**Response:**
```json
{
  "success": true,
  "detections": [
    {
      "class": "red",
      "confidence": 0.95,
      "bbox": [100, 200, 150, 300]
    }
  ],
  "image_url": "/results/output_123.jpg"
}
```

#### POST /detect-video
Phát hiện từ video

#### GET /health
Kiểm tra trạng thái server

## 📊 Dataset

Hệ thống hỗ trợ các định dạng dataset:
- YOLO format (recommended)
- COCO format
- Pascal VOC format

Classes được hỗ trợ:
- `red`: Đèn đỏ
- `yellow`: Đèn vàng
- `green`: Đèn xanh
- `off`: Đèn tắt

## 🛠️ Configuration

Chỉnh sửa file `configs/config.yaml`:

```yaml
model:
  architecture: yolov8n
  input_size: 640
  confidence_threshold: 0.5
  iou_threshold: 0.45

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001
  optimizer: Adam

data:
  train_path: data/processed/train
  val_path: data/processed/val
  test_path: data/processed/test
```

## 🧪 Testing

```bash
# Chạy tất cả tests
pytest tests/

# Chạy test cụ thể
pytest tests/test_detector.py -v

# Test với coverage
pytest --cov=src tests/
```

## 📈 Performance

| Model | mAP@0.5 | FPS (GPU) | FPS (CPU) |
|-------|---------|-----------|-----------|
| YOLOv8n | 0.89 | 120 | 25 |
| YOLOv8s | 0.92 | 95 | 18 |
| YOLOv8m | 0.94 | 70 | 12 |

## 🤝 Contributing

Mọi đóng góp đều được chào đón! Vui lòng:
1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📝 License

MIT License - xem file LICENSE để biết thêm chi tiết

## 👥 Authors

- Your Name - [@yourhandle](https://github.com/yourhandle)

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- Dataset từ [nguồn dataset của bạn]
- Inspired by various traffic light detection projects
