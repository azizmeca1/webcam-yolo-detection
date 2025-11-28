YOLO Object Detection with OpenCV
Real-time object detection system using YOLOv8 and OpenCV for webcam streaming with FPS counter and confidence scores.
🚀 Features

Real-time object detection from webcam feed
YOLOv8 pre-trained model implementation
Bounding boxes with corner rectangles
Confidence score display for each detection
FPS (Frames Per Second) counter
80 COCO dataset classes support
Optimized for performance with streaming mode

📋 Prerequisites

Python 3.8 or higher
Webcam or video input device
CUDA-compatible GPU (optional, for better performance)

🛠️ Installation

Clone the repository:

bashgit clone https://github.com/azizmeca1/yolo-object-detection.git
cd yolo-object-detection

Install required packages:

bashpip install ultralytics opencv-python cvzone

Download YOLOv8 weights:

bash# The weights will be downloaded automatically on first run
# Or manually download from: https://github.com/ultralytics/assets/releases
📁 Project Structure
yolo-object-detection/
│
├── main.py                 # Main detection script
├── Yolo-Weights/
│   └── yolov8n.pt         # YOLOv8 nano model weights
├── Videos/                 # Optional video files for testing
├── requirements.txt
└── README.md
🎮 Usage
Webcam Detection (Default)
bashpython main.py
Video File Detection
Uncomment line 9 in the code:
pythoncap = cv2.VideoCapture("../Videos/motorbikes.mp4")
Controls

Press 'q' to quit the application
FPS is displayed in the console

🔧 Configuration
Change Camera Resolution
pythoncap.set(3, 1280)  # Width
cap.set(4, 720)   # Height
Use Different YOLO Models
pythonmodel = YOLO("yolov8n.pt")  # Nano (fastest)
model = YOLO("yolov8s.pt")  # Small
model = YOLO("yolov8m.pt")  # Medium
model = YOLO("yolov8l.pt")  # Large
model = YOLO("yolov8x.pt")  # Extra Large (most accurate)
Confidence Threshold
Modify the detection loop to filter by confidence:
pythonif conf > 0.5:  # Only show detections above 50% confidence
    cvzone.cornerRect(img, (x1, y1, w, h))
📊 Supported Classes
The model can detect 80 different object classes including:

People, vehicles (car, bus, truck, bicycle, motorbike)
Animals (dog, cat, bird, horse, etc.)
Common objects (bottle, chair, laptop, phone, etc.)
Food items (pizza, apple, sandwich, etc.)

[See full list in code]
🎯 Performance

YOLOv8n: ~45 FPS on CPU, ~200+ FPS on GPU
Resolution: 1280x720 (default)
Model Size: 6.2 MB (nano version)

🐛 Troubleshooting
Webcam Not Opening
python# Check available cameras
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is available")
        cap.release()
Image Window Not Displaying

Ensure you're not running in a headless environment
Check OpenCV installation: python -c "import cv2; print(cv2.__version__)"
Try adding cv2.waitKey(10) instead of cv2.waitKey(1)

Low FPS

Use a smaller YOLO model (yolov8n.pt)
Reduce camera resolution
Enable GPU acceleration (CUDA)
Close other applications

📦 Dependencies
txtultralytics>=8.0.0
opencv-python>=4.8.0
cvzone>=1.6.1
numpy>=1.24.0
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the project
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request
