📜 README.md (for your GitHub repository)

# 🗑️ Automated Waste Sorting using Computer Vision & YOLOv8

![Project Banner](https://via.placeholder.com/1200x400?text=Automated+Waste+Sorting)  
*A deep learning-based approach for real-time waste detection and classification using YOLOv8 and MongoDB.*

## 🚀 Overview
This project utilizes **YOLOv8** to detect and classify different types of litter in real-time using **computer vision**. The trained model helps in automated waste sorting, making recycling more efficient.

## 📂 Dataset
We use the **TACO (Trash Annotations in Context)** dataset, which consists of labeled images of waste items for training our object detection model.

- **Dataset Source:** [Roboflow TACO Dataset](https://universe.roboflow.com/mohamed-traore-2ekkp/taco-trash-annotations-in-context)
- **Classes:** 60 different types of waste materials (plastic, metal, paper, etc.)

## 🛠️ Installation
To set up the project, follow these steps:

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/ziatily2/Automated-Waste-Sorting-using-Computer-Vision-and-MongoDB.git
cd Automated-Waste-Sorting-using-Computer-Vision-and-MongoDB

2️⃣ Install Dependencies

pip install -r requirements.txt

If you don’t have requirements.txt, install them manually:

pip install ultralytics opencv-python roboflow numpy torch torchvision matplotlib

3️⃣ Download the Dataset

Run the following in a Jupyter Notebook or Google Colab:

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("mohamed-traore-2ekkp").project("taco-trash-annotations-in-context")
version = project.version(16)
dataset = version.download("yolov8")

This will download the dataset into the current directory.
🎯 Training the Model

To train the YOLOv8 model on the dataset:

from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Train the model
results = model.train(data="tacoily.yaml", epochs=100, imgsz=640, batch=16)

    epochs=100: Train for 100 epochs.
    imgsz=640: Image size set to 640x640.
    batch=16: Batch size of 16.

The trained weights will be saved in:

runs/detect/train/weights/best.pt

🎥 Real-Time Detection

Run the trained model to detect waste in real-time using a webcam.

import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame")
        break
    
    # Perform inference
    results = model(frame)

    # Draw detections on the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("YOLOv8 Waste Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

📈 Model Performance

After training, evaluate the model using:

metrics = model.val()

This outputs:

    Precision, Recall, and mAP (mean Average Precision).
    Confusion matrix and detection results.

📌 Directory Structure

📂 Automated-Waste-Sorting-using-Computer-Vision-and-MongoDB
 ├── 📂 dataset
 │   ├── train/
 │   ├── val/
 │   ├── test/
 ├── 📂 runs
 │   ├── detect/
 │   │   ├── train/
 │   │   │   ├── weights/
 │   │   │   │   ├── best.pt
 │   │   │   │   ├── last.pt
 ├── 📝 README.md
 ├── 📝 data.yaml
 ├── 📝 tacoily.yaml
 ├── 📜 infer.py
 ├── 📜 train.py
 ├── 📜 detector.py
 ├── 📦 requirements.txt

🛠️ Issues & Troubleshooting

    Model not detecting all classes correctly?
    🔹 Ensure the dataset is properly labeled.
    🔹 Train for more epochs (epochs=200).
    🔹 Try different confidence thresholds (conf=0.25).

    Webcam not working?
    🔹 Make sure OpenCV is installed (pip install opencv-python).
    🔹 Use cv2.VideoCapture(1) if 0 does not work.
