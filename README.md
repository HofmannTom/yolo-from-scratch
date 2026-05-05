🦷 Dental Object Detection (Mini-YOLO from Scratch)

This project implements a simplified YOLO-style object detection model from scratch using PyTorch.

The goal is to detect dental structures in panoramic X-ray images using a custom-built neural network — without relying on pretrained detection frameworks.

🚀 Features
Custom CNN-based object detector
YOLO-style grid prediction (7x7)
Bounding box prediction: [x, y, w, h, confidence]
Custom loss function:
Box regression loss
Object confidence loss
No-object penalty
Non-Maximum Suppression (NMS)
Visualization of predictions vs ground truth
Model saving (model.pt)
🧠 Model Overview

Input:

Grayscale image (1 × 256 × 256)

Output:

Grid: (7 × 7 × 5)
Each cell predicts:
x, y (relative position)
width, height
confidence score
🔄 Pipeline
Image → CNN → Feature Maps → Grid Prediction → Bounding Boxes → NMS → Final Output
📦 Training

Run training:

python src/train.py

Model is saved automatically:

model.pt
🖼️ Results

Predictions are saved to:

results/prediction.png

Example:

Green = Predictions
Red = Ground Truth
🔧 Requirements

Install dependencies:

pip install -r requirements.txt
📁 Project Structure
src/
 ├── train.py
 ├── model.py
 ├── dataset.py
 ├── utils.py
🎯 Key Concepts Implemented
Convolutional Neural Networks (CNN)
Grid-based object detection (YOLO concept)
Intersection over Union (IoU)
Non-Maximum Suppression (NMS)
Custom loss functions
📌 Notes

This is a learning-focused implementation and not optimized for production.

👨‍💻 Author

Thomas Hofmann

