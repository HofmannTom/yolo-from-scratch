\# 🦷 Dental YOLO (from scratch)



This project implements a simplified YOLO-style object detector for dental X-ray images using PyTorch.



\## 🚀 Features

\- Custom dataset loader (YOLO format)

\- Grid-based detection (7x7)

\- CNN backbone

\- YOLO-style loss function

\- Training + visualization



\## 📊 Results

\- Loss decreases from \~1300 → \~200

\- Model predicts bounding boxes on dental images

\- Some false positives remain (no NMS yet)



\## 🧠 How it works

\- Image → CNN → Feature maps

\- Feature maps → Linear → 7x7 grid

\- Each cell predicts:

&#x20; - x, y, w, h

&#x20; - confidence



\## ▶️ Run training



```bash

python -m src.train

