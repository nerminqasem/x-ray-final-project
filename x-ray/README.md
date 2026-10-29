# PneumoVision

**AI-Based Pneumonia Detection and Chest X-ray Analysis Platform**

A web-based application for automatic detection of pneumonia from chest X-ray images using deep learning (DenseNet121, MobileNetV3) and image preprocessing (CLAHE, Histogram Equalization, Gamma Correction).

---

## Features

- Upload chest X-ray images (PNG, JPG, JPEG, DICOM)
- Apply image preprocessing (CLAHE, HistEq, Gamma)
- Classify image as NORMAL or PNEUMONIA
- Confidence score and inference time
- Performance dashboard with charts, confusion matrix, ROC curve
- Dataset analytics
- Prediction history
- Admin panel with logs

---

## Tech Stack

- **Backend:** Flask
- **AI:** PyTorch (DenseNet121, MobileNetV3)
- **Image Processing:** OpenCV, Pillow, pydicom
- **Database:** SQLite
- **Charts:** Chart.js
- **Container:** Docker, Docker Compose

---

## Project Structure
