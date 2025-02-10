# Traffic Management - Vehicle Detection, Tracking, and License Plate Recognition

## Problem Description
This AI-based traffic management system includes:
- **Vehicle Detection**: Utilizing **YOLOv8** to detect vehicles in video streams.
- **Vehicle Tracking**: Using **ByteTrack** to track vehicle movements.
- **License Plate Recognition**:
  - **License Plate Detection**: YOLOv8 is used to detect license plates.
  - **OCR for License Plates**: **PaddleOCR** is employed to recognize text on license plates.
- **Speed Calculation**:
  - Speed is calculated based on the time a vehicle appears and the distance traveled.
  - Vehicles exceeding the speed limit are marked **in red**.

---

## Installation

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### Run Demo with **Streamlit**
```bash
streamlit run speed_detect_demo.py
```

### Run via **CLI**
```bash
python speed_NhanDang.py
```
[Demo Video](demo_video/demo.mp4)
