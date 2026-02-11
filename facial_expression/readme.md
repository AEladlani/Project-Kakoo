# 🎭 Real-Time Facial Emotion Recognition (FER)

This project implements a **real-time facial emotion recognition system** that works with both **uploaded videos** and **live webcam input**.  
It detects faces, aligns them using facial landmarks, and predicts emotions using a deep learning model, with temporal smoothing for stable results.

Built with **Python, OpenCV, MediaPipe, PyTorch, and Streamlit**.

---

## 🚀 Features

- 📹 Emotion recognition from:
  - Uploaded video files (`.mp4`, `.avi`, `.mov`)
  - Live webcam stream
- 👤 Robust face detection and alignment using facial landmarks
- 🧠 Deep learning–based emotion classification (PyTorch)
- 🔁 Temporal smoothing using a **FIFO sliding window**
- 📊 Emotion distribution visualization after processing
- 🛑 Start / Stop controls for camera mode
- ⚡ Real-time performance

---

## 🧠 System Overview

The pipeline follows these steps:

1. **Frame Acquisition**
   - Read frames from video or webcam using OpenCV.

2. **Face Landmark Detection**
   - Use a **MediaPipe Face Landmarker** model to detect facial landmarks.
   - Ensures accurate face localization even under pose variation.

3. **Face Alignment**
   - Align the detected face based on landmarks.
   - Improves robustness and consistency for emotion prediction.

4. **Emotion Classification (PyTorch Model)**
   - The aligned face is passed to a trained **facial emotion recognition model**.
   - Outputs an emotion class index and confidence score.

5. **Temporal Smoothing (FIFO Window)**
   - Predictions are stored in a fixed-size sliding window.
   - The oldest prediction is removed when the window is full.
   - The final emotion is selected via majority vote to reduce flickering.

6. **Visualization & Statistics**
   - Emotion labels are rendered on each frame.
   - After stopping, emotion percentages are displayed.

---

## 🧩 Models Used

### 🔹 Face Landmark Model
- **Framework:** MediaPipe
- **Purpose:** Face detection and facial landmark extraction
- **Why:** Fast, accurate, and robust across different lighting and head poses
- **Output:** Facial landmarks used for face alignment

### 🔹 Emotion Recognition Model
- **Framework:** PyTorch
- **Input:** Aligned face image
- **Output:** Emotion class + confidence
- **Usage:** Real-time inference on each detected face
-- **Weights:** can be downloaded here **[weights](https://drive.google.com/drive/folders/1YRh8XarO1A4SVUahAVKPwMtWLDmP1LNl?usp=sharing)**

---

## 🪟 Temporal Smoothing Strategy

To stabilize predictions, the system uses a **FIFO (First-In-First-Out) sliding window**:

- Keeps the last `N` predictions
- Removes the oldest prediction when a new one arrives
- Computes the most frequent emotion in the window

This avoids rapid emotion changes caused by noisy frame-level predictions.

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **OpenCV** – video processing
- **MediaPipe** – face landmarks & alignment
- **PyTorch** – emotion recognition model
- **Streamlit** – interactive web interface
- **NumPy / Collections** – data handling

---

## ▶️ How to Run

```bash
streamlit run app.py
