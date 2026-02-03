# Kakoo Online Tracker

## Overview

This project implements a **real-time face and eye tracking system** using **MediaPipe** and **OpenCV**. It detects key facial landmarks including pupils and eye corners to predict **gaze direction** and track **head position and orientation**. It is suitable for applications like online human-computer interaction studies.

A notebook **experiment.ipynb** explain the steps in details.

### Features

- Detects facial landmarks with MediaPipe Face Landmarker.
- Tracks **pupil positions** to estimate gaze direction (Left, Right, Center, Blinking).
- Tracks **head position and orientation** (X/Y position, horizontal/vertical direction).
- Displays results in **real-time** with OpenCV panels.
- Stores tracking history for **offline visualization** and analysis.
- Supports **webcam input** or **video file upload**.
- Shows plots and statistics of gaze, head direction, face positions, and distance after tracking.

---

To run the app locally:
## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/AEladlani/Project-Kakoo.git
cd Project-Kakoo/face_eyes_tracking


2. pip install -r requirements.txt

3. streamlit run app.py
