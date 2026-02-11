# Project-Kakoo
# 🧠 AI Projects Suite: Sentiment, Facial Expression & Face/Eye Tracking

This repository contains **three AI projects** that all run in **one shared Python environment**.  
The projects are designed to work together under the same setup because **MediaPipe only supports Python 3.9, 3.10, 3.11, and 3.12**.  
All projects are tested and recommended with **Python 3.9.25**.

---

## 📁 Repository Structure

.
├── sentiment/
│   ├── notebook.ipynb
│   └── new_sent.py
│
├── facial_expression/
│   ├── notebook.ipynb
│   └── app.py
│
├── track_eyes_face/
│   ├── notebook.ipynb
│   └── app.py
│
├── all_requirements.txt
└── README.md

Each project includes:
- A **Jupyter Notebook** explaining the methodology, models, and experiments
- A **Streamlit app** for local deployment

---

## 🧩 Projects Description

### 1️⃣ Sentiment Analysis
- Multilingual sentiment analysis for text
- Uses **Transformers** and **deepmultilingualpunctuation**
- Streamlit-based interactive application

Run:
cd sentiment  
streamlit run new_sent.py

---

### 2️⃣ Facial Expression Recognition
- Facial emotion recognition from **video or webcam**
- Face detection and alignment using **MediaPipe Face Landmarks**
- Emotion classification using a **PyTorch ResNet-based model**
- Temporal smoothing using a FIFO window for stable predictions
- Emotion distribution visualization after inference

Run:
cd facial_expression  
streamlit run app.py

---

### 3️⃣ Face & Eye Tracking
- Face and eye landmark tracking
- Built using **MediaPipe** and **OpenCV**
- Useful for attention analysis and facial behavior tracking

Run:
cd track_eyes_face  
streamlit run app.py

---

## 🐍 Python & Environment

Recommended Python version:
Python 3.9.25

Supported Python versions:
- 3.9
- 3.10
- 3.11
- 3.12

All projects must be run inside the **same virtual environment**.

---

## 📦 Dependencies

All required packages are listed in **all_requirements.txt**.

Python: 3.9.25  
deepmultilingualpunctuation  
tqdm==4.67.1  
streamlit==1.50.0  
colorama  
transformers==4.57.6  
matplotlib==3.9.4  
mediapipe==0.10.21  
torch==2.8.0+cu128  
opencv-python==4.11.0  
Pillow==11.3.0  

---

## 🚀 How to Run

Clone the repository:
git clone <repository-url>  
cd <repository-name>

Create a virtual environment:
python3.9 -m venv ai_env

Activate it:

Linux / macOS:
source ai_env/bin/activate

Windows:
ai_env\Scripts\activate

Install dependencies:
pip install --upgrade pip  
pip install -r all_requirements.txt

If CUDA-enabled PyTorch is not installed automatically:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

Run a project:

Sentiment Analysis:
cd sentiment  
streamlit run new_sent.py

Facial Expression Recognition:
cd facial_expression  
streamlit run app.py

Face & Eye Tracking:
cd track_eyes_face  
streamlit run app.py

---

## 📓 Notebooks

Each project includes a notebook that explains:
- Data preprocessing
- Model architecture
- Inference pipeline
- Experiments and evaluation

These notebooks are recommended for understanding the project internals before running the Streamlit applications.
