# app.py
import streamlit as st
import cv2
import numpy as np
import tempfile
from tracker import process_frame
from visualization import visualize_tracking_data
from PIL import Image

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Kakoo Online Interview Tracker", layout="wide")
# Show Kakoo logo
logo = Image.open("kakoo.png")
st.image(logo, width=200)
st.title("Kakoo Online Intervie Tracker")

# ----------------- SESSION STATE -----------------
for key in ["eyes_hist", "dir_hist", "depth_hist", "pos_hist", "face_det", "cap", "run_video"]:
    if key not in st.session_state:
        if key == "run_video":
            st.session_state[key] = False
        else:
            st.session_state[key] = None if key == "cap" else []

# ----------------- INPUT MODE -----------------
mode = st.radio("Input source", ["Webcam", "Upload video"], horizontal=True)

# ----------------- VIDEO UPLOAD -----------------
video_file = None
if mode == "Upload video":
    video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
    if video_file and st.session_state.cap is None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        st.session_state.cap = cv2.VideoCapture(tfile.name)

# ----------------- START / STOP BUTTONS -----------------
col1, col2 = st.columns(2)
with col1:
    if st.button("Start"):
        st.session_state.run_video = True
        # Initialize webcam if mode is Webcam
        if mode == "Webcam" and st.session_state.cap is None:
            st.session_state.cap = cv2.VideoCapture(0)
with col2:
    if st.button("Stop & Show Analysis"):
        st.session_state.run_video = False
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
        # Show plots
        visualize_tracking_data(
            eye_hist=st.session_state.eyes_hist,
            dir_hist=st.session_state.dir_hist,
            pos_hist=st.session_state.pos_hist,
            depth_hist=st.session_state.depth_hist,
            title_prefix="Online Interview Tracker"
        )

# ----------------- VIDEO DISPLAY -----------------
frame_spot = st.empty()  # Placeholder for live frames

if st.session_state.run_video:
    cap = st.session_state.cap
    if cap is None or not cap.isOpened():
        st.warning("No video source available")
        st.session_state.run_video = False
    else:
        while st.session_state.run_video:
            ret, frame = cap.read()
            if not ret:
                st.session_state.run_video = False
                cap.release()
                st.session_state.cap = None
                st.warning("Video ended or webcam not available")
                break

            frame = cv2.resize(frame, (720, 640))
            frame = process_frame(
                frame,
                st.session_state.eyes_hist,
                st.session_state.dir_hist,
                st.session_state.pos_hist,
                st.session_state.depth_hist,
                st.session_state.face_det)
            frame_spot.image(frame, channels="BGR")