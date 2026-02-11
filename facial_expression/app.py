import streamlit as st
import cv2
import torch
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from collections import Counter
import tempfile
import os
from PIL import Image
# ===================== STREAMLIT CONFIG =====================
st.set_page_config(page_title="FER KAKOO", layout="wide")
logo = Image.open("kakoo.png")
st.image(logo, width=200)
st.title("🎭 Facial Expression Recognition")
# ===================== DEVICE =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Device: **{device.upper()}**")
# ===================== LOAD MODELS (ONCE) =====================
@st.cache_resource
def load_models():
    from libreface.resnet18 import ResNet
    from libreface.solver_inference_image import solver_inference_image
    from libreface.detect_mediapipe_image import get_aligned_image
    fer_model = ResNet(num_labels=8, dropout=0.1, fm_distillation=True).to(device)
    ckpt = torch.load("weights_libreface/Facial_Expression_Recognition/weights/resnet.pt",
                      map_location=device)
    fer_model.load_state_dict(ckpt["model"])
    fer_model.eval()
    fer_solver = solver_inference_image( student_model=fer_model, device=device, image_size=224)
    # Mediapipe
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="face_land/face_landmarker.task"),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1)
    landmarker = FaceLandmarker.create_from_options(options)
    return fer_solver, landmarker, get_aligned_image
fer_solver, landmarker, get_aligned_image = load_models()
# ===================== UTILS =====================
idx_to_fe = {
    0: "Neutral",
    1: "Happiness",
    2: "Sadness",
    3: "Surprise",
    4: "Fear",
    5: "Disgust",
    6: "Anger",
    7: "Contempt"}
def crop_face(img, results):
    if results.face_landmarks:
        h, w = img.shape[:2]
        l_m = results.face_landmarks[0]
        pts = np.array([(int(l.x * w), int(l.y * h)) for l in l_m])
        x, y = pts[:, 0], pts[:, 1]
        return img[y.min():y.max(), x.min():x.max()]
    else:
        return img
def color(pred):
    if pred in ['Anger', 'Disgust']:
        return 'Negative', (0, 0, 255)
    elif pred in ['Happiness', 'Neutral']:
        return 'Positive', (0, 255, 0)
    elif pred in ['Surprise', 'Fear']:
        return 'Stress', (0, 175, 255)
    elif pred in ['Sadness', 'Contempt']:
        return 'Worry', (0, 120, 255)
def plot_percentages(red_cls):
    counter = Counter(red_cls)
    labels = list(counter.keys())
    values = [counter[l] for l in labels]
    colors_map = {
        'Positive': (0,1,0),
        'Negative': (1,0,0),
        'Stress': (1,0.7,0),
        'Worry': (1,0.47,0)}
    colors = [colors_map[l] for l in labels]
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%',colors=colors, startangle=90)
    ax.axis('equal')
    ax.set_title("Emotion Presence (%)")
    st.pyplot(fig)
# ===================== UI =====================
source = st.radio("Select Input Source", ["Camera", "Upload Video"])
FRAME_WINDOW = 7
# ---------- STOP BUTTON STATE ----------
if "red_cls" not in st.session_state:
    st.session_state.red_cls = []
if "stable_emotion" not in st.session_state:
    st.session_state.stable_emotion = None
if "stop" not in st.session_state:
    st.session_state.stop = False
stop_btn = st.button("🛑 Stop")
if stop_btn:
    st.session_state.stop = True
FRAME_WINDOW = 5
# ================= UPLOAD VIDEO =================
if source == "Upload Video":
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())
        cap = cv2.VideoCapture(tfile.name)
        frame_slot = st.empty()
        emotion_window = []
        stable_emotion = None
        red_cls = []
        #st.session_state.red_cls = []
        while cap.isOpened() and not st.session_state.stop:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = landmarker.detect(mp_image)
            if results.face_landmarks:
                align = get_aligned_image(rgb, results, verbose=False)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=align)
                results = landmarker.detect(mp_image)
                face = crop_face(align, results)
                idx, conf = fer_solver.run(face)
                emotion_window.append(idx_to_fe[idx])
                # 🔥 FIFO SLIDING WINDOW
                if len(emotion_window) > FRAME_WINDOW:
                    emotion_window.pop(0)
                stable_emotion = Counter(emotion_window).most_common(1)[0][0]
            # 🔥 DRAW EVERY FRAME
            if stable_emotion is not None:
                emo_red, col = color(stable_emotion)
                st.session_state.red_cls.append(emo_red)
                cv2.putText(frame, emo_red, (30, 40),cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)
                cv2.putText(frame, stable_emotion, (30, 80),cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)
            frame_slot.image(frame, channels="BGR")
        cap.release()
        st.session_state.stop = False
        if len(st.session_state.red_cls) > 0:
            plot_percentages(st.session_state.red_cls)
        else:
            st.warning("No emotions detected.")
# ================= CAMERA =================
elif source == "Camera":
    if "camera_on" not in st.session_state:
        st.session_state.camera_on = False
    if "stop" not in st.session_state:
        st.session_state.stop = False
    if "red_cls" not in st.session_state:
        st.session_state.red_cls = []
    if "show_results" not in st.session_state:
        st.session_state.show_results = False
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Camera"):
            st.session_state.camera_on = True
            st.session_state.stop = False
            st.session_state.red_cls = []
            st.session_state.show_results = False
    with col2:
        if st.button("Stop Camera"):
            st.session_state.stop = True
            st.session_state.camera_on = False
            st.session_state.show_results = True
    # -------- CAMERA LOOP --------
    if st.session_state.camera_on:
        cap = cv2.VideoCapture(0)
        frame_slot = st.empty()
        emotion_window = []
        stable_emotion = None
        while cap.isOpened() and not st.session_state.stop:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = landmarker.detect(mp_image)
            if results.face_landmarks:
                align = get_aligned_image(rgb, results, verbose=False)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=align)
                results = landmarker.detect(mp_image)
                face = crop_face(align, results)
                idx, conf = fer_solver.run(face)
                emotion_window.append(idx_to_fe[idx])
                # 🔥 FIFO SLIDING WINDOW
                if len(emotion_window) > FRAME_WINDOW:
                    emotion_window.pop(0)
                stable_emotion = Counter(emotion_window).most_common(1)[0][0]
            if stable_emotion is not None:
                emo_red, col = color(stable_emotion)
                st.session_state.red_cls.append(emo_red)
                cv2.putText(frame, emo_red, (30, 40),cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)
                cv2.putText(frame, stable_emotion, (30, 80),cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)
            frame_slot.image(frame, channels="BGR")
        cap.release()
    # -------- SHOW RESULTS AFTER STOP --------
    if st.session_state.show_results:
        if len(st.session_state.red_cls) > 0:
            plot_percentages(st.session_state.red_cls)
        else:
            st.warning("No emotions detected.")
        st.session_state.show_results = False