import cv2
import numpy as np
import mediapipe as mp
from tracking_utils import *
# all your existing helpers live here

# ---------- MediaPipe init ----------
model_path = "face_land/face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1)
landmarker = FaceLandmarker.create_from_options(options)


LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 246, 161, 163, 144, 145, 153, 154, 155]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 466, 388, 390, 373, 374, 380, 381, 382]

L_EYE_LEFT_CORNER  = 33
L_EYE_RIGHT_CORNER = 133
R_EYE_LEFT_CORNER  = 362
R_EYE_RIGHT_CORNER = 263

LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

BLINK_THRESHOLD = 0.25


def process_frame(
    frame,
    eyes_hist,
    dir_hist,
    pos_hist,
    depth_hist,
    face_det):
    frame = cv2.resize(frame, (720, 640))
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    panel_w = w // 2
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)

    # ---------- NO FACE ----------
    if not result.face_landmarks:
        face_det.append("face doesn't exist")
        cv2.putText(
            panel, "No face", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        return np.concatenate((frame, panel), axis=1)

    # ---------- FACE EXISTS ----------
    lm = result.face_landmarks[0]
    pts = np.array([(int(l.x * w), int(l.y * h)) for l in lm])

    fram = frame.copy()
    draw_eye_landmarks(fram, lm, w, h)
    # -------- HEAD --------
    distance, d, hpos, hp, vpos, v = head_pos(pts, w, h)
    horiz, ox, vert, oy = head_dir( pts, w, h,horiz_thresh=0.25,vert_thresh=0.1)
    # -------- BLINK --------
    blinking = is_blinking(lm, w, h, .25)
    # -------- GAZE --------
    if blinking:
        eye_dir = "Blinking"
        gaze_norm, n, s = 0, 0, 0
    else:
        eye_dir, gaze_norm, n, lp, rp, s = estimate_gaze(
            lm, w, h, ox,
            LEFT_IRIS, RIGHT_IRIS,
            L_EYE_LEFT_CORNER, L_EYE_RIGHT_CORNER,
            R_EYE_LEFT_CORNER, R_EYE_RIGHT_CORNER,
            pts)
        cv2.circle(fram, tuple(lp.astype(int)), 2, (0, 0, 255), -1)
        cv2.circle(fram, tuple(rp.astype(int)), 2, (0, 0, 255), -1)

    # -------- FACE CROP --------
    xs = [int(pt.x * w) for pt in lm]
    ys = [int(pt.y * h) for pt in lm]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    face_crop = fram[max(y1, 0):y2, max(x1, 0):x2].copy()
    face_crop = cv2.resize(face_crop, (panel_w - 20, h // 2 - 20))
    panel[10:h // 2 - 10, 10:panel_w - 10] = face_crop

    # -------- COLORS --------
    head_col = head_dir_color(vert, horiz)
    dist_col = dist_color(distance)
    pos_col = pos_color(hpos, vpos)
    eye_col = eye_dir_color(eye_dir)

    # -------- STORE HISTORY --------
    eyes_hist.append(eye_dir if not blinking else "blinking")
    dir_hist.append((vert, horiz))
    pos_hist.append((hpos, vpos))
    depth_hist.append(distance)
    face_det.append("face exists")

    # -------- TEXT --------
    y_text = h // 2 + 20

    cv2.putText(panel, "Distance:", (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (250, 250, 250), 2)
    y_text += 20
    cv2.putText(panel, f"{distance}: {d:.2f}", (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, dist_col, 2)

    y_text += 40
    cv2.putText(panel, "Position:", (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (250, 250, 250), 2)
    y_text += 20
    cv2.putText(panel, f"Y:{hpos}: {hp:.2f}", (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, pos_col, 2)
    y_text += 20
    cv2.putText(panel, f"X:{vpos}: {v:.2f}", (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, pos_col, 2)

    y_text += 40
    cv2.putText(panel, "Direction (Hor,Ver):", (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (250, 250, 250), 2)
    y_text += 20
    cv2.putText(panel, f"Hor: {horiz}:{ox:.2f}", (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, head_col, 2)
    y_text += 20
    cv2.putText(panel, f"Ver: {vert}:{oy:.2f}", (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, head_col, 2)

    y_text += 40
    cv2.putText(panel, "Eyes direction", (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (250, 250, 250), 2)
    y_text += 20
    cv2.putText(panel, f"{eye_dir} : {gaze_norm:.2f}", (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, eye_col, 2)
    y_text += 20
    cv2.putText(panel, f"{n:.2f} / {s:.2f}", (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, eye_col, 2)
    return np.concatenate((frame, panel), axis=1)