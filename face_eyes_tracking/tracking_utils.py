import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 246, 161, 163, 144, 145, 153, 154, 155]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 466, 388, 390, 373, 374, 380, 381, 382]

L_EYE_LEFT_CORNER  = 33
L_EYE_RIGHT_CORNER = 133
R_EYE_LEFT_CORNER  = 362
R_EYE_RIGHT_CORNER = 263

LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

BLINK_THRESHOLD = 0.25


def head_pos(pts, w, h):
    """pts: (468, 2) numpy array of face landmarks in pixel coordinates    """
    x = pts[:, 0]
    y = pts[:, 1]
    # ---- Face extremes ----
    left_x   = x.min()
    right_x  = x.max()
    top_y    = y.min()
    bottom_y = y.max()
    # ---- Face center ----
    face_cx = (left_x + right_x)# / 2
    face_cy = (top_y + bottom_y)# / 2
    frame_cx = w# / 2
    frame_cy = h# / 2
    # ---- Face size ----
    face_width  = right_x - left_x
    face_height = bottom_y - top_y
    # ---- NORMALIZED offsets ----
    offset_x = (face_cx - frame_cx) / face_width
    offset_y = (face_cy - frame_cy) / face_height
    # ---- Distance (scale-invariant) ----
    face_height_ratio = face_height / h
    if face_height_ratio > 0.45:
        distance = "Close"
    elif face_height_ratio < 0.19:
        distance = "Far"
    else:
        distance = "Middle"
    # ---- Horizontal position ----
    if offset_x < -(.5/face_height_ratio): # 0.9:
        position = "Left"
    elif offset_x > (.5/face_height_ratio): #0.9:
        position = "Right"
    else:
        position = "Center"
    # ---- Vertical position ----
    if offset_y < -(.35/face_height_ratio): #0.5:
        hposition = "Up"
    elif offset_y > (.5/face_height_ratio): #0.75:
        hposition = "Down"
    else:
        hposition = "Center"
    return distance, face_height_ratio, hposition, offset_y, position, offset_x



def head_dir(pts, w, h, horiz_thresh=0.2, vert_thresh=0.1):
    x = pts[:, 0]
    y = pts[:, 1]
    # ---- Face extremes & center ----
    left_x, right_x = x.min(), x.max()
    top_y, bottom_y = y.min(), y.max()
    face_cx = (left_x + right_x) / 2
    face_cy = (top_y + bottom_y) / 2
    # ---- Reference points (nose / eyes) ----
    nose = pts[1]  # MediaPipe nose tip
    nose_x, nose_y = nose
    # ---- Horizontal direction ----
    offset_x = (nose_x - face_cx) / (right_x - left_x)  # normalized
    if offset_x < - .33:  #horiz_thresh * 1.5:
        horiz = "far Right"
    elif offset_x < -horiz_thresh:
        horiz = "Right"
    elif offset_x > .33:  #horiz_thresh * 1.5:
        horiz = "far Left"
    elif offset_x > horiz_thresh:
        horiz = "Left"
    else:
        horiz = "Center"
    # ---- Vertical direction ----
    offset_y = (nose_y - face_cy) / (bottom_y - top_y)
    if offset_y < -vert_thresh/2:
        vert = "Up"
    elif offset_y > vert_thresh*1.5:
        vert = "Down"
    else:
        vert = "Center"
    return horiz, offset_x, vert, offset_y



def eye_openness(lm, h_pair, v_pair, w, h):
    hx1, hx2 = h_pair
    vx1, vx2 = v_pair
    horiz = abs(lm[hx1].x - lm[hx2].x) * w
    vert  = abs(lm[vx1].y - lm[vx2].y) * h
    return vert / horiz if horiz != 0 else 0

def is_blinking(lm, w, h, threshold):
    left_open  = eye_openness(lm, (33,133), (159,145), w, h)
    right_open = eye_openness(lm, (362,263), (386,374), w, h)
    s = ((left_open + right_open) / 2)
    return s < threshold
    
def draw_eye_landmarks(frame, lm, w, h):
    for idx in LEFT_EYE + RIGHT_EYE:
        x = int(lm[idx].x * w)
        y = int(lm[idx].y * h)
        cv2.circle(frame, (x,y), 1, (0,255,0), -1)


gaze_buffer = []
GAZE_BUFFER_SIZE = 5
prev_smoothed = None
previous_gaze = "Center"


def estimate_gaze(lm, w, h, ox, LEFT_IRIS, RIGHT_IRIS, L_EYE_LEFT_CORNER, L_EYE_RIGHT_CORNER, R_EYE_LEFT_CORNER, R_EYE_RIGHT_CORNER, face_pts):
    global gaze_buffer, prev_smoothed, previous_gaze
    # --- Compute pupil center ---
    left_pupil = np.mean([[lm[i].x * w, lm[i].y * h] for i in LEFT_IRIS], axis=0)
    right_pupil = np.mean([[lm[i].x * w, lm[i].y * h] for i in RIGHT_IRIS], axis=0)

    l_left, l_right = lm[L_EYE_LEFT_CORNER].x*w, lm[L_EYE_RIGHT_CORNER].x*w
    r_left, r_right = lm[R_EYE_LEFT_CORNER].x*w, lm[R_EYE_RIGHT_CORNER].x*w

    norm_L = (left_pupil[0] - l_left) / (l_right - l_left + 1e-6)
    norm_R = (right_pupil[0] - r_left) / (r_right - r_left + 1e-6)
    norm_avg = (norm_L + norm_R) / 2

    # -------- BUFFER UPDATE (ONLY WHEN HEAD IS STABLE) --------
    if abs(ox) <= 0.25:
        gaze_buffer.append(norm_avg)
        if len(gaze_buffer) > GAZE_BUFFER_SIZE:
            gaze_buffer.pop(0)

    smooth = float(np.mean(gaze_buffer)) if gaze_buffer else norm_avg

    # -------- VELOCITY-AWARE REJECTION --------
    if prev_smoothed is not None:
        if abs(smooth - prev_smoothed) > 0.15:
            gaze_world_norm = prev_smoothed + ox
            return previous_gaze, gaze_world_norm, norm_avg, left_pupil, right_pupil, ox

    gaze_world_norm = smooth + ox

    # -------- CLASSIFICATION --------
    if abs(ox) <= 0.15:
        if smooth < 0.42:
            gaze_world = "Right"
        elif smooth > 0.58:
            gaze_world = "Left"
        else:
            gaze_world = "Center"

    elif abs(ox) <= 0.25:
        if smooth < 0.37:
            gaze_world = "Right"
        elif smooth > 0.62:
            gaze_world = "Left"
        else:
            gaze_world = "Center"
    else:
        gaze_world = "Head Turned"        
    # -------- MEMORY UPDATE --------
    prev_smoothed = smooth
    previous_gaze = gaze_world
    return gaze_world, gaze_world_norm, norm_avg, left_pupil, right_pupil, ox


def pos_color(hposition, position):
    if hposition == 'Center' and  position == 'Center':
        return (0,255,0)
    else:
        return  (0,150,255)
def dist_color(far):
    if far == 'Middle':
        return (0,255,0)
    else:
        return  (0,150,255)
        
def head_dir_color(vert, horiz):
    if vert == 'Center' and horiz=='Center':
        return (0,255,0)
    elif horiz== 'far Right' or  horiz== 'far Left':
        return  (0,0,255)
    else:
        return  (0,150,255)

def eye_dir_color(eye_dir):
    if eye_dir == 'Center':
        return (0,255,0)
    elif eye_dir == "Blinking":
        return (0,150,255)
    else:
        return  (0,0,255)