# visualization.py
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import streamlit as st


LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 246, 161, 163, 144, 145, 153, 154, 155]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 466, 388, 390, 373, 374, 380, 381, 382]

L_EYE_LEFT_CORNER  = 33
L_EYE_RIGHT_CORNER = 133
R_EYE_LEFT_CORNER  = 362
R_EYE_RIGHT_CORNER = 263

LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

BLINK_THRESHOLD = 0.25

def visualize_tracking_data(
    eye_hist=None,
    dir_hist=None,
    pos_hist=None,
    depth_hist=None,
    title_prefix="Tracking Data"):
    if not eye_hist:
        st.warning("No tracking data available.")
        return
    frames = np.arange(len(eye_hist))
    # ---------- Eye gaze timeline ----------
    gaze_map = {"Left": -1, "Center": 0, "Right": 1}
    color_map = {
        "Left": "orange",
        "Center": "green",
        "Right": "orange",
        "blinking": "blue",
        "Head Turned": "red"}

    gaze_numeric = np.array([gaze_map.get(e, np.nan) for e in eye_hist])

    fig, ax = plt.subplots(figsize=(16, 4))
    # Base trajectory (neutral black line)
    ax.plot(frames, gaze_numeric, color="black", linewidth=1.5, alpha=0.3)

    # Overlay colored states
    for state, color in color_map.items():
        idx = [i for i, e in enumerate(eye_hist) if e == state]
        if state in gaze_map:
            y = gaze_map[state]
            ax.scatter(idx, [y] * len(idx), color=color, s=20, label=state)
        else:
            # blinking / head turned → plot on center line
            ax.scatter(idx, [0] * len(idx), color=color, s=20, label=state)

    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Left", "Center", "Right"])
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Eye state")
    ax.set_title(f"{title_prefix}: Eye Gaze Timeline", fontsize=18)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")
    st.pyplot(fig)

    # ---------- Head direction timeline ----------
    if dir_hist:
        horiz_map = {"far Left": -2, "Left": -1, "Center": 0, "Right": 1, "far Right": 2}
        color_map_hd = {"far Left": "red", "Left": "orange", "Center": "green",
                        "Right": "orange", "far Right": "red"}

        frames_hd = np.arange(len(dir_hist))
        horiz_vals = np.array([horiz_map[h[1]] for h in dir_hist])

        fig, ax = plt.subplots(figsize=(16, 4))
        # Base line
        ax.plot(frames_hd, horiz_vals, color="black", linewidth=1.5, alpha=0.3)

        # Overlay colored states
        for label, val in horiz_map.items():
            idx = [i for i, h in enumerate(dir_hist) if h[1] == label]
            ax.scatter(idx, [val] * len(idx), s=20, color=color_map_hd[label], label=label)

        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.set_yticklabels(["far L", "L", "Center", "R", "far R"])
        ax.set_xlabel("Frame")
        ax.set_ylabel("Head Dir")
        ax.set_title(f"{title_prefix}: Head Direction Timeline", fontsize=18)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper right")
        st.pyplot(fig)

    # ---------- Eyes pie chart ----------
    counts = Counter(eye_hist)
    labels = list(counts.keys())
    sizes = list(counts.values())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=180,
           wedgeprops={"edgecolor": "black"})
    ax.set_title(f"{title_prefix}: Eye Gaze & Blink Distribution", fontsize=18)
    ax.axis("equal")
    st.pyplot(fig)

    # ---------- Face Direction pie chart ----------
    if dir_hist:
        dir_labels = [f"Y:{y} X:{x}" for y, x in dir_hist]
        counts = Counter(dir_labels)
        labels = list(counts.keys())
        sizes = list(counts.values())
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor": "black"})
        ax.set_title(f"{title_prefix}: Face Directions Distribution", fontsize=18)
        ax.axis("equal")
        st.pyplot(fig)
    # ---------- Face position pie chart ----------
    if pos_hist:
        pos_labels = [f"Y:{y} X:{x}" for y, x in pos_hist]
        counts = Counter(pos_labels)
        labels = list(counts.keys())
        sizes = list(counts.values())
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90,wedgeprops={"edgecolor": "black"})
        ax.set_title(f"{title_prefix}: Face Positions Distribution", fontsize=18)
        ax.axis("equal")
        st.pyplot(fig)
    # ---------- Depth pie chart ----------
    if depth_hist:
        counts = Counter(depth_hist)
        labels = list(counts.keys())
        sizes = list(counts.values())
        #colors = ["limegreen", "orange", "crimson"][:len(labels)]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor": "black"})
        ax.set_title(f"{title_prefix}: Face Distance Distribution", fontsize=18)
        ax.axis("equal")
        st.pyplot(fig)