import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
import tempfile
import os
import random
from collections import Counter

from rep_counter import count_curl_reps, count_press_reps, count_squat_reps
from rules import curl_form, press_form, squat_form

APP_NAME = "RefynD"
APP_SUBTITLE = "Your personal AI form assistant"

YOLO_POSE_PATH = "weights/yolov8n-pose.pt"
CLASSIFIER_PATH = "weights/exercise_cls_3class_fixed.pt"
SEQ_LEN = 30


class ExerciseBiLSTM(nn.Module):
    def __init__(self, in_dim=55, hidden=96, num_classes=3, dropout=0.35):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :]))


def get_biomechanical_features(kpts):
    hip_center = (kpts[11, :2] + kpts[12, :2]) / 2.0
    rel_coords = kpts.copy()
    rel_coords[:, :2] = kpts[:, :2] - hip_center

    shoulder_width = np.linalg.norm(kpts[5, :2] - kpts[6, :2]) + 1e-6
    rel_coords[:, :2] /= shoulder_width

    def angle(a, b, c):
        ba = a - b
        bc = c - b
        denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
        cos_a = np.dot(ba, bc) / denom
        cos_a = np.clip(cos_a, -1.0, 1.0)
        return np.degrees(np.arccos(cos_a))

    l_elbow = angle(rel_coords[5, :2], rel_coords[7, :2], rel_coords[9, :2])
    r_elbow = angle(rel_coords[6, :2], rel_coords[8, :2], rel_coords[10, :2])
    l_knee = angle(rel_coords[11, :2], rel_coords[13, :2], rel_coords[15, :2])
    r_knee = angle(rel_coords[12, :2], rel_coords[14, :2], rel_coords[16, :2])

    angles = np.array([l_elbow, r_elbow, l_knee, r_knee], dtype=np.float32) / 180.0
    return np.concatenate([rel_coords.flatten(), angles]).astype(np.float32)


@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pose_model = YOLO(YOLO_POSE_PATH)

    model = ExerciseBiLSTM(in_dim=55, hidden=96, num_classes=3).to(device)
    ckpt = torch.load(CLASSIFIER_PATH, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return pose_model, model, device


def process_video(video_path, pose_model, classifier_model, device):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        return None

    all_kpts = []
    confs = []

    for frame in frames:
        h, w = frame.shape[:2]
        results = pose_model(frame, verbose=False)[0]

        if results.keypoints is None or len(results.keypoints.xy) == 0:
            all_kpts.append(np.zeros((17, 3), dtype=np.float32))
            confs.append(0)
            continue

        i = 0
        xy = results.keypoints.xy[i].cpu().numpy().astype(np.float32)
        cf = results.keypoints.conf[i].cpu().numpy().astype(np.float32)

        xy[:, 0] /= max(w, 1)
        xy[:, 1] /= max(h, 1)

        kpts = np.concatenate([xy, cf[:, None]], axis=1).astype(np.float32)
        all_kpts.append(kpts)
        confs.append(np.mean(cf))

    idxs = np.linspace(0, len(all_kpts) - 1, SEQ_LEN).astype(int)
    features = [get_biomechanical_features(all_kpts[i]) for i in idxs]

    x = torch.tensor(np.stack(features), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(classifier_model(x), dim=1).cpu().numpy()[0]

    pred = int(np.argmax(probs))
    ex_names = {0: "CURL", 1: "PRESS", 2: "SQUAT"}
    exercise = ex_names[pred]

    if pred == 0:
        reps, _ = count_curl_reps(all_kpts)
        form_fn = curl_form
    elif pred == 1:
        reps, _ = count_press_reps(all_kpts)
        form_fn = press_form
    else:
        reps, _ = count_squat_reps(all_kpts)
        form_fn = squat_form

    all_feedback = []
    for i in range(0, len(all_kpts), 10):
        msgs = form_fn(all_kpts[i])
        if msgs:
            all_feedback.extend(msgs)

    if all_feedback:
        msg_counts = Counter(all_feedback)
        feedback = [msg for msg, count in msg_counts.most_common(2)]
    else:
        good_msgs = ["Good form", "Looking good", "Nice work", "Perfect form"]
        feedback = [random.choice(good_msgs)]

    duration = len(frames) / 30.0
    tempo = duration / reps if reps and reps > 0 else None

    return {
        "exercise": exercise,
        "confidence": {
            "curl": float(probs[0]),
            "press": float(probs[1]),
            "squat": float(probs[2])
        },
        "reps": reps if reps else 0,
        "duration": round(duration, 1),
        "tempo": round(tempo, 2) if tempo else None,
        "feedback": feedback,
        "pose_confidence": round(np.mean(confs), 2)
    }


st.set_page_config(
    page_title=APP_NAME,
    page_icon="🏋️",
    layout="wide"
)

st.markdown("""
<style>

/* MAIN HEADER */
.main-header {
    text-align: center;
    padding: 1.5rem;
    background: linear-gradient(135deg, #053264 0%, #0a4a8a 50%, #ccffbc 100%);
    border-radius: 15px;
    margin-bottom: 2rem;
}

.main-header h1 {
    color: #ffffff;
    margin: 0;
    font-size: 2.5rem;
    letter-spacing: 1px;
}

.main-header p {
    color: rgba(255,255,255,0.85);
    margin: 0.5rem 0 0 0;
}

/* METRIC CARDS */
.metric-card {
    background: #ffffff;
    padding: 1rem;
    border-radius: 12px;
    text-align: center;
    border: 2px solid #05326420;
}

/* NUMBERS */
.metric-number {
    font-size: 2rem;
    font-weight: bold;
    margin: 0;
    color: #053264;
}

/* LABELS */
.metric-label {
    font-size: 0.8rem;
    color: #6b7280;
}

/* GOOD FEEDBACK */
.feedback-good {
    background-color: #ccffbc;
    padding: 1rem;
    border-radius: 10px;
    color: #053264;
    font-weight: 600;
    text-align: center;
}

/* BAD FEEDBACK */
.feedback-bad {
    background-color: #053264;
    padding: 1rem;
    border-radius: 10px;
    color: #ffffff;
    font-weight: 600;
    text-align: center;
}

/* PROGRESS BAR */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #053264, #ccffbc) !important;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #053264;
    color: white;
}

/* BUTTON */
.stButton button {
    background-color: #053264;
    color: white;
    border-radius: 8px;
}

.stButton button:hover {
    background-color: #0a4a8a;
}

</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="main-header">
    <h1>{APP_NAME}</h1>
    <p>{APP_SUBTITLE}</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## About")
    st.markdown(f"""
    **{APP_NAME}** uses artificial intelligence to analyze your exercise form and provide feedback from uploaded exercise videos.
    
    **Supported exercises:**
    - Bicep Curls
    - Shoulder Press
    - Squats
    
    **What it tracks:**
    - Exercise classification
    - Rep counting
    - Form quality
    - Movement tempo
    """)

    st.markdown("---")
    st.markdown("### Tips")
    st.markdown("""
    - Upload a clear video
    - Ensure full body is visible
    - Use good lighting
    - Keep the camera stable
    """)

    st.markdown("---")
    st.markdown("*Built for your fitness journey*")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Upload Video")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])

    video_path = None
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        st.video(video_path)

with col2:
    st.markdown("### Results")

    if video_path:
        with st.spinner("Analyzing your movement..."):
            pose_model, classifier_model, device = load_models()
            result = process_video(video_path, pose_model, classifier_model, device)

            try:
                os.unlink(video_path)
            except Exception:
                pass

            if result:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Detected Exercise</p>
                    <p class="metric-number">{result['exercise']}</p>
                </div>
                """, unsafe_allow_html=True)

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Reps</p>
                        <p class="metric-number">{result['reps']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Duration</p>
                        <p class="metric-number">{result['duration']}s</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col_c:
                    tempo_text = f"{result['tempo']}s/rep" if result['tempo'] else "--"
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Tempo</p>
                        <p class="metric-number">{tempo_text}</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("#### Classification Confidence")
                for ex, conf in result["confidence"].items():
                    st.progress(conf, text=f"{ex.upper()}: {int(conf * 100)}%")

                st.markdown("#### Form Feedback")
                feedback_text = result["feedback"][0]
                is_good = any(word in feedback_text.lower() for word in ["good", "nice", "perfect", "looking"])

                if is_good:
                    st.markdown(f'<div class="feedback-good">✓ {feedback_text}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="feedback-bad">! {feedback_text}</div>', unsafe_allow_html=True)

                if len(result["feedback"]) > 1:
                    st.info(f"Tip: {result['feedback'][1]}")

                st.caption(f"Pose detection confidence: {int(result['pose_confidence'] * 100)}%")
            else:
                st.error("Could not process the video. Please make sure your full body is visible.")

st.markdown("---")
st.markdown(f"<center><small>{APP_NAME} | Computer Vision Exercise Analysis</small></center>", unsafe_allow_html=True)