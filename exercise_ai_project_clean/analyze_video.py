import os
import json
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
import random
from collections import Counter

from pose_utils import load_video_frames, extract_kpts, sample_sequence_for_classifier
from rep_counter import count_curl_reps, count_press_reps, count_squat_reps
from rules import curl_form, press_form, squat_form

YOLO_POSE_PATH = "weights/yolov8n-pose.pt"
CLASSIFIER_PATH = "weights/exercise_cls_3class_fixed.pt"

SEQ_LEN = 30

IDX_TO_EX = {
    0: "CURL",
    1: "PRESS",
    2: "SQUAT"
}


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


def load_models():
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    pose_model = YOLO(YOLO_POSE_PATH)

    clf = ExerciseBiLSTM().to(device)
    ckpt = torch.load(CLASSIFIER_PATH, map_location=device)
    clf.load_state_dict(ckpt["state_dict"])
    clf.eval()

    return pose_model, clf, device


def classify_sequence(seq, clf, device):
    features = np.array([get_biomechanical_features(f) for f in seq], dtype=np.float32)
    x = torch.from_numpy(features).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(clf(x), dim=1).cpu().numpy()[0]

    pred = int(np.argmax(probs))
    return pred, probs


def classify_video_windows(all_kpts, clf, device, seq_len=30, num_windows=5):
    n = len(all_kpts)

    if n == 0:
        return 0, np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)

    if n < seq_len:
        seq = sample_sequence_for_classifier(all_kpts, seq_len=seq_len)
        return classify_sequence(seq, clf, device)

    starts = np.linspace(0, n - seq_len, num_windows).astype(int)

    probs_list = []
    for s in starts:
        window = all_kpts[s:s + seq_len]
        if len(window) != seq_len:
            continue

        seq = np.stack(window, axis=0).astype(np.float32)
        _, probs = classify_sequence(seq, clf, device)
        probs_list.append(probs)

    if len(probs_list) == 0:
        seq = sample_sequence_for_classifier(all_kpts, seq_len=seq_len)
        return classify_sequence(seq, clf, device)

    probs = np.mean(np.stack(probs_list, axis=0), axis=0)
    pred = int(np.argmax(probs))
    return pred, probs


def smooth_signal(x, alpha=0.3):
    if len(x) == 0:
        return np.array([], dtype=np.float32)

    out = [float(x[0])]
    for i in range(1, len(x)):
        out.append(alpha * float(x[i]) + (1.0 - alpha) * out[-1])
    return np.array(out, dtype=np.float32)


def simple_angle(a, b, c):
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cosang = np.dot(ba, bc) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def get_curl_key_frames(all_kpts):
    left_angles, right_angles = [], []

    for k in all_kpts:
        left_angles.append(simple_angle(k[5, :2], k[7, :2], k[9, :2]))
        right_angles.append(simple_angle(k[6, :2], k[8, :2], k[10, :2]))

    signal = smooth_signal(left_angles if np.std(left_angles) >= np.std(right_angles) else right_angles)

    if len(signal) < 3:
        return []

    top_thr = np.percentile(signal, 30)
    bot_thr = np.percentile(signal, 75)

    key_idxs = []
    for i in range(1, len(signal) - 1):
        if signal[i] <= top_thr and signal[i] <= signal[i - 1] and signal[i] <= signal[i + 1]:
            key_idxs.append(i)
        if signal[i] >= bot_thr and signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1]:
            key_idxs.append(i)

    return sorted(set(key_idxs))


def get_press_key_frames(all_kpts):
    wrist_y = [float((k[9, 1] + k[10, 1]) / 2.0) for k in all_kpts]
    signal = smooth_signal(wrist_y)

    if len(signal) < 3:
        return []

    top_thr = np.percentile(signal, 30)
    bot_thr = np.percentile(signal, 70)

    key_idxs = []
    for i in range(1, len(signal) - 1):
        if signal[i] <= top_thr and signal[i] <= signal[i - 1] and signal[i] <= signal[i + 1]:
            key_idxs.append(i)
        if signal[i] >= bot_thr and signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1]:
            key_idxs.append(i)

    return sorted(set(key_idxs))


def get_squat_key_frames(all_kpts):
    knee_angles = [
        (simple_angle(k[11, :2], k[13, :2], k[15, :2]) +
         simple_angle(k[12, :2], k[14, :2], k[16, :2])) / 2.0
        for k in all_kpts
    ]

    signal = smooth_signal(knee_angles)

    if len(signal) < 3:
        return []

    low_thr = np.percentile(signal, 30)
    high_thr = np.percentile(signal, 70)

    key_idxs = []
    for i in range(1, len(signal) - 1):
        if signal[i] <= low_thr and signal[i] <= signal[i - 1] and signal[i] <= signal[i + 1]:
            key_idxs.append(i)
        if signal[i] >= high_thr and signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1]:
            key_idxs.append(i)

    return sorted(set(key_idxs))


def analyze_video(video_path):
    pose_model, clf, device = load_models()

    frames, fps = load_video_frames(video_path)
    if len(frames) == 0:
        return {"error": "empty video"}

    all_kpts, confs = [], []

    for frame in frames:
        k, mean_conf, _ = extract_kpts(frame, pose_model)
        all_kpts.append(k)
        confs.append(mean_conf)

    pred, probs = classify_video_windows(all_kpts, clf, device)

    avg_conf = float(np.mean(confs))
    margin = float(np.max(probs) - np.sort(probs)[-2])
    max_prob = float(np.max(probs))
    duration_sec = len(frames) / max(fps, 1e-6)

    if max_prob < 0.50 and margin < 0.05:
        return {"exercise": "UNCERTAIN", "confidence": probs.tolist()}

    ex_name = IDX_TO_EX[pred]

    if pred == 0:
        reps, _ = count_curl_reps(all_kpts)
        form_fn = curl_form
        key_idxs = get_curl_key_frames(all_kpts)
    elif pred == 1:
        reps, _ = count_press_reps(all_kpts)
        form_fn = press_form
        key_idxs = get_press_key_frames(all_kpts)
    else:
        reps, _ = count_squat_reps(all_kpts)
        form_fn = squat_form
        key_idxs = get_squat_key_frames(all_kpts)

    all_feedback = []

    if form_fn and len(key_idxs) > 0:
        for i in key_idxs:
            msgs = form_fn(all_kpts[i])
            if msgs:
                all_feedback.extend(msgs)

        if len(all_feedback) > 0:
            msg_counts = Counter(all_feedback)
            persistent = [m for m, c in msg_counts.items() if c >= 2]

            if persistent:
                feedback = [persistent[0]]
                bad = 1
            else:
                feedback = [random.choice(["Good form!", "Looking good!", "Nice work!"])]
                bad = 0
        else:
            feedback = [random.choice(["Good form!", "Looking good!", "Nice work!"])]
            bad = 0
    else:
        feedback = [random.choice(["Good form!", "Looking good!", "Nice work!"])]
        bad = 0

    tempo = duration_sec / reps if reps and reps > 0 else None

    return {
        "video": os.path.basename(video_path),
        "exercise": ex_name,
        "confidence": {
            "curl": float(probs[0]),
            "press": float(probs[1]),
            "squat": float(probs[2]),
            "avg_pose_conf": avg_conf
        },
        "summary": {
            "reps": int(reps) if reps is not None else None,
            "duration_sec": round(duration_sec, 2),
            "tempo": round(tempo, 2) if tempo is not None else None,
            "form_warnings": int(bad)
        },
        "feedback": feedback
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python analyze_video.py <video_path>")

    video_path = sys.argv[1]
    report = analyze_video(video_path)
    print(json.dumps(report, indent=2))