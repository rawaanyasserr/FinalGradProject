import cv2
import numpy as np
from ultralytics import YOLO

CONF_THR = 0.20
SEQ_LEN = 30

def zeros_kpts():
    return np.zeros((17, 3), dtype=np.float32)

def pick_best_person(r):
    try:
        if r.boxes is None or len(r.boxes) == 0:
            return 0
        confs = r.boxes.conf.detach().cpu().numpy()
        return int(np.argmax(confs))
    except Exception:
        return 0

def extract_kpts(frame, pose_model):
    h, w = frame.shape[:2]
    r = pose_model(frame, verbose=False)[0]

    if r.keypoints is None or len(r.keypoints.xy) == 0:
        return zeros_kpts(), 0.0, False

    i = pick_best_person(r)
    xy = r.keypoints.xy[i].cpu().numpy().astype(np.float32)
    cf = r.keypoints.conf[i].cpu().numpy().astype(np.float32)

    xy[:, 0] /= max(w, 1)
    xy[:, 1] /= max(h, 1)

    m = cf >= CONF_THR
    xy[~m] = 0.0
    cf[~m] = 0.0

    vis = cf[cf > 0]
    mean_conf = float(vis.mean()) if vis.size else 0.0

    kpts = np.concatenate([xy, cf[:, None]], axis=1).astype(np.float32)
    reliable = mean_conf >= 0.18
    return kpts, mean_conf, reliable

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 30.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)

    cap.release()
    return frames, fps

def sample_sequence_for_classifier(all_kpts, seq_len=SEQ_LEN):
    if len(all_kpts) == 0:
        return np.zeros((seq_len, 17, 3), dtype=np.float32)

    if len(all_kpts) < seq_len:
        pad = [all_kpts[-1]] * (seq_len - len(all_kpts))
        all_kpts = list(all_kpts) + pad

    idxs = np.linspace(0, len(all_kpts) - 1, seq_len).astype(int)
    seq = [all_kpts[i] for i in idxs]
    return np.stack(seq, axis=0).astype(np.float32)