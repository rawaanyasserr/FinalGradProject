import numpy as np
import pandas as pd
from scipy.signal import find_peaks

L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


def angle(a, b, c):
    a = np.array(a[:2], dtype=np.float32)
    b = np.array(b[:2], dtype=np.float32)
    c = np.array(c[:2], dtype=np.float32)

    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cosang = np.dot(ba, bc) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def smooth_signal(x, window=5):
    if len(x) < window:
        return np.array(x, dtype=np.float32)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def count_reps_peak_based(signal, min_distance=8, min_prominence=0.1):
    if len(signal) < 10:
        return 0, []

    sig_norm = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-6)

    peaks, _ = find_peaks(
        sig_norm,
        distance=min_distance,
        prominence=min_prominence,
        height=0.3
    )

    valleys, _ = find_peaks(
        -sig_norm,
        distance=min_distance,
        prominence=min_prominence,
        height=0.3
    )

    if len(peaks) == 0 and len(valleys) == 0:
        return 0, []

    rep_count = max(len(peaks), len(valleys))

    if abs(len(peaks) - len(valleys)) <= 1:
        rep_count = (len(peaks) + len(valleys)) // 2

    return max(0, rep_count), peaks.tolist()


def count_curl_reps(all_kpts):
    left_angles = []
    right_angles = []

    for k in all_kpts:
        if k[L_ELBOW, 2] > 0.3 and k[L_WRIST, 2] > 0.3:
            left_angles.append(angle(k[L_SHOULDER], k[L_ELBOW], k[L_WRIST]))
        else:
            left_angles.append(np.nan)

        if k[R_ELBOW, 2] > 0.3 and k[R_WRIST, 2] > 0.3:
            right_angles.append(angle(k[R_SHOULDER], k[R_ELBOW], k[R_WRIST]))
        else:
            right_angles.append(np.nan)

    angles = []
    for l, r in zip(left_angles, right_angles):
        if not np.isnan(l) and not np.isnan(r):
            angles.append((l + r) / 2)
        elif not np.isnan(l):
            angles.append(l)
        elif not np.isnan(r):
            angles.append(r)
        else:
            angles.append(np.nan)

    angles = pd.Series(angles).interpolate().bfill().ffill().values
    signal = smooth_signal(angles, window=5)
    reps, _ = count_reps_peak_based(signal, min_distance=8, min_prominence=0.15)

    return reps, signal.tolist()


def count_press_reps(all_kpts):
    wrist_y = []

    for k in all_kpts:
        if k[L_WRIST, 2] > 0.3 and k[R_WRIST, 2] > 0.3:
            wy = (k[L_WRIST, 1] + k[R_WRIST, 1]) / 2.0
        elif k[L_WRIST, 2] > 0.3:
            wy = k[L_WRIST, 1]
        elif k[R_WRIST, 2] > 0.3:
            wy = k[R_WRIST, 1]
        else:
            wy = np.nan
        wrist_y.append(wy)

    wrist_y = pd.Series(wrist_y).interpolate().bfill().ffill().values
    signal = smooth_signal(wrist_y, window=5)
    reps, _ = count_reps_peak_based(signal, min_distance=8, min_prominence=0.05)

    return reps, signal.tolist()


def count_squat_reps(all_kpts):
    knee_angles = []

    for k in all_kpts:
        left_conf = min(k[L_HIP, 2], k[L_KNEE, 2], k[L_ANKLE, 2])
        right_conf = min(k[R_HIP, 2], k[R_KNEE, 2], k[R_ANKLE, 2])

        if left_conf > 0.3 and right_conf > 0.3:
            lk = angle(k[L_HIP], k[L_KNEE], k[L_ANKLE])
            rk = angle(k[R_HIP], k[R_KNEE], k[R_ANKLE])
            knee_angles.append((lk + rk) / 2)
        elif left_conf > 0.3:
            knee_angles.append(angle(k[L_HIP], k[L_KNEE], k[L_ANKLE]))
        elif right_conf > 0.3:
            knee_angles.append(angle(k[R_HIP], k[R_KNEE], k[R_ANKLE]))
        else:
            knee_angles.append(np.nan)

    knee_angles = pd.Series(knee_angles).interpolate().bfill().ffill().values
    signal = smooth_signal(knee_angles, window=7)
    signal_inv = -signal
    reps, _ = count_reps_peak_based(signal_inv, min_distance=10, min_prominence=0.1)

    return reps, signal.tolist()