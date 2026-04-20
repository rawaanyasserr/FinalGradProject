import numpy as np


def curl_form(kpts):
    feedback = []

    l_shoulder = kpts[5, :2]
    r_shoulder = kpts[6, :2]
    l_elbow = kpts[7, :2]
    r_elbow = kpts[8, :2]
    l_wrist = kpts[9, :2]
    r_wrist = kpts[10, :2]

    if kpts[5, 2] < 0.4 or kpts[6, 2] < 0.4:
        return []

    shoulder_width = abs(r_shoulder[0] - l_shoulder[0])
    if shoulder_width < 0.01:
        shoulder_width = 0.2

    l_elbow_offset = abs(l_elbow[0] - l_shoulder[0])
    r_elbow_offset = abs(r_elbow[0] - r_shoulder[0])

    l_swinging = l_elbow[0] > l_shoulder[0] + 0.05
    r_swinging = r_elbow[0] < r_shoulder[0] - 0.05

    if l_swinging and l_elbow_offset > 0.08:
        feedback.append("Left elbow swinging forward - keep it at your side")
    elif r_swinging and r_elbow_offset > 0.08:
        feedback.append("Right elbow swinging forward - keep it at your side")

    if feedback:
        return feedback

    l_wrist_below = l_wrist[1] > l_shoulder[1]
    r_wrist_below = r_wrist[1] > r_shoulder[1]

    l_rom = l_shoulder[1] - l_wrist[1]
    r_rom = r_shoulder[1] - r_wrist[1]

    if l_wrist_below and r_wrist_below:
        if l_rom < -0.05 and r_rom < -0.05:
            feedback.append("Curl all the way up - full range of motion")

    return feedback


def press_form(kpts):
    feedback = []

    l_wrist = kpts[9, :2]
    r_wrist = kpts[10, :2]
    l_shoulder = kpts[5, :2]
    r_shoulder = kpts[6, :2]
    l_elbow = kpts[7, :2]
    r_elbow = kpts[8, :2]

    if kpts[5, 2] < 0.4 or kpts[6, 2] < 0.4:
        return []

    shoulder_width = abs(r_shoulder[0] - l_shoulder[0])
    if shoulder_width < 0.01:
        shoulder_width = 0.2

    l_elbow_flare = abs(l_elbow[0] - l_shoulder[0]) > 0.10
    r_elbow_flare = abs(r_elbow[0] - r_shoulder[0]) > 0.10

    if l_elbow_flare:
        feedback.append("Tuck left elbow in - don't flare out")
    elif r_elbow_flare:
        feedback.append("Tuck right elbow in - don't flare out")

    if feedback:
        return feedback

    l_wrist_offset = abs(l_wrist[0] - l_shoulder[0])
    r_wrist_offset = abs(r_wrist[0] - r_shoulder[0])

    if l_wrist_offset > 0.12:
        feedback.append("Keep left wrist directly above shoulder")
    elif r_wrist_offset > 0.12:
        feedback.append("Keep right wrist directly above shoulder")

    return feedback


def squat_form(kpts):
    feedback = []

    def angle_at(b, a, c):
        ba = a - b
        bc = c - b
        denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
        cos_angle = np.dot(ba, bc) / denom
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    l_hip = kpts[11, :2]
    r_hip = kpts[12, :2]
    l_knee = kpts[13, :2]
    r_knee = kpts[14, :2]
    l_ankle = kpts[15, :2]
    r_ankle = kpts[16, :2]
    l_shoulder = kpts[5, :2]

    if kpts[11, 2] < 0.4 or kpts[12, 2] < 0.4:
        return []

    hip_center = (l_hip + r_hip) / 2
    shoulder_center = (l_shoulder + kpts[6, :2]) / 2
    knee_center = (l_knee + r_knee) / 2

    back_angle = angle_at(hip_center, shoulder_center, knee_center)

    if back_angle < 140:
        feedback.append("Keep your chest up - don't lean forward")
        return feedback

    l_angle = angle_at(l_knee, l_hip, l_ankle)
    r_angle = angle_at(r_knee, r_hip, r_ankle)
    min_angle = min(l_angle, r_angle)

    if min_angle > 115:
        feedback.append("Squat deeper - aim for thighs parallel to ground")

    return feedback