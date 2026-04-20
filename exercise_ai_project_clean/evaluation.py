# evaluate_performance.py
import os
import json
import time
import numpy as np
import pandas as pd
from analyze_video import analyze_video
from collections import Counter



test_videos = {
    "test_videos/curl_good_1.mov": {
        "true_exercise": "CURL",
        "true_reps": 6,  # CHANGE THIS - count manually
        "expected_form": "good",
        "has_elbow_swing": False,
        "has_rom_issue": False
    },
    "test_videos/curl_good_2.mov": {
        "true_exercise": "CURL",
        "true_reps": 6,
        "expected_form": "good",
        "has_elbow_swing": False,
        "has_rom_issue": False
    },
    "test_videos/curl_good_3.mov": {
        "true_exercise": "CURL",
        "true_reps": 6,
        "expected_form": "good",
        "has_elbow_swing": False,
        "has_rom_issue": False
    },
    
    "test_videos/curl_good_5.mov": {
        "true_exercise": "CURL",
        "true_reps": 5,
        "expected_form": "good",
        "has_elbow_swing": False,
        "has_rom_issue": False
    },
    "test_videos/elbows_swinging_11.mov": {
        "true_exercise": "CURL",
        "true_reps": 4,
        "expected_form": "bad",
        "has_elbow_swing": True,
        "has_rom_issue": False
    },
    "test_videos/elbows_swinging_14.mov": {
        "true_exercise": "CURL",
        "true_reps": 4,
        "expected_form": "bad",
        "has_elbow_swing": True,
        "has_rom_issue": False
    },
    "test_videos/not_full_rom_3.mov": {
        "true_exercise": "CURL",
        "true_reps": 5,
        "expected_form": "bad",
        "has_elbow_swing": False,
        "has_rom_issue": True
    },
    
    "test_videos/press_good_test_3.mov": {
        "true_exercise": "PRESS",
        "true_reps": 5,
        "expected_form": "good",
        "has_alignment_issue": False,
        "has_elbow_flare": False
    },
    "test_videos/press_good_test_4.mov": {
        "true_exercise": "PRESS",
        "true_reps": 4,
        "expected_form": "good",
        "has_alignment_issue": False,
        "has_elbow_flare": False
    },
    "test_videos/press_good_test_7.mov": {
        "true_exercise": "PRESS",
        "true_reps": 5,
        "expected_form": "good",
        "has_alignment_issue": False,
        "has_elbow_flare": False
    },
    "test_videos/press_good_test_10.mov": {
        "true_exercise": "PRESS",
        "true_reps": 10,
        "expected_form": "good",
        "has_alignment_issue": False,
        "has_elbow_flare": False
    },
    
    "test_videos/squat_test_1.mov": {
        "true_exercise": "SQUAT",
        "true_reps": 9,
        "expected_form": "bad",
        "has_depth_issue": False,
        "has_lean_issue": True
    },
    "test_videos/squat_test_2.mov": {
        "true_exercise": "SQUAT",
        "true_reps": 10,
        "expected_form": "bad",
        "has_depth_issue": True,
        "has_lean_issue": True
    },
    "test_videos/heels_moving_up_1.mov": {
        "true_exercise": "SQUAT",
        "true_reps": 7,
        "expected_form": "bad",
        "has_depth_issue": True,
        "has_lean_issue": False
    },
    "test_videos/heels_moving_up_2.mov": {
        "true_exercise": "SQUAT",
        "true_reps": 4,
        "expected_form": "bad",
        "has_depth_issue": True,
        "has_lean_issue": False
    },
}


def evaluate_classification():
    print("\n" + "="*60)
    print("CLASSIFICATION ACCURACY")
    
    print("="*60)
    
    correct = 0
    total = 0
    confusion = {"CURL": {"CURL": 0, "PRESS": 0, "SQUAT": 0},
                 "PRESS": {"CURL": 0, "PRESS": 0, "SQUAT": 0},
                 "SQUAT": {"CURL": 0, "PRESS": 0, "SQUAT": 0}}
    
    for video, truth in test_videos.items():
        if not os.path.exists(video):
            print(f"SKIP: {video} not found")
            continue
            
        result = analyze_video(video)
        pred_exercise = result["exercise"]
        true_exercise = truth["true_exercise"]
        
        total += 1
        if pred_exercise == true_exercise:
            correct += 1
        
        confusion[true_exercise][pred_exercise] += 1
        print(f"{video.split('/')[-1]}: TRUE={true_exercise}, PRED={pred_exercise} {'✓' if pred_exercise == true_exercise else '✗'}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {correct}/{total} = {accuracy*100:.1f}%")
    
    print("\nConfusion Matrix:")
    print("            Predicted")
    print("            Curl  Press Squat")
    for true in ["CURL", "PRESS", "SQUAT"]:
        print(f"Actual {true}:  {confusion[true]['CURL']:3d}   {confusion[true]['PRESS']:3d}    {confusion[true]['SQUAT']:3d}")
    
    return accuracy

def evaluate_rep_counting():
    print("REP COUNTING ACCURACY")
    
    errors = []
    results = []
    
    for video, truth in test_videos.items():
        if not os.path.exists(video):
            continue
            
        result = analyze_video(video)
        pred_reps = result["summary"]["reps"]
        true_reps = truth["true_reps"]
        error = abs(pred_reps - true_reps)
        errors.append(error)
        
        results.append({
            "video": video.split('/')[-1],
            "true_reps": true_reps,
            "pred_reps": pred_reps,
            "error": error
        })
        
        print(f"{video.split('/')[-1]}: true={true_reps}, pred={pred_reps}, error={error}")
    
    avg_error = np.mean(errors)
    within_1 = sum(1 for e in errors if e <= 1) / len(errors)
    
    print(f"\nAverage absolute error: {avg_error:.2f} reps")
    print(f"Within ±1 rep: {within_1*100:.1f}%")
    
    return avg_error, within_1

def evaluate_form_feedback():
    print("FORM FEEDBACK ACCURACY")
    
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for video, truth in test_videos.items():
        if not os.path.exists(video):
            continue
            
        result = analyze_video(video)
        feedback = result["feedback"]
        has_feedback = len(feedback) > 0 and "good" not in feedback[0].lower() and "looking" not in feedback[0].lower()
        
        if truth["expected_form"] == "bad":
            if has_feedback:
                true_positives += 1
                print(f"{video.split('/')[-1]}: BAD form → {feedback[0][:30]}... ✓")
            else:
                false_negatives += 1
                print(f"{video.split('/')[-1]}: BAD form → NO FEEDBACK ✗")
        else:  # good form
            if has_feedback:
                false_positives += 1
                print(f"{video.split('/')[-1]}: GOOD form → {feedback[0][:30]}... ✗ (false positive)")
            else:
                true_negatives += 1
                print(f"{video.split('/')[-1]}: GOOD form → {feedback[0]} ✓")
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nTrue Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Negatives: {false_negatives}")
    print(f"\nPrecision: {precision*100:.1f}%")
    print(f"Recall: {recall*100:.1f}%")
    print(f"F1-Score: {f1*100:.1f}%")
    
    return precision, recall, f1


if __name__ == "__main__":
    print("="*60)
    print("SYSTEM PERFORMANCE EVALUATION")
    print("="*60)
    
    print("\nChecking test videos...")
    for video in test_videos.keys():
        if os.path.exists(video):
            print(f"✓ {video}")
        else:
            print(f"✗ {video} (MISSING)")
    
    class_acc = evaluate_classification()
    rep_error, rep_within_1 = evaluate_rep_counting()
    form_precision, form_recall, form_f1 = evaluate_form_feedback()
    
    
    print("\n" + "="*60)
    print("FINAL SUMMARY FOR DISSERTATION")
    print("="*60)
    print(f"""
| Metric | Value |
|--------|-------|
| Classification Accuracy | {class_acc*100:.1f}% |
| Rep Counting Error (MAE) | {rep_error:.2f} reps |
| Rep Counting (±1 rep) | {rep_within_1*100:.1f}% |
| Form Feedback Precision | {form_precision*100:.1f}% |
| Form Feedback Recall | {form_recall*100:.1f}% |
| Form Feedback F1-Score | {form_f1*100:.1f}% |
""")