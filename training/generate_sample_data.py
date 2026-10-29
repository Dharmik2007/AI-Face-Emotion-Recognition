"""
Sample Data Generator for Verification & Quick Demonstration.
Creates a valid, structured mini FER-2013 CSV dataset (dataset/fer2013_sample.csv)
with realistic synthetic facial expression action units for all 7 classes.

Enables BCA students and examiners to test training, evaluation, and the web app
immediately on any PC without waiting for the large 300MB Kaggle download.
"""

import math
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure project root is in sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd

import config


def create_synthetic_face(emotion_id: int, seed: int = 0) -> np.ndarray:
    """
    Generate a 48x48 synthetic face image with feature contours
    corresponding to the specified FER-2013 emotion class.

    0: Angry
    1: Disgust
    2: Fear
    3: Happy
    4: Sad
    5: Surprise
    6: Neutral
    """
    np.random.seed(seed)
    random.seed(seed)

    # Base head: subtle circular/oval gradient
    img = np.full((48, 48), 160, dtype=np.uint8)

    # Add background vignette / head oval
    center = (24, 24)
    axes = (18 + random.randint(-1, 1), 21 + random.randint(-1, 1))
    mask = np.zeros((48, 48), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    # Blend background
    bg = np.random.randint(40, 70, (48, 48), dtype=np.uint8)
    img = np.where(mask == 255, img, bg)

    # Eyes positions
    left_eye = (17 + random.randint(-1, 1), 18)
    right_eye = (31 + random.randint(-1, 1), 18)

    # Draw eyes
    eye_radius = 2
    if emotion_id in [2, 5]:  # Fear or Surprise: wide open eyes
        eye_radius = 3
    cv2.circle(img, left_eye, eye_radius, 30, -1)
    cv2.circle(img, right_eye, eye_radius, 30, -1)

    # Eyebrows & Mouth variations per emotion
    if emotion_id == 0:  # Angry: V-slanted eyebrows, tight pressed mouth
        cv2.line(img, (13, 14), (21, 17), 20, 2)
        cv2.line(img, (27, 17), (35, 14), 20, 2)
        # Tight mouth
        cv2.line(img, (18, 34), (30, 34), 30, 2)

    elif emotion_id == 1:  # Disgust: wrinkled nose lines, raised upper lip
        cv2.line(img, (14, 15), (20, 16), 25, 2)
        cv2.line(img, (28, 16), (34, 15), 25, 2)
        # Nose wrinkle
        cv2.line(img, (23, 24), (25, 24), 40, 1)
        cv2.line(img, (24, 23), (24, 25), 40, 1)
        # Asymmetric mouth
        pts = np.array([[18, 35], [24, 32], [30, 36]], np.int32)
        cv2.polylines(img, [pts], False, 30, 2)

    elif emotion_id == 2:  # Fear: raised flat eyebrows, parted mouth
        cv2.line(img, (13, 13), (21, 14), 20, 2)
        cv2.line(img, (27, 14), (35, 13), 20, 2)
        # O-parted mouth
        cv2.ellipse(img, (24, 34), (5, 3), 0, 0, 360, 30, -1)

    elif emotion_id == 3:  # Happy: relaxed eyebrows, wide upward curve smile
        cv2.line(img, (14, 14), (20, 14), 25, 1)
        cv2.line(img, (28, 14), (34, 14), 25, 1)
        # Upward curved smile
        cv2.ellipse(img, (24, 32), (8, 5), 0, 0, 180, 20, 2)

    elif emotion_id == 4:  # Sad: inverted V eyebrows (/ \), downward curved mouth
        cv2.line(img, (13, 16), (20, 13), 20, 2)
        cv2.line(img, (28, 13), (35, 16), 20, 2)
        # Downward curved mouth
        cv2.ellipse(img, (24, 37), (7, 4), 0, 180, 360, 25, 2)

    elif emotion_id == 5:  # Surprise: high arched eyebrows, open round mouth
        cv2.ellipse(img, (17, 13), (4, 3), 0, 180, 360, 20, 2)
        cv2.ellipse(img, (31, 13), (4, 3), 0, 180, 360, 20, 2)
        # Large round mouth
        cv2.circle(img, (24, 34), 4, 20, -1)

    else:  # Neutral (6): straight horizontal eyebrows and flat mouth
        cv2.line(img, (14, 15), (20, 15), 30, 1)
        cv2.line(img, (28, 15), (34, 15), 30, 1)
        cv2.line(img, (18, 34), (30, 34), 30, 1)

    # Add Gaussian noise and blur for photographic realistic variation
    noise = np.random.normal(0, 4, img.shape).astype(np.float32)
    img_noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img_noisy, (3, 3), 0.5)

    return blurred


def generate_sample_fer_dataset(
    output_path: str = str(config.SAMPLE_DATASET_PATH),
    samples_per_class: int = 100
) -> str:
    """
    Generate fer2013_sample.csv containing samples_per_class * 7 images,
    split into Training (70%), PublicTest (15%), and PrivateTest (15%).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    records = []
    total_samples = samples_per_class * config.NUM_CLASSES
    print(f"Generating synthetic benchmark dataset ({total_samples} samples across 7 classes)...")

    for emotion_id in range(config.NUM_CLASSES):
        for s in range(samples_per_class):
            seed = emotion_id * 1000 + s
            face_img = create_synthetic_face(emotion_id, seed=seed)
            pixels_str = " ".join(face_img.flatten().astype(str))

            # Split logic: 70% Training, 15% PublicTest, 15% PrivateTest
            ratio = s / float(samples_per_class)
            if ratio < 0.70:
                usage = "Training"
            elif ratio < 0.85:
                usage = "PublicTest"
            else:
                usage = "PrivateTest"

            records.append({
                "emotion": emotion_id,
                "pixels": pixels_str,
                "Usage": usage
            })

    # Shuffle records
    random.seed(42)
    random.shuffle(records)

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Sample dataset saved to: {output_path} ({len(df)} rows)")
    return output_path


if __name__ == "__main__":
    generate_sample_fer_dataset()
