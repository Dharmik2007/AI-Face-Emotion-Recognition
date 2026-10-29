"""
FaceSense AI – Video Dataset Preprocessing & Temporal Sequence Builder
Processes video files (e.g., RAVDESS), extracts facial crops, computes CNN spatial
feature vectors, and constructs temporal sequences for LSTM training.

Subject-Independent Partitioning:
Ensures frames/sequences from the same human actor do not appear in both
training and testing sets, preventing identity leakage.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from utils.feature_extractor import get_feature_extractor
from utils.image_processing import crop_and_preprocess_face, detect_faces

# RAVDESS Emotion Code to FER-2013 Category Mapping
# RAVDESS: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
RAVDESS_EMOTION_MAP = {
    "01": "Neutral",
    "02": "Neutral",  # Calm mapped to Neutral
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fear",
    "07": "Disgust",
    "08": "Surprise"
}

# Reverse mapping to FER-2013 7-class integer indices
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(config.CLASS_NAMES)}


def parse_video_metadata(video_path: Path) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Parse emotion label and subject/actor ID from video filename or directory structure.

    Returns:
        (emotion_name, emotion_index, actor_id_str)
    """
    stem = video_path.stem
    parent_name = video_path.parent.name

    # Check for official RAVDESS naming: Modality-Channel-Emotion-Intensity-Statement-Repetition-Actor.mp4
    # e.g., 02-01-06-01-02-01-01.mp4
    tokens = stem.split("-")
    if len(tokens) >= 7:
        emotion_code = tokens[2]
        actor_id = tokens[6]
        emotion_name = RAVDESS_EMOTION_MAP.get(emotion_code)
        if emotion_name and emotion_name in CLASS_TO_INDEX:
            return emotion_name, CLASS_TO_INDEX[emotion_name], f"actor_{actor_id}"

    # Fallback to directory name matching class names (e.g., dataset/videos/Happy/clip1.mp4)
    if parent_name in CLASS_TO_INDEX:
        actor_match = re.search(r"actor[_-]?(\d+)", stem, re.IGNORECASE)
        actor_id = actor_match.group(1) if actor_match else parent_name
        return parent_name, CLASS_TO_INDEX[parent_name], f"subject_{actor_id}"

    return None, None, None


def extract_features_from_video(
    video_path: Path,
    feature_extractor,
    sample_stride: int = 2
) -> List[np.ndarray]:
    """
    Read frames from a video clip, detect and crop faces, and extract 256-D feature vectors.

    Args:
        video_path: Path to .mp4 or .avi file.
        feature_extractor: Initialized CNNFeatureExtractor instance.
        sample_stride: Read every N-th frame to match live webcam temporal cadence (~15 fps).

    Returns:
        List of 1D feature vectors (each of shape (256,)).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    features_list = []
    frame_idx = 0
    last_known_bbox = None

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % sample_stride != 0:
            continue

        # Detect face
        faces = detect_faces(frame_bgr, min_neighbors=5)
        primary_bbox = faces[0] if faces else last_known_bbox
        if faces:
            last_known_bbox = faces[0]

        # Preprocess and normalize face crop
        tensor, _ = crop_and_preprocess_face(
            image_bgr=frame_bgr,
            bbox=primary_bbox,
            target_size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
        )

        # Extract 256-D spatial embedding
        try:
            feat_vec = feature_extractor.extract_single_vector(tensor)
            features_list.append(feat_vec)
        except Exception:
            continue

    cap.release()
    return features_list


def build_sequences_from_features(
    features_list: List[np.ndarray],
    sequence_length: int = 16,
    stride: int = 4
) -> List[np.ndarray]:
    """
    Slice consecutive feature vectors into overlapping sliding-window sequences.

    Returns:
        List of NumPy arrays, each of shape (sequence_length, feature_dim).
    """
    sequences = []
    total_frames = len(features_list)
    if total_frames < sequence_length:
        return []

    for start in range(0, total_frames - sequence_length + 1, stride):
        window = features_list[start : start + sequence_length]
        sequences.append(np.array(window, dtype=np.float32))

    return sequences


def prepare_video_dataset(
    video_dir: Optional[Path] = None,
    sequence_length: int = 16,
    stride: int = 4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full pipeline to build subject-independent train, validation, and test datasets
    from video clips in dataset/videos/.

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    video_dir = Path(video_dir or config.VIDEOS_DATASET_DIR)
    if not video_dir.exists():
        raise FileNotFoundError(f"Video dataset directory not found at {video_dir}")

    video_files = list(video_dir.glob("**/*.mp4")) + list(video_dir.glob("**/*.avi"))
    if not video_files:
        raise ValueError(f"No video files (.mp4, .avi) found in {video_dir}")

    feature_extractor = get_feature_extractor()
    if not feature_extractor.is_available():
        raise RuntimeError("CNN feature extractor is not ready. Train the CNN model first.")

    # Group video clips by actor / subject to guarantee subject-independent splits
    subject_sequences: Dict[str, List[Tuple[np.ndarray, int]]] = {}

    print(f"[*] Processing {len(video_files)} video clips from {video_dir}...")
    for idx, v_file in enumerate(video_files, 1):
        emotion_name, emotion_idx, actor_id = parse_video_metadata(v_file)
        if emotion_name is None:
            continue

        feats = extract_features_from_video(v_file, feature_extractor)
        seqs = build_sequences_from_features(feats, sequence_length=sequence_length, stride=stride)

        if not seqs:
            continue

        if actor_id not in subject_sequences:
            subject_sequences[actor_id] = []

        for s in seqs:
            subject_sequences[actor_id].append((s, emotion_idx))

        if idx % 10 == 0 or idx == len(video_files):
            print(f"    - Processed {idx}/{len(video_files)} videos...")

    # Partition subjects into Train (70%), Val (15%), Test (15%)
    all_subjects = sorted(list(subject_sequences.keys()))
    if len(all_subjects) >= 3:
        n_test = max(1, int(len(all_subjects) * 0.15))
        n_val = max(1, int(len(all_subjects) * 0.15))
        test_subjects = set(all_subjects[-n_test:])
        val_subjects = set(all_subjects[-(n_test + n_val) : -n_test])
        train_subjects = set(all_subjects[: -(n_test + n_val)])
    else:
        # If very few subjects, allocate all to train/val
        train_subjects = set(all_subjects)
        val_subjects = set(all_subjects)
        test_subjects = set(all_subjects)

    train_data, val_data, test_data = [], [], []

    for sub, items in subject_sequences.items():
        if sub in test_subjects:
            test_data.extend(items)
        elif sub in val_subjects:
            val_data.extend(items)
        else:
            train_data.extend(items)

    def to_arrays(dataset_items):
        if not dataset_items:
            return np.empty((0, sequence_length, config.FEATURE_DIM)), np.empty((0, config.NUM_CLASSES))
        X = np.array([item[0] for item in dataset_items], dtype=np.float32)
        y_indices = np.array([item[1] for item in dataset_items], dtype=np.int32)
        # One-hot encoding for categorical crossentropy
        y_onehot = np.zeros((len(y_indices), config.NUM_CLASSES), dtype=np.float32)
        y_onehot[np.arange(len(y_indices)), y_indices] = 1.0
        return X, y_onehot

    X_train, y_train = to_arrays(train_data)
    X_val, y_val = to_arrays(val_data)
    X_test, y_test = to_arrays(test_data)

    print(f"[✓] Sequence Dataset Built:")
    print(f"    - Train sequences: {len(X_train)} (Shape: {X_train.shape})")
    print(f"    - Val sequences:   {len(X_val)} (Shape: {X_val.shape})")
    print(f"    - Test sequences:  {len(X_test)} (Shape: {X_test.shape})")

    return X_train, y_train, X_val, y_val, X_test, y_test
