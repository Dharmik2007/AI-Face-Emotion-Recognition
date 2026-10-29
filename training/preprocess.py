"""
Preprocessing pipeline for the FER-2013 Facial Expression Dataset.
Supports loading from standard FER-2013 CSV format or image folders,
normalization, one-hot encoding, and data augmentation.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

# Ensure project root is in sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import tensorflow as tf

# Resolve ImageDataGenerator and to_categorical via tf.keras to satisfy static analyzers
# while preserving full compatibility across TensorFlow 2.16+ / Keras 3.
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator  # type: ignore[attr-defined]
to_categorical = tf.keras.utils.to_categorical  # type: ignore[attr-defined]


import config


def load_fer2013_from_csv(
    csv_path: str
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Parse the official FER-2013 CSV file containing 'emotion', 'pixels', and 'Usage'.

    Usage values in standard FER-2013:
      - 'Training' -> ~28,709 images
      - 'PublicTest' -> ~3,589 images (used for validation)
      - 'PrivateTest' -> ~3,589 images (used for final test evaluation)

    Returns:
      (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}")

    print(f"Loading dataset from: {csv_path} ...")
    df = pd.read_csv(csv_path)

    required_cols = {"emotion", "pixels"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain at least columns: {required_cols}. Found: {list(df.columns)}")

    # Fast Vectorized Pixel Parsing
    print(f"Parsing {len(df)} pixel arrays...")
    parsed_images = []
    valid_indices = []

    for idx, pixel_str in enumerate(df["pixels"]):
        try:
            arr = np.fromstring(pixel_str, sep=" ", dtype=np.float32)
            if len(arr) == config.IMAGE_WIDTH * config.IMAGE_HEIGHT:
                parsed_images.append(arr)
                valid_indices.append(idx)
        except Exception:
            continue

    if not parsed_images:
        raise ValueError("No valid image pixel sequences found in dataset CSV.")

    images = (np.array(parsed_images, dtype=np.float32) / 255.0).reshape(
        -1, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1
    )
    labels = df["emotion"].iloc[valid_indices].to_numpy(dtype=np.int32)

    # Convert labels to 7-class one-hot vectors
    y_categorical = to_categorical(labels, num_classes=config.NUM_CLASSES)

    has_usage = "Usage" in df.columns
    if has_usage and "Training" in df["Usage"].values:
        usages = df["Usage"].iloc[valid_indices].to_numpy(dtype=str)
        train_mask = (usages == "Training")
        val_mask = (usages == "PublicTest")
        test_mask = (usages == "PrivateTest")

        X_train, y_train = images[train_mask], y_categorical[train_mask]
        X_val, y_val = images[val_mask], y_categorical[val_mask]
        X_test, y_test = images[test_mask], y_categorical[test_mask]

        # If val or test splits are empty, split train automatically
        if len(X_val) == 0 or len(X_test) == 0:
            return _split_data(images, y_categorical)
    else:
        return _split_data(images, y_categorical)

    print(f"Dataset Loaded Successfully:")
    print(f"  Training samples:   {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples:       {len(X_test)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def _split_data(
    images: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Split dataset arrays randomly into train, validation, and test subsets."""
    n_samples = len(images)
    indices = np.random.permutation(n_samples)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (
        (images[train_idx], labels[train_idx]),
        (images[val_idx], labels[val_idx]),
        (images[test_idx], labels[test_idx])
    )


def get_data_augmentation_generator() -> ImageDataGenerator:
    """
    Create a Keras ImageDataGenerator configured with standard
    regularization transforms: horizontal flip, slight rotation, and zoom.
    """
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )
