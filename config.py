"""
Configuration settings for Human Facial Expression Recognition Project.
BCA Semester 5 - Neural Networks Mini Project.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATASET_DIR = BASE_DIR / "dataset"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
PLOTS_DIR = STATIC_DIR / "plots"

# File paths
MODEL_PATH = MODEL_DIR / "fer_cnn_model.keras"
HAARCASCADE_PATH = MODEL_DIR / "haarcascade_frontalface_default.xml"
SAMPLE_DATASET_PATH = DATASET_DIR / "fer2013_sample.csv"
FULL_DATASET_PATH = DATASET_DIR / "fer2013.csv"
EVALUATION_METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"

# CNN + LSTM Temporal Sequence Paths & Parameters
LSTM_MODEL_PATH = MODEL_DIR / "fer_cnn_lstm_model.keras"
LSTM_METADATA_PATH = MODEL_DIR / "lstm_metadata.json"
VIDEOS_DATASET_DIR = DATASET_DIR / "videos"
SEQUENCES_CACHE_DIR = DATASET_DIR / "sequences"
SEQUENCE_LENGTH = 16          # Default frames in temporal sliding window
FEATURE_DIM = 256             # Intermediate CNN spatial embedding dimension
FEATURE_LAYER_NAME = "bn_dense"  # Layer in CNN producing the 256-dim embedding

# Project Branding
PROJECT_NAME = "FaceSense AI"
PROJECT_SUBTITLE = "Real-Time Facial Expression Analysis Using CNN & LSTM"

# Flask Configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "facesense-ai-secret-key-2026")
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 Megabytes limit
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
CONFIDENCE_THRESHOLD = 35.0  # Minimum confidence % to consider a prediction certain

# Neural Network Input Specifications
IMAGE_WIDTH = 48
IMAGE_HEIGHT = 48
IMAGE_CHANNELS = 1  # Grayscale for FER-2013
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
NUM_CLASSES = 7

# Seven Standard FER-2013 Facial Expression Categories
CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# Mapping indices to class names
CLASS_MAPPING = {i: name for i, name in enumerate(CLASS_NAMES)}

# Color palette for each expression for UI charts & badges
CLASS_COLORS = {
    "Angry": "#ef4444",      # Vibrant Red
    "Disgust": "#84cc16",    # Olive Green
    "Fear": "#a855f7",       # Purple
    "Happy": "#10b981",      # Emerald Green
    "Sad": "#3b82f6",        # Royal Blue
    "Surprise": "#f59e0b",   # Amber / Orange
    "Neutral": "#64748b"     # Slate Gray
}

# Facial Expression Action Unit / Feature Descriptions for Viva & Presentations
CLASS_DESCRIPTIONS = {
    "Angry": "Lowered and drawn-together eyebrows, tightened eyelids, and pressed lips.",
    "Disgust": "Wrinkled nose bridge, raised upper lip, and squinted eyes.",
    "Fear": "Raised and pulled-together eyebrows, widened eyes, and parted lips.",
    "Happy": "Corners of the lips drawn upwards (zygomaticus major), crinkled outer eye corners (crow's feet).",
    "Sad": "Inner corners of eyebrows raised, drooping eyelid corners, and pulled-down lip corners.",
    "Surprise": "Arched eyebrows, widened eyelids, and dropped open jaw.",
    "Neutral": "Relaxed facial muscles without prominent furrowing, elevation, or stretching."
}

# Ethical and Scientific Disclaimer
ETHICAL_DISCLAIMER = (
    "Scientific Note: This machine learning model identifies visible surface facial muscle patterns "
    "categorized in the FER-2013 benchmark. In accordance with modern cognitive science, surface facial "
    "expressions do not directly determine or guarantee an individual's true internal psychological state "
    "or feelings."
)
