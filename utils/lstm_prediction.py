"""
FaceSense AI – LSTM Sequence Prediction Engine
Handles loading, inference, and metadata management for the CNN + LSTM temporal model.

Features:
- Genuine LSTM sequence inference on (1, sequence_length, feature_dim) tensors
- Dynamic model caching in memory (loads once and reuses)
- Strict adherence to real-world performance: Never fabricates predictions if model is untrained
- Full compatibility with 7-class FER emotional categories
"""

import json
import logging
import os
import threading
import time
from typing import Dict, Optional, Tuple

import keras
import numpy as np

import config

logger = logging.getLogger(__name__)

# Global LSTM model cache
_LSTM_MODEL: Optional[keras.Model] = None
_LSTM_LOCK = threading.Lock()


def get_default_metadata() -> dict:
    """Return default metadata structure when LSTM model is pending training."""
    return {
        "model_name": "FaceSense_CNN_LSTM",
        "trained": False,
        "status": "not_trained",
        "reason": "Awaiting labeled facial video sequences (e.g. RAVDESS). Training pipeline ready in training/train_lstm.py.",
        "architecture": {
            "type": "CNN_Feature_Extractor_Plus_LSTM",
            "input_shape": [config.SEQUENCE_LENGTH, config.FEATURE_DIM],
            "sequence_length": config.SEQUENCE_LENGTH,
            "feature_dim": config.FEATURE_DIM,
            "intermediate_cnn_layer": config.FEATURE_LAYER_NAME,
            "lstm_units": 128,
            "dense_units": 64,
            "num_classes": config.NUM_CLASSES
        },
        "classes": config.CLASS_NAMES,
        "evaluation": None,
        "dataset_name": None,
        "instructions": (
            "To train the genuine temporal model: Download RAVDESS video clips into "
            "dataset/videos/ and execute 'python training/train_lstm.py'."
        )
    }


def load_lstm_metadata() -> dict:
    """Read metadata from model/lstm_metadata.json or return default structure."""
    meta_path = str(config.LSTM_METADATA_PATH)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Error reading LSTM metadata: %s", str(e))
    return get_default_metadata()


def save_lstm_metadata(metadata: dict) -> None:
    """Save training metadata to model/lstm_metadata.json."""
    meta_path = str(config.LSTM_METADATA_PATH)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    logger.info("Saved LSTM metadata to %s", meta_path)


def load_lstm_model(force_reload: bool = False) -> Optional[keras.Model]:
    """
    Load the trained CNN + LSTM model into memory.
    Thread-safe and cached.
    """
    global _LSTM_MODEL
    with _LSTM_LOCK:
        if _LSTM_MODEL is not None and not force_reload:
            return _LSTM_MODEL

        model_path = str(config.LSTM_MODEL_PATH)
        if not os.path.exists(model_path):
            _LSTM_MODEL = None
            return None

        try:
            logger.info("Loading CNN + LSTM model from %s...", model_path)
            model = keras.models.load_model(model_path, compile=False)
            
            # Warm-up inference with dummy sequence
            dummy_seq = np.zeros(
                (1, config.SEQUENCE_LENGTH, config.FEATURE_DIM),
                dtype=np.float32
            )
            _ = model(dummy_seq, training=False)
            _LSTM_MODEL = model
            logger.info("CNN + LSTM model loaded successfully.")
            return _LSTM_MODEL
        except Exception as e:
            logger.error("Failed to load LSTM model from %s: %s", model_path, str(e))
            _LSTM_MODEL = None
            return None


def is_lstm_available() -> bool:
    """Return True only if a trained LSTM model file exists on disk."""
    meta = load_lstm_metadata()
    if not meta.get("trained", False):
        return False
    return os.path.exists(str(config.LSTM_MODEL_PATH))


def predict_temporal_sequence(sequence_tensor: np.ndarray) -> dict:
    """
    Run genuine LSTM inference on a temporal sequence tensor.

    Args:
        sequence_tensor: NumPy array of shape (1, sequence_length, feature_dim).

    Returns:
        Prediction dictionary containing predicted_label, confidence, probabilities, and latency.
        If model is untrained, returns status='not_trained' without fabricating predictions.
    """
    t0 = time.time()
    model = load_lstm_model()
    metadata = load_lstm_metadata()

    # Honest handling: Never fabricate predictions if untrained
    if model is None or not metadata.get("trained", False):
        return {
            "status": "not_trained",
            "trained": False,
            "message": (
                "LSTM temporal model is not trained yet. The training pipeline is ready in "
                "training/train_lstm.py. Run it with a labeled video dataset (e.g. RAVDESS)."
            ),
            "predicted_label": None,
            "confidence": 0.0,
            "probabilities": {},
            "color": "#94a3b8",
            "description": "Model pending video sequence training.",
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "is_uncertain": True,
            "uncertainty_note": "Awaiting video training dataset."
        }

    # Validate tensor shape
    if not isinstance(sequence_tensor, np.ndarray):
        sequence_tensor = np.asarray(sequence_tensor, dtype=np.float32)

    if sequence_tensor.ndim == 2:
        sequence_tensor = np.expand_dims(sequence_tensor, axis=0)

    expected_shape = (1, config.SEQUENCE_LENGTH, config.FEATURE_DIM)
    if sequence_tensor.shape != expected_shape:
        raise ValueError(
            f"Expected sequence tensor of shape {expected_shape}, received {sequence_tensor.shape}"
        )

    # Real LSTM forward inference
    preds = model(sequence_tensor, training=False)
    if hasattr(preds, "numpy"):
        raw_probs = preds.numpy()[0]
    else:
        raw_probs = np.asarray(preds)[0]

    latency_ms = round((time.time() - t0) * 1000, 1)

    # Find dominant class
    pred_idx = int(np.argmax(raw_probs))
    pred_label = config.CLASS_MAPPING.get(pred_idx, "Unknown")
    confidence = round(float(raw_probs[pred_idx]) * 100, 2)

    # Class probabilities dictionary
    probs_dict = {
        config.CLASS_MAPPING[i]: round(float(raw_probs[i]) * 100, 2)
        for i in range(len(raw_probs))
    }

    is_uncertain = confidence < config.CONFIDENCE_THRESHOLD
    uncertainty_note = ""
    if is_uncertain:
        uncertainty_note = (
            f"Low temporal confidence ({confidence}% < {config.CONFIDENCE_THRESHOLD}% threshold). "
            "Temporal pattern exhibits transitional or mixed cues."
        )

    return {
        "status": "success",
        "trained": True,
        "predicted_label": pred_label,
        "confidence": confidence,
        "probabilities": probs_dict,
        "color": config.CLASS_COLORS.get(pred_label, "#10b981"),
        "description": config.CLASS_DESCRIPTIONS.get(pred_label, ""),
        "latency_ms": latency_ms,
        "is_uncertain": is_uncertain,
        "uncertainty_note": uncertainty_note
    }


def get_lstm_summary_info() -> dict:
    """Return comprehensive summary of the CNN + LSTM temporal system."""
    meta = load_lstm_metadata()
    available = is_lstm_available()
    return {
        "model_name": meta.get("model_name", "FaceSense_CNN_LSTM"),
        "trained": available,
        "status": "trained" if available else "not_trained",
        "reason": meta.get("reason", "Awaiting video sequences dataset (e.g. RAVDESS)."),
        "architecture": {
            "type": "CNN_Feature_Extractor_Plus_LSTM",
            "input_shape": [config.SEQUENCE_LENGTH, config.FEATURE_DIM],
            "sequence_length": config.SEQUENCE_LENGTH,
            "feature_dim": config.FEATURE_DIM,
            "intermediate_cnn_layer": config.FEATURE_LAYER_NAME,
            "lstm_units": 128,
            "dense_units": 64,
            "dropout_rate": 0.35,
            "num_classes": config.NUM_CLASSES,
        },
        "classes": config.CLASS_NAMES,
        "evaluation": meta.get("evaluation", None),
        "dataset_name": meta.get("dataset_name", None),
        "instructions": meta.get(
            "instructions",
            "Place video files in dataset/videos/ and run python training/train_lstm.py"
        )
    }


def get_comparison_specs() -> dict:
    """Return theoretical and architectural comparison matrix between CNN and CNN+LSTM."""
    lstm_info = get_lstm_summary_info()
    return {
        "modes": {
            "cnn": {
                "name": "Spatial CNN Mode",
                "input": "Single Static Frame (48x48x1 Grayscale)",
                "analysis_type": "Spatial Features (Local textures, edges, contours)",
                "prediction_scope": "Instantaneous Frame Expression",
                "temporal_context": "None (Each frame processed independently)",
                "inference_throughput": "1 Frame per inference cycle (~15-30ms)",
                "primary_use_case": "Fast snapshot classification & low-latency tracking",
                "training_dataset": "FER-2013 (35,887 static image faces)",
                "test_accuracy": "56.09% (FER-2013 held-out test set)",
                "status": "active"
            },
            "cnn_lstm": {
                "name": "Spatiotemporal CNN + LSTM Mode",
                "input": f"{config.SEQUENCE_LENGTH} Consecutive Frames (Sequence of 256-D CNN embeddings)",
                "analysis_type": "Spatiotemporal Dynamics (Micro-expression onset, apex, offset)",
                "prediction_scope": "Temporal Sequence Expression",
                "temporal_context": f"Sliding window of {config.SEQUENCE_LENGTH} consecutive frames",
                "inference_throughput": "Sliding sequence buffer inference (~25-45ms)",
                "primary_use_case": "Continuous video stream analysis, reducing single-frame flickers",
                "training_dataset": "RAVDESS Video Dataset / Video sequences",
                "test_accuracy": (
                    lstm_info.get("evaluation", {}).get("test_accuracy", "Pending Video Training")
                    if lstm_info.get("trained") else "Awaiting Video Training"
                ),
                "status": "trained" if lstm_info.get("trained") else "awaiting_training"
            }
        },
        "comparison_table": [
            {"dimension": "Input Representation", "cnn": "1x 48x48 Grayscale Face Crop", "lstm": f"{config.SEQUENCE_LENGTH}x 256-D CNN Feature Vectors"},
            {"dimension": "Feature Space", "cnn": "Spatial Convolutions (Conv2D, BatchNorm, MaxPool)", "lstm": "Recurrent Temporal Memory (128 LSTM units with tanh/sigmoid gates)"},
            {"dimension": "Temporal Awareness", "cnn": "None (Memoryless)", "lstm": f"Yes ({config.SEQUENCE_LENGTH}-frame temporal receptive field)"},
            {"dimension": "Sensitivity to Noise", "cnn": "Vulnerable to single-frame blinks or motion blur", "lstm": "Robust across temporal continuity"},
            {"dimension": "Prediction Latency", "cnn": "~15 - 25 ms", "lstm": "~25 - 45 ms"},
            {"dimension": "Operational Status", "cnn": "Trained & Active (56.09% accuracy)", "lstm": "Trained & Active" if lstm_info.get("trained") else "Pipeline Ready (Awaiting Video Dataset)"}
        ]
    }

