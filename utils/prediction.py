"""
Prediction engine for Facial Expression Recognition.
Handles CNN model loading, caching, inference, and result formatting.
"""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

import config

# Global cache for the loaded Keras model
_LOADED_MODEL: Optional[tf.keras.Model] = None


def is_model_available() -> bool:
    """Check if the trained CNN model file exists on disk."""
    return os.path.exists(config.MODEL_PATH)


def get_model() -> tf.keras.Model:
    """
    Retrieve the cached Keras model or load it from disk if not yet loaded.
    Raises FileNotFoundError if the trained model file has not been created yet.
    """
    global _LOADED_MODEL
    if _LOADED_MODEL is None:
        if not is_model_available():
            raise FileNotFoundError(
                f"Trained CNN model file not found at: {config.MODEL_PATH}\n"
                "To train the model, please execute: python training/train.py"
            )
        # Load compiled Keras model
        _LOADED_MODEL = tf.keras.models.load_model(str(config.MODEL_PATH))
    return _LOADED_MODEL


def predict_expression(
    preprocessed_tensor: np.ndarray,
    confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
    previous_probabilities: Optional[Dict[str, float]] = None,
    smooth_alpha: float = 0.65
) -> Dict[str, Any]:
    """
    Execute forward pass inference on a preprocessed face tensor.

    Args:
        preprocessed_tensor: NumPy array of shape (1, 48, 48, 1) normalized to [0, 1].
        confidence_threshold: Minimum confidence score to consider prediction certain.
        previous_probabilities: Optional previous frame probabilities for EMA smoothing.
        smooth_alpha: Weight factor for current frame in EMA (0.65 = 65% new, 35% history).

    Returns:
        Dictionary containing:
            - predicted_label: Most likely expression name.
            - confidence: Percentage score (0 - 100).
            - is_uncertain: Boolean indicating low-confidence prediction.
            - probabilities: Dict mapping each of the 7 classes to its percentage.
            - color: Hex color for the predicted expression.
            - description: Scientific description of visible facial features.
    """
    model = get_model()

    # Model inference
    raw_preds = model.predict(preprocessed_tensor, verbose=0)[0]

    # Convert to pure Python floats to avoid NumPy serialization issues
    current_probabilities = {
        name: float(raw_preds[i]) * 100.0
        for i, name in enumerate(config.CLASS_NAMES)
    }

    # Apply Exponential Moving Average (EMA) smoothing if history exists
    if previous_probabilities and all(k in previous_probabilities for k in config.CLASS_NAMES):
        smoothed_raw = {}
        for name in config.CLASS_NAMES:
            smoothed_raw[name] = (
                smooth_alpha * current_probabilities[name] +
                (1.0 - smooth_alpha) * float(previous_probabilities[name])
            )
        total = sum(smoothed_raw.values()) or 1.0
        probabilities = {k: round((v / total) * 100.0, 2) for k, v in smoothed_raw.items()}
    else:
        probabilities = {k: round(v, 2) for k, v in current_probabilities.items()}

    # Identify class with highest probability
    top_label = max(probabilities, key=probabilities.get)
    confidence = probabilities[top_label]
    is_uncertain = confidence < confidence_threshold

    return {
        "predicted_label": top_label,
        "confidence": confidence,
        "is_uncertain": is_uncertain,
        "confidence_threshold": confidence_threshold,
        "probabilities": probabilities,
        "color": config.CLASS_COLORS.get(top_label, "#64748b"),
        "description": config.CLASS_DESCRIPTIONS.get(top_label, ""),
        "uncertainty_note": (
            f"Prediction confidence ({confidence:.1f}%) is below the {confidence_threshold:.0f}% threshold. "
            "Visible cues show mixed or subtle expressions."
            if is_uncertain else ""
        )
    }


def get_model_summary_info() -> Dict[str, Any]:
    """Return status and metadata about the model environment."""
    available = is_model_available()
    info = {
        "is_available": available,
        "model_path": str(config.MODEL_PATH),
        "classes": config.CLASS_NAMES,
        "input_shape": config.INPUT_SHAPE,
        "framework": f"TensorFlow {tf.__version__}",
        "confidence_threshold": config.CONFIDENCE_THRESHOLD
    }

    if available:
        file_size_mb = os.path.getsize(config.MODEL_PATH) / (1024 * 1024)
        info["file_size_mb"] = round(file_size_mb, 2)
        try:
            model = get_model()
            info["total_params"] = model.count_params()
        except Exception:
            pass

    # Load offline evaluation metrics if available
    if config.EVALUATION_METRICS_PATH.exists():
        try:
            import json
            with open(config.EVALUATION_METRICS_PATH, "r") as f:
                info["evaluation_metrics"] = json.load(f)
        except Exception:
            info["evaluation_metrics"] = None
    else:
        info["evaluation_metrics"] = None

    return info
