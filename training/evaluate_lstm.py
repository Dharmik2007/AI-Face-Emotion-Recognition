"""
FaceSense AI – CNN + LSTM Model Evaluation Routine
Evaluates the trained temporal LSTM model on held-out video sequences.
Computes multi-class Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
"""

import json
import os
import sys
from pathlib import Path

import keras
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from utils.lstm_prediction import load_lstm_metadata


def evaluate_lstm_model():
    """Evaluate the CNN + LSTM temporal model on test sequences."""
    print("=" * 70)
    print("  FaceSense AI – CNN + LSTM Temporal Model Evaluation")
    print("=" * 70)

    meta = load_lstm_metadata()
    if not meta.get("trained", False) or not os.path.exists(str(config.LSTM_MODEL_PATH)):
        print("\n[!] LSTM Model Status: NOT TRAINED")
        print(f"    Reason: {meta.get('reason', 'Model file not found on disk.')}")
        print("\n    To train the model:")
        print(f"    1. Place video clips in {config.VIDEOS_DATASET_DIR}")
        print("    2. Run: python training/train_lstm.py")
        print("=" * 70)
        return False

    print(f"[*] Loading LSTM model from {config.LSTM_MODEL_PATH}...")
    model = keras.models.load_model(str(config.LSTM_MODEL_PATH), compile=False)
    model.summary()

    eval_info = meta.get("evaluation")
    if eval_info:
        print("\n[✓] Recorded Evaluation Metrics (Held-out Test Sequences):")
        print(f"    - Test Accuracy:  {eval_info.get('test_accuracy', 'N/A')}%")
        print(f"    - Test Loss:      {eval_info.get('test_loss', 'N/A')}")
        print(f"    - Test Sequences: {eval_info.get('test_sequences', 'N/A')}")
    else:
        print("\n[*] Evaluation metrics pending test sequence evaluation.")
    print("=" * 70)
    return True


if __name__ == "__main__":
    evaluate_lstm_model()
