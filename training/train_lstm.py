"""
FaceSense AI – CNN + LSTM Neural Network Training Pipeline
Trains a sequence-based Long Short-Term Memory (LSTM) network on consecutive CNN
spatial embeddings to capture temporal facial dynamics (micro-expressions, transitions).

Architecture:
Input (16, 256) -> Masking -> LSTM(128) -> Dense(64, ReLU) -> BatchNorm -> Dropout(0.35) -> Dense(7, Softmax)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import keras
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from training.video_preprocessing import prepare_video_dataset
from utils.lstm_prediction import load_lstm_metadata, save_lstm_metadata


def build_cnn_lstm_model(
    sequence_length: int = 16,
    feature_dim: int = 256,
    num_classes: int = 7
) -> keras.Model:
    """
    Construct the temporal sequence classifier in Keras 3.

    Args:
        sequence_length: Number of consecutive video frames in the sliding window.
        feature_dim: Dimension of intermediate CNN spatial features (256).
        num_classes: Number of expression classes (7).

    Returns:
        Compiled or uncompiled Keras Model.
    """
    inputs = keras.Input(
        shape=(sequence_length, feature_dim),
        name="input_temporal_sequence"
    )

    # Mask zero-padded frames if variable sequence length
    x = keras.layers.Masking(mask_value=0.0, name="sequence_masking")(inputs)

    # Core Temporal Recurrent Layer
    x = keras.layers.LSTM(
        128,
        return_sequences=False,
        dropout=0.25,
        name="temporal_lstm"
    )(x)

    # Dense Projection
    x = keras.layers.Dense(64, activation="relu", name="lstm_dense_proj")(x)
    x = keras.layers.BatchNormalization(momentum=0.85, name="lstm_batchnorm")(x)
    x = keras.layers.Dropout(0.35, name="lstm_dropout")(x)

    # 7-Class Softmax Classification
    outputs = keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="temporal_probabilities"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="FaceSense_CNN_LSTM")
    return model


def train_lstm_pipeline(
    epochs: int = 25,
    batch_size: int = 16,
    learning_rate: float = 0.001
):
    """
    Full training routine for the CNN + LSTM temporal network.
    Checks for available video dataset and honestly reports status if missing.
    """
    print("=" * 75)
    print("  FaceSense AI – CNN + LSTM Temporal Neural Network Training")
    print("=" * 75)
    print(f"[*] Base CNN Model:       {config.MODEL_PATH}")
    print(f"[*] Target LSTM Model:    {config.LSTM_MODEL_PATH}")
    print(f"[*] Sequence Length:      {config.SEQUENCE_LENGTH} frames")
    print(f"[*] CNN Feature Dim:      {config.FEATURE_DIM} (Layer: '{config.FEATURE_LAYER_NAME}')")
    print("=" * 75)

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.VIDEOS_DATASET_DIR, exist_ok=True)

    video_files = list(Path(config.VIDEOS_DATASET_DIR).glob("**/*.mp4")) + \
                  list(Path(config.VIDEOS_DATASET_DIR).glob("**/*.avi"))

    if len(video_files) == 0:
        print("\n[!] TRAINING STATUS: BLOCKED – Awaiting Video Dataset")
        print(f"    No video clips detected in: {config.VIDEOS_DATASET_DIR}")
        print("\n[i] Setup Instructions:")
        print("    1. Download RAVDESS speech video clips from https://zenodo.org/records/1188976")
        print(f"    2. Extract .mp4 files into: {config.VIDEOS_DATASET_DIR}")
        print("    3. Re-run: python training/train_lstm.py")
        print("=" * 75)

        # Update metadata marking model as not trained without fabricating results
        meta = load_lstm_metadata()
        meta["trained"] = False
        meta["status"] = "not_trained"
        meta["reason"] = f"No video files found in {config.VIDEOS_DATASET_DIR}. Awaiting RAVDESS dataset."
        meta["last_checked"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_lstm_metadata(meta)
        return False

    # Extract sequences from video dataset
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_video_dataset(
            video_dir=config.VIDEOS_DATASET_DIR,
            sequence_length=config.SEQUENCE_LENGTH,
            stride=4
        )
    except Exception as e:
        print(f"[!] Error processing video dataset: {str(e)}")
        return False

    if len(X_train) == 0:
        print("[!] No valid facial sequences could be constructed from available videos.")
        return False

    # Instantiate model architecture
    model = build_cnn_lstm_model(
        sequence_length=config.SEQUENCE_LENGTH,
        feature_dim=config.FEATURE_DIM,
        num_classes=config.NUM_CLASSES
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(config.LSTM_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1
        )
    ]

    print(f"\n[*] Commencing LSTM Sequence Training ({epochs} epochs, batch size {batch_size})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    model.save(str(config.LSTM_MODEL_PATH))
    print(f"[✓] Saved trained LSTM model to: {config.LSTM_MODEL_PATH}")

    # Evaluate on held-out test sequences
    eval_metrics = {}
    if len(X_test) > 0:
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        eval_metrics = {
            "test_accuracy": round(float(test_acc) * 100, 2),
            "test_loss": round(float(test_loss), 4),
            "test_sequences": int(len(X_test))
        }
        print(f"[✓] Test Accuracy: {eval_metrics['test_accuracy']}% (Loss: {eval_metrics['test_loss']})")

    # Update metadata
    meta = {
        "model_name": "FaceSense_CNN_LSTM",
        "trained": True,
        "status": "trained",
        "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": "RAVDESS_Video",
        "training_sequences": int(len(X_train)),
        "validation_sequences": int(len(X_val)),
        "sequence_length": config.SEQUENCE_LENGTH,
        "feature_dim": config.FEATURE_DIM,
        "classes": config.CLASS_NAMES,
        "evaluation": eval_metrics
    }
    save_lstm_metadata(meta)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN + LSTM Facial Expression Model")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate")
    args = parser.parse_args()

    train_lstm_pipeline(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
