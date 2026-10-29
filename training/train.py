"""
CNN Model Training Pipeline for Facial Expression Recognition.
BCA Semester 5 - Neural Networks Mini Project.

Architecture:
- Input: 48x48x1 Grayscale Face Image
- Conv Block 1: Conv2D(32) + BatchNorm + ReLU + Conv2D(32) + BatchNorm + ReLU + MaxPool + Dropout(0.25)
- Conv Block 2: Conv2D(64) + BatchNorm + ReLU + Conv2D(64) + BatchNorm + ReLU + MaxPool + Dropout(0.25)
- Conv Block 3: Conv2D(128) + BatchNorm + ReLU + MaxPool + Dropout(0.25)
- Dense Head: Flatten + Dense(256) + BatchNorm + ReLU + Dropout(0.5) + Dense(7, Softmax)
- Optimizer: Adam
- Loss: Categorical Crossentropy
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import config
from training.generate_sample_data import generate_sample_fer_dataset
from training.preprocess import get_data_augmentation_generator, load_fer2013_from_csv


def build_cnn_model(input_shape=config.INPUT_SHAPE, num_classes=config.NUM_CLASSES) -> Sequential:
    """
    Construct the Deep Convolutional Neural Network architecture for FaceSense AI.
    Features:
    - 3 Convolutional stages with paired Conv2D layers
    - BatchNormalization with tuned momentum=0.85 for stable statistics
    - Spatial MaxPooling downsampling
    - Increasing Dropout regularization (0.25 -> 0.30 -> 0.50) to prevent overfitting
    - Dense representation head (256 units) + Softmax probability distribution
    """
    model = Sequential([
        # Input Layer
        Input(shape=input_shape, name="input_face_image"),

        # --- Convolutional Block 1 (48x48 -> 24x24) ---
        Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1"),
        BatchNormalization(momentum=0.85, name="bn1_1"),
        Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_2"),
        BatchNormalization(momentum=0.85, name="bn1_2"),
        MaxPooling2D(pool_size=(2, 2), name="pool1"),
        Dropout(0.25, name="dropout1"),

        # --- Convolutional Block 2 (24x24 -> 12x12) ---
        Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1"),
        BatchNormalization(momentum=0.85, name="bn2_1"),
        Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_2"),
        BatchNormalization(momentum=0.85, name="bn2_2"),
        MaxPooling2D(pool_size=(2, 2), name="pool2"),
        Dropout(0.25, name="dropout2"),

        # --- Convolutional Block 3 (12x12 -> 6x6) ---
        Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_1"),
        BatchNormalization(momentum=0.85, name="bn3_1"),
        Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_2"),
        BatchNormalization(momentum=0.85, name="bn3_2"),
        MaxPooling2D(pool_size=(2, 2), name="pool3"),
        Dropout(0.30, name="dropout3"),

        # --- Classification Head ---
        Flatten(name="flatten"),
        Dense(256, activation="relu", name="dense_features"),
        BatchNormalization(momentum=0.85, name="bn_dense"),
        Dropout(0.5, name="dropout_dense"),

        # Output Layer: 7 Softmax probabilities
        Dense(num_classes, activation="softmax", name="output_probabilities")
    ], name="FaceSense_FER_CNN")

    return model


def plot_and_save_training_history(history: tf.keras.callbacks.History, save_path: str):
    """
    Plot training & validation accuracy and loss graphs, saving to static/plots/.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#f8fafc")

    # Accuracy Plot
    ax1.set_facecolor("#ffffff")
    ax1.plot(epochs_range, acc, "o-", color="#3b82f6", label="Training Accuracy", linewidth=2)
    ax1.plot(epochs_range, val_acc, "s-", color="#10b981", label="Validation Accuracy", linewidth=2)
    ax1.set_title("Model Accuracy Across Epochs", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Loss Plot
    ax2.set_facecolor("#ffffff")
    ax2.plot(epochs_range, loss, "o-", color="#ef4444", label="Training Loss", linewidth=2)
    ax2.plot(epochs_range, val_loss, "s-", color="#f59e0b", label="Validation Loss", linewidth=2)
    ax2.set_title("Model Loss Across Epochs", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss (Categorical Crossentropy)", fontsize=11)
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Training history visualization saved to: {save_path}")


def train(
    dataset_path: str,
    epochs: int = 25,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    augment: bool = True
):
    """
    Execute full training pipeline.
    """
    print("=" * 70)
    print("HUMAN FACIAL EXPRESSION RECOGNITION - CNN TRAINING PIPELINE")
    print("=" * 70)

    # Check if dataset file exists
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}.")
        if "sample" in dataset_path or not os.path.exists(config.FULL_DATASET_PATH):
            print("Auto-generating verification sample dataset (dataset/fer2013_sample.csv)...")
            dataset_path = generate_sample_fer_dataset()
        else:
            raise FileNotFoundError(f"Specified dataset path {dataset_path} does not exist.")

    # 1. Load Preprocessed Data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_fer2013_from_csv(dataset_path)

    # 2. Build Model
    print("\nConstructing CNN Architecture...")
    model = build_cnn_model()
    model.summary()

    # 3. Compile Model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 4. Setup Callbacks
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        filepath=str(config.MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        verbose=1
    )

    callbacks = [checkpoint_callback, early_stopping, reduce_lr]

    # 5. Model Training
    print(f"\nCommencing Model Training (Epochs: {epochs}, Batch Size: {batch_size})...")

    if augment:
        print("Data Augmentation: ENABLED (Rotation, shifts, horizontal flip)")
        datagen = get_data_augmentation_generator()
        datagen.fit(X_train)

        steps = int(np.ceil(len(X_train) / float(batch_size)))
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_val, y_val),
            steps_per_epoch=steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    else:
        print("Data Augmentation: DISABLED")
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

    # 6. Ensure Best Model is Saved
    model.save(str(config.MODEL_PATH))
    print(f"\nModel saved successfully to: {config.MODEL_PATH}")

    # 7. Plot Learning Curves
    history_plot_path = str(config.PLOTS_DIR / "training_history.png")
    plot_and_save_training_history(history, history_plot_path)

    # 8. Comprehensive Test Evaluation and Metrics Generation
    print("\nRunning comprehensive evaluation report on held-out test split...")
    from training.evaluate import evaluate
    eval_results = evaluate(model_path=str(config.MODEL_PATH), dataset_path=dataset_path)

    print("\nTraining and full evaluation pipeline completed successfully!")
    return model, history, eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FaceSense AI CNN model for Facial Expression Recognition")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(config.FULL_DATASET_PATH if config.FULL_DATASET_PATH.exists() else config.SAMPLE_DATASET_PATH),
        help="Path to fer2013 dataset CSV file"
    )
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--no_augment", action="store_true", help="Disable data augmentation")

    args = parser.parse_args()
    train(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        augment=not args.no_augment
    )
