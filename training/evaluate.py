"""
Evaluation Script for Facial Expression Recognition CNN Model.
BCA Semester 5 - Neural Networks Mini Project.

Calculates:
- Overall Test Accuracy & Loss
- Precision, Recall, and F1-score per class
- Macro & Weighted Averages
- Confusion Matrix Heatmap (saved to static/plots/confusion_matrix.png)
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import config
from training.generate_sample_data import generate_sample_fer_dataset
from training.preprocess import load_fer2013_from_csv


def plot_and_save_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str):
    """
    Render and save a normalized confusion matrix heatmap using Matplotlib.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Normalize confusion matrix by row (true class counts)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor("#ffffff")

    # Display heatmap
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Normalized Proportion", rotation=-90, va="bottom", fontsize=10)

    # Ticks and Labels
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    ax.set_title("Facial Expression Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Ground Truth Class", fontsize=11, fontweight="semibold")
    ax.set_xlabel("Predicted Class by CNN", fontsize=11, fontweight="semibold")

    # Annotate cells with both count and percentage
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_norm = cm_norm[i, j]
            count = cm[i, j]
            text_color = "white" if val_norm > thresh else "#0f172a"
            ax.text(
                j, i,
                f"{count}\n({val_norm:.1%})",
                ha="center", va="center",
                color=text_color,
                fontsize=8,
                fontweight="medium"
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix heatmap saved to: {save_path}")


def evaluate(model_path: str = str(config.MODEL_PATH), dataset_path: str = None):
    """
    Load saved model and evaluate on unseen test partition.
    """
    print("=" * 70)
    print("CNN MODEL COMPREHENSIVE EVALUATION REPORT")
    print("=" * 70)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            "Please train the model first by running: python training/train.py"
        )

    if dataset_path is None or not os.path.exists(dataset_path):
        if os.path.exists(config.FULL_DATASET_PATH):
            dataset_path = str(config.FULL_DATASET_PATH)
        elif os.path.exists(config.SAMPLE_DATASET_PATH):
            dataset_path = str(config.SAMPLE_DATASET_PATH)
        else:
            print("No dataset CSV found; generating verification sample dataset...")
            dataset_path = generate_sample_fer_dataset()

    print(f"Loading Test Dataset from: {dataset_path} ...")
    _, _, (X_test, y_test) = load_fer2013_from_csv(dataset_path)

    print(f"Loading Trained Model from: {model_path} ...")
    model = tf.keras.models.load_model(model_path)

    # 1. Model Predictions
    print(f"Running inference on {len(X_test)} test samples...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # 2. Overall Accuracy & Loss
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print("\n" + "-" * 70)
    print(f"OVERALL PERFORMANCE METRICS:")
    print(f"  Test Accuracy: {test_acc * 100:.2f}%")
    print(f"  Test Loss:     {test_loss:.4f}")
    print("-" * 70)

    # 3. Detailed Classification Report (Precision, Recall, F1)
    print("\nDETAILED PER-CLASS CLASSIFICATION REPORT:")
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=config.CLASS_NAMES,
        digits=4,
        zero_division=0
    )
    print(report_text)
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=config.CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    # 4. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_plot_path = str(config.PLOTS_DIR / "confusion_matrix.png")
    plot_and_save_confusion_matrix(cm, config.CLASS_NAMES, cm_plot_path)

    # 5. Export Metrics JSON for UI Dashboard
    per_class_metrics = {}
    for name in config.CLASS_NAMES:
        if name in report_dict:
            per_class_metrics[name] = {
                "precision": round(report_dict[name]["precision"] * 100, 2),
                "recall": round(report_dict[name]["recall"] * 100, 2),
                "f1_score": round(report_dict[name]["f1-score"] * 100, 2),
                "support": int(report_dict[name]["support"])
            }

    metrics_payload = {
        "test_accuracy": round(float(test_acc) * 100, 2),
        "test_loss": round(float(test_loss), 4),
        "total_test_samples": int(len(X_test)),
        "dataset_name": Path(dataset_path).name,
        "is_real_dataset": "sample" not in Path(dataset_path).name.lower(),
        "per_class": per_class_metrics,
        "macro_avg": {
            "precision": round(report_dict.get("macro avg", {}).get("precision", 0) * 100, 2),
            "recall": round(report_dict.get("macro avg", {}).get("recall", 0) * 100, 2),
            "f1_score": round(report_dict.get("macro avg", {}).get("f1-score", 0) * 100, 2)
        },
        "weighted_avg": {
            "precision": round(report_dict.get("weighted avg", {}).get("precision", 0) * 100, 2),
            "recall": round(report_dict.get("weighted avg", {}).get("recall", 0) * 100, 2),
            "f1_score": round(report_dict.get("weighted avg", {}).get("f1-score", 0) * 100, 2)
        },
        "confusion_matrix": cm.tolist()
    }

    import json
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    with open(config.EVALUATION_METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Evaluation metrics JSON exported to: {config.EVALUATION_METRICS_PATH}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    return {
        "accuracy": test_acc,
        "loss": test_loss,
        "classification_report": report_text,
        "metrics_json": metrics_payload,
        "confusion_matrix": cm
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained CNN model on FER-2013 test set")
    parser.add_argument("--model", type=str, default=str(config.MODEL_PATH), help="Path to trained model .keras file")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset CSV")
    args = parser.parse_args()
    evaluate(model_path=args.model, dataset_path=args.dataset)
