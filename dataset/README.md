# FER-2013 Dataset Setup Guide

This project is built around the **FER-2013 (Facial Expression Recognition 2013)** dataset, originally introduced by Pierre-Luc Carrier and Aaron Courville for the ICML 2013 Challenges in Representation Learning.

---

## 1. Dataset Overview

The dataset contains **35,887 grayscale images** of faces, each fixed at **48×48 pixels**. The faces are centered and roughly occupy about the same amount of space in each image.

### 7 Facial Expression Classes:
| Label ID | Expression Class | Typical Visual Cues |
| :---: | :---: | :--- |
| **0** | **Angry** | Furrowed brows, tightened eyelids, compressed lips |
| **1** | **Disgust** | Wrinkled nose bridge, raised upper lip |
| **2** | **Fear** | Raised & straightened brows, widened eyes, parted lips |
| **3** | **Happy** | Raised cheek muscles, smiling mouth with crinkled outer eyes |
| **4** | **Sad** | Drooping corners of the mouth, inner brow corners raised |
| **5** | **Surprise** | High arched eyebrows, rounded wide open mouth |
| **6** | **Neutral** | Relaxed facial features without dominant muscle tension |

---

## 2. Standard CSV Format (`fer2013.csv`)

The official dataset is distributed as a single CSV file:
- `emotion`: Integer from `0` to `6` identifying the target expression.
- `pixels`: A string of `2,304` space-separated integers (`0`–`255`), representing the flattened 48×48 grayscale image row by row.
- `Usage`: Partition designation:
  - `Training`: 28,709 samples
  - `PublicTest`: 3,589 samples (validation set)
  - `PrivateTest`: 3,589 samples (final test evaluation set)

---

## 3. How to Obtain the Full Dataset

1. Visit Kaggle's Challenge page:
   [FER-2013 on Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
   or search for **"FER-2013"** on Kaggle datasets.
2. Download `fer2013.csv` (or the archive containing it).
3. Place `fer2013.csv` directly in this directory:
   ```
   dataset/
   └── fer2013.csv
   ```
4. Train the model using:
   ```bash
   python training/train.py --dataset dataset/fer2013.csv --epochs 25
   ```

---

## 4. Quick Verification Dataset (`fer2013_sample.csv`)

If you are setting up the project for the first time or want to verify that the training, evaluation, and web dashboard pipelines work end-to-end without waiting to download 300MB:
```bash
python training/generate_sample_data.py
```
This script creates a 700-sample balanced benchmark CSV at `dataset/fer2013_sample.csv`. You can train immediately with:
```bash
python training/train.py --dataset dataset/fer2013_sample.csv --epochs 15
```

---

## 5. Scientific & Ethical Disclaimer

Facial expression recognition models categorize **visible geometric muscle deformations** of the human face. In modern psychological science (Barrett et al., 2019), these outward muscular patterns do **not** have a 1-to-1 correspondence with internal subjective emotional states. The project labels identify visible expression categories rather than internal psychological truth.
