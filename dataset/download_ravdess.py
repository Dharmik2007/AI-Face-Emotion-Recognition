"""
FaceSense AI – Video Dataset Acquisition Helper (RAVDESS)
Prepares the video dataset directory and provides setup instructions
for temporal facial expression analysis.

Dataset Recommended:
RAVDESS (The Ryerson Audio-Visual Database of Emotional Speech and Song)
Zenodo: https://zenodo.org/records/1188976
License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

INSTRUCTIONS = f"""
================================================================================
FaceSense AI – Real Video Dataset Setup Instructions (RAVDESS)
================================================================================

To train the CNN + LSTM temporal model on genuine human video expressions:

1. Target Directory:
   {config.VIDEOS_DATASET_DIR}

2. Supported Video Formats:
   - RAVDESS Audio-Visual or Video-Only speech files (.mp4)
     Example filenames:
       02-01-01-01-01-01-01.mp4  (Neutral)
       02-01-03-01-01-01-01.mp4  (Happy)
       02-01-04-01-01-01-01.mp4  (Sad)
       02-01-05-01-01-01-01.mp4  (Angry)
       02-01-06-01-01-01-01.mp4  (Fearful)
       02-01-07-01-01-01-01.mp4  (Disgust)
       02-01-08-01-01-01-01.mp4  (Surprised)

   - OR Standard Emotion Subfolders:
     dataset/videos/Angry/*.mp4
     dataset/videos/Disgust/*.mp4
     dataset/videos/Fear/*.mp4
     dataset/videos/Happy/*.mp4
     dataset/videos/Sad/*.mp4
     dataset/videos/Surprise/*.mp4
     dataset/videos/Neutral/*.mp4

3. Downloading from Zenodo:
   Visit: https://zenodo.org/records/1188976
   Download any Actor Video zip file (e.g., Video_Speech_Actor_01.zip),
   extract the .mp4 files directly into:
   {config.VIDEOS_DATASET_DIR}

4. Running the Training Pipeline:
   Once video files are placed, execute:
   python training/train_lstm.py

================================================================================
"""


def setup_video_directory():
    """Ensure dataset/videos directory exists and display guidance."""
    os.makedirs(config.VIDEOS_DATASET_DIR, exist_ok=True)
    os.makedirs(config.SEQUENCES_CACHE_DIR, exist_ok=True)

    video_files = list(Path(config.VIDEOS_DATASET_DIR).glob("**/*.mp4"))
    print(INSTRUCTIONS)
    print(f"Current Status: Found {len(video_files)} video files in {config.VIDEOS_DATASET_DIR}")
    if len(video_files) == 0:
        print("[!] No video files detected yet. Place .mp4 files into the directory above.")
    else:
        print(f"[✓] Ready to train! Run 'python training/train_lstm.py' to process sequences.")


if __name__ == "__main__":
    setup_video_directory()
