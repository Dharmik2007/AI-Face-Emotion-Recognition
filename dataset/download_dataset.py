"""
Dataset acquisition utility for FaceSense AI.
Downloads the official FER-2013 benchmark dataset from verified public archive
and extracts fer2013.csv into dataset/.
"""

import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent
CSV_PATH = DATASET_DIR / "fer2013.csv"
ZIP_PATH = DATASET_DIR / "fer2013.csv.zip"
DOWNLOAD_URL = "https://huggingface.co/datasets/chitradrishti/fer2013/resolve/main/fer2013.csv.zip"


def download_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    mb_downloaded = (count * block_size) / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    sys.stdout.write(f"\rDownloading FER-2013: {percent}% ({mb_downloaded:.1f}/{total_mb:.1f} MB)")
    sys.stdout.flush()


def setup_real_dataset():
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 10_000_000:
        print(f"Real FER-2013 dataset already exists at {CSV_PATH} ({CSV_PATH.stat().st_size / (1024*1024):.1f} MB)")
        return str(CSV_PATH)

    print(f"Fetching official FER-2013 archive from:\n  {DOWNLOAD_URL}")
    t0 = time.time()
    urllib.request.urlretrieve(DOWNLOAD_URL, str(ZIP_PATH), reporthook=download_progress)
    print(f"\nDownload completed in {time.time() - t0:.1f}s. Extracting archive...")

    with zipfile.ZipFile(str(ZIP_PATH), "r") as zf:
        zf.extractall(str(DATASET_DIR))

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    # Find the extracted CSV
    if not CSV_PATH.exists():
        # Check if extracted inside a subfolder
        for p in DATASET_DIR.glob("**/*.csv"):
            if "fer2013" in p.name.lower() and p != CSV_PATH:
                p.rename(CSV_PATH)
                break

    print(f"Real FER-2013 dataset ready at: {CSV_PATH}")
    return str(CSV_PATH)


if __name__ == "__main__":
    setup_real_dataset()
