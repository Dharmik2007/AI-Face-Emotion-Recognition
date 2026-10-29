"""
Image Processing Utilities for Facial Expression Recognition.
Handles OpenCV face detection, cropping, normalization, and base64 conversions.
"""

import base64
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config

# Cache the cascade classifier to avoid loading it on every request
_CASCADE_CLASSIFIER: Optional[cv2.CascadeClassifier] = None


def get_face_cascade() -> cv2.CascadeClassifier:
    """
    Load and return the Haar Cascade face detector classifier.
    Caches the instance in memory.
    """
    global _CASCADE_CLASSIFIER
    if _CASCADE_CLASSIFIER is None:
        cascade_path = str(config.HAARCASCADE_PATH)
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(
                f"Haar Cascade XML not found at {cascade_path}. "
                "Ensure model/haarcascade_frontalface_default.xml is present."
            )
        _CASCADE_CLASSIFIER = cv2.CascadeClassifier(cascade_path)
        if _CASCADE_CLASSIFIER.empty():
            raise RuntimeError(f"Failed to load Haar Cascade from {cascade_path}")
    return _CASCADE_CLASSIFIER


def detect_faces(image_bgr: np.ndarray, min_neighbors: int = 5) -> List[Tuple[int, int, int, int]]:
    """
    Detect frontal human faces using Haar Cascade.

    Args:
        image_bgr: NumPy array of the input image in BGR format.
        min_neighbors: Minimum neighboring rectangles to retain detection.

    Returns:
        List of bounding box tuples (x, y, w, h).
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    # Convert to grayscale
    if len(image_bgr.shape) == 3 and image_bgr.shape[2] == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for robust illumination handling
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    cascade = get_face_cascade()

    # First attempt with standard robust threshold
    faces = cascade.detectMultiScale(
        gray_eq,
        scaleFactor=1.1,
        minNeighbors=min_neighbors,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Controlled fallback only if no face is detected
    if len(faces) == 0 and min_neighbors > 4:
        faces = cascade.detectMultiScale(
            gray_eq,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(36, 36),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

    # Convert numpy array to list of tuples sorted by face size (largest first)
    faces_list = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    faces_list.sort(key=lambda b: b[2] * b[3], reverse=True)

    # Filter out spurious false-positive detections (e.g., wall textures, shadows)
    # Retain only secondary faces that are reasonably proportional to the primary face
    if len(faces_list) > 1:
        primary_area = faces_list[0][2] * faces_list[0][3]
        filtered = [faces_list[0]]
        for (x, y, w, h) in faces_list[1:]:
            area = w * h
            # Keep secondary face only if area is at least 25% of the primary face
            if area >= 0.25 * primary_area and w >= 48 and h >= 48:
                filtered.append((x, y, w, h))
        faces_list = filtered

    return faces_list


def check_face_quality(
    image_bgr: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]] = None
) -> dict:
    """
    Perform pre-inference face quality checks:
    - Illumination / Lighting (underexposed vs overexposed)
    - Blur / Focus (Laplacian variance)
    - Contrast distribution

    Returns a diagnostic report dictionary.
    """
    if image_bgr is None or image_bgr.size == 0:
        return {"status": "invalid", "is_usable": False, "message": "Empty image provided."}

    # Extract ROI or full frame
    if bbox is not None:
        x, y, w, h = bbox
        roi = image_bgr[max(0, y):y + h, max(0, x):x + w]
    else:
        roi = image_bgr

    if roi.size == 0:
        return {"status": "invalid", "is_usable": False, "message": "Zero-dimension face region."}

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

    # 1. Lighting / Luminance check
    mean_luminance = float(np.mean(gray))
    # 2. Sharpness / Blur check via Laplacian variance
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # 3. Contrast check via standard deviation
    contrast = float(np.std(gray))

    status = "optimal"
    warnings = []

    if mean_luminance < 35.0:
        status = "low_light"
        warnings.append("Low illumination: Face region is dark; accuracy may decrease.")
    elif mean_luminance > 225.0:
        status = "overexposed"
        warnings.append("High glare / Overexposed: Bright lighting wash may obscure facial contours.")

    if laplacian_var < 30.0:
        if status == "optimal":
            status = "blurry"
        warnings.append("Motion blur detected: Face features appear soft or unfocused.")

    return {
        "status": status,
        "is_usable": True,
        "luminance": round(mean_luminance, 1),
        "blur_score": round(laplacian_var, 1),
        "contrast": round(contrast, 1),
        "warnings": warnings,
        "message": "Quality optimal." if not warnings else "; ".join(warnings)
    }


def crop_and_preprocess_face(
    image_bgr: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    target_size: Tuple[int, int] = (48, 48),
    padding_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crops the face region (or uses entire image if bbox is None),
    converts to grayscale, resizes to target_size, and normalizes pixel values.

    Args:
        image_bgr: Source image in BGR format.
        bbox: Optional (x, y, w, h) bounding box.
        target_size: (width, height) desired for CNN input (default 48x48).
        padding_ratio: Margin to add around face bounding box.

    Returns:
        tensor: Preprocessed tensor of shape (1, 48, 48, 1) with values in [0.0, 1.0].
        visual_crop: The cropped face image in BGR format for web display.
    """
    h_img, w_img = image_bgr.shape[:2]

    if bbox is not None:
        x, y, w, h = bbox
        # Add slight margin to capture full facial context
        pad_x = int(w * padding_ratio)
        pad_y = int(h * padding_ratio)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)

        face_roi = image_bgr[y1:y2, x1:x2]
    else:
        face_roi = image_bgr

    # Visual crop copy
    visual_crop = face_roi.copy()

    # Grayscale conversion
    if len(face_roi.shape) == 3 and face_roi.shape[2] == 3:
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_roi.copy()

    # Resize to expected model dimensions (48x48) using AREA interpolation for downscaling
    resized = cv2.resize(face_gray, target_size, interpolation=cv2.INTER_AREA)

    # Normalize pixel intensity from [0, 255] to [0.0, 1.0]
    normalized = resized.astype(np.float32) / 255.0

    # Expand dimensions to match Keras input: (batch_size=1, height=48, width=48, channels=1)
    tensor = np.expand_dims(normalized, axis=(0, -1))

    return tensor, visual_crop


def draw_face_annotations(
    image_bgr: np.ndarray,
    faces: List[Tuple[int, int, int, int]],
    label: Optional[str] = None,
    confidence: Optional[float] = None
) -> np.ndarray:
    """
    Annotates the image with a stylish bounding box and expression tag.
    """
    annotated = image_bgr.copy()
    if not faces:
        return annotated

    # Choose color based on label if provided, else default emerald
    primary_color = (16, 185, 129)  # Emerald BGR: (129, 185, 16)
    if label and label in config.CLASS_COLORS:
        hex_color = config.CLASS_COLORS[label].lstrip("#")
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        primary_color = (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR

    for idx, (x, y, w, h) in enumerate(faces):
        # Draw bounding rectangle
        thickness = 2
        cv2.rectangle(annotated, (x, y), (x + w, y + h), primary_color, thickness)

        # Draw decorative corner accents
        corner_len = min(15, w // 4, h // 4)
        c_thick = 3
        # Top-left
        cv2.line(annotated, (x, y), (x + corner_len, y), primary_color, c_thick)
        cv2.line(annotated, (x, y), (x, y + corner_len), primary_color, c_thick)
        # Top-right
        cv2.line(annotated, (x + w, y), (x + w - corner_len, y), primary_color, c_thick)
        cv2.line(annotated, (x + w, y), (x + w, y + corner_len), primary_color, c_thick)
        # Bottom-left
        cv2.line(annotated, (x, y + h), (x + corner_len, y + h), primary_color, c_thick)
        cv2.line(annotated, (x, y + h), (x, y + h - corner_len), primary_color, c_thick)
        # Bottom-right
        cv2.line(annotated, (x + w, y + h), (x + w - corner_len, y + h), primary_color, c_thick)
        cv2.line(annotated, (x + w, y + h), (x + w, y + h - corner_len), primary_color, c_thick)

        # Draw label tag for primary (first) face
        if idx == 0 and label:
            text = f"{label}"
            if confidence is not None:
                text += f" ({confidence:.1f}%)"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            font_thick = 1
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thick)

            tag_y1 = max(0, y - text_h - 10)
            tag_y2 = y
            tag_x1 = x
            tag_x2 = min(annotated.shape[1], x + text_w + 14)

            # Draw tag background
            cv2.rectangle(annotated, (tag_x1, tag_y1), (tag_x2, tag_y2), primary_color, -1)
            # Draw text inside tag
            cv2.putText(
                annotated,
                text,
                (tag_x1 + 7, tag_y2 - 6),
                font,
                font_scale,
                (255, 255, 255),
                font_thick,
                cv2.LINE_AA
            )

    return annotated


def image_to_base64(image_bgr: np.ndarray, format_ext: str = ".jpg") -> str:
    """
    Encode an OpenCV BGR image into a Base64 data URL string.
    """
    success, buffer = cv2.imencode(format_ext, image_bgr)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    if format_ext == ".webp":
        mime = "image/webp"
    elif format_ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    else:
        mime = "image/png"
    return f"data:{mime};base64,{b64_str}"


def base64_to_image(b64_str: str) -> Optional[np.ndarray]:
    """
    Decode a Base64 data URL or raw Base64 string into an OpenCV BGR numpy array.
    """
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def is_allowed_file(filename: str) -> bool:
    """
    Validate whether the uploaded filename has an allowed extension.
    """
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in config.ALLOWED_EXTENSIONS
