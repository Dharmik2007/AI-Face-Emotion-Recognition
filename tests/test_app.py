"""
Automated Test Suite for Human Facial Expression Recognition Project.
Tests:
- Home page, Health check, About page routes
- Invalid and corrupted image uploads
- Image preprocessing and face detection
- Missing model graceful handling
- Prediction response format and probabilities
- Error handling without server crashes
"""

import io
import os
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from flask import template_rendered

import config
from app import app
from utils.image_processing import (
    base64_to_image,
    crop_and_preprocess_face,
    detect_faces,
    draw_face_annotations,
    image_to_base64,
    is_allowed_file,
)
from utils.prediction import is_model_available, predict_expression


@pytest.fixture
def client():
    """Create Flask test client with test configuration."""
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as client:
        yield client


def create_dummy_image_bytes(width=100, height=100, color=(120, 120, 120), ext=".jpg"):
    """Generate in-memory encoded image bytes for upload testing."""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    success, buffer = cv2.imencode(ext, img)
    return io.BytesIO(buffer.tobytes())


def test_home_page(client):
    """Verify that the home page loads successfully with essential project elements."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Human Facial Expression Recognition" in html
    assert "BCA Semester 5" in html
    for class_name in config.CLASS_NAMES:
        assert class_name in html


def test_health_endpoint(client):
    """Verify system health check route returns valid JSON and status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "classes" in data
    assert len(data["classes"]) == 7


def test_about_page(client):
    """Verify architecture documentation page loads with CNN layer specifications."""
    response = client.get("/about")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Project Architecture" in html
    assert "Conv2D" in html
    assert "Softmax" in html


def test_allowed_file_validation():
    """Verify file extension validator accepts only supported formats including WEBP."""
    assert is_allowed_file("face.jpg") is True
    assert is_allowed_file("face.jpeg") is True
    assert is_allowed_file("face.png") is True
    assert is_allowed_file("face.webp") is True
    assert is_allowed_file("face.WEBP") is True
    assert is_allowed_file("face.JPG") is True
    assert is_allowed_file("script.py") is False
    assert is_allowed_file("document.pdf") is False
    assert is_allowed_file("malicious.exe") is False
    assert is_allowed_file("no_extension") is False


def test_invalid_file_extension_upload(client):
    """Verify uploading an unsupported file type returns an error without crashing."""
    data = {
        "image": (io.BytesIO(b"dummy text file content"), "test.txt")
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data", headers={"X-Requested-With": "XMLHttpRequest"})
    # Should reject with 400 Bad Request or 503 if model missing check comes first
    assert response.status_code in [400, 503]
    json_data = response.get_json()
    assert "error" in json_data["status"]


def test_empty_file_upload(client):
    """Verify uploading an empty file is caught cleanly."""
    data = {
        "image": (io.BytesIO(b""), "empty.jpg")
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data", headers={"X-Requested-With": "XMLHttpRequest"})
    assert response.status_code in [400, 503]
    json_data = response.get_json()
    assert "error" in json_data["status"]


def test_image_preprocessing_pipeline():
    """Verify face preprocessing crops, normalizes to [0, 1], and formats to (1, 48, 48, 1)."""
    dummy_face = np.full((120, 120, 3), 180, dtype=np.uint8)
    bbox = (20, 20, 60, 60)

    tensor, crop = crop_and_preprocess_face(dummy_face, bbox=bbox, target_size=(48, 48))

    assert tensor.shape == (1, 48, 48, 1)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0
    assert crop.shape[:2] != (0, 0)


def test_fallback_when_no_bbox():
    """Verify preprocessing handles uncropped full image fallback when bbox is None."""
    dummy_img = np.full((100, 100, 3), 200, dtype=np.uint8)
    tensor, crop = crop_and_preprocess_face(dummy_img, bbox=None, target_size=(48, 48))

    assert tensor.shape == (1, 48, 48, 1)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_base64_roundtrip():
    """Verify image to base64 encoding and decoding preserves image integrity."""
    original = np.zeros((50, 50, 3), dtype=np.uint8)
    original[10:30, 10:30] = [0, 255, 0]  # Green square

    b64 = image_to_base64(original, ".png")
    assert b64.startswith("data:image/png;base64,")

    decoded = base64_to_image(b64)
    assert decoded is not None
    assert decoded.shape == original.shape


def test_face_annotation_drawing():
    """Verify annotation function draws bounding box and returns image with identical dimensions."""
    sample_img = np.zeros((200, 200, 3), dtype=np.uint8)
    faces = [(40, 40, 80, 80)]
    annotated = draw_face_annotations(sample_img, faces, label="Happy", confidence=92.5)

    assert annotated.shape == sample_img.shape
    # Ensure drawing modified pixels inside image
    assert not np.array_equal(annotated, sample_img)


def test_missing_model_behavior(client):
    """Verify application handles missing model file gracefully without crashing."""
    with patch("app.is_model_available", return_value=False):
        # AJAX upload
        data = {
            "image": (create_dummy_image_bytes(), "test.jpg")
        }
        response = client.post("/predict", data=data, content_type="multipart/form-data", headers={"X-Requested-With": "XMLHttpRequest"})
        assert response.status_code == 503
        json_data = response.get_json()
        assert "Model file not found" in json_data["message"]

        # Webcam AJAX
        wb_response = client.post("/predict_webcam", json={"frame": "data:image/jpeg;base64,abc"})
        assert wb_response.status_code == 503


def test_prediction_output_with_mock_model():
    """Verify that when a model outputs logits, probabilities sum to 100% and format correctly."""
    mock_model = MagicMock()
    # Mock raw Softmax predictions: highest for 'Happy' (index 3)
    mock_model.predict.return_value = np.array([[0.05, 0.02, 0.03, 0.75, 0.05, 0.05, 0.05]])

    with patch("utils.prediction.get_model", return_value=mock_model):
        sample_tensor = np.zeros((1, 48, 48, 1), dtype=np.float32)
        result = predict_expression(sample_tensor)

        assert result["predicted_label"] == "Happy"
        assert result["confidence"] == 75.0
        assert len(result["probabilities"]) == 7
        assert "Happy" in result["probabilities"]
        assert result["probabilities"]["Happy"] == 75.0

        # Verify sum of probabilities is ~100%
        prob_sum = sum(result["probabilities"].values())
        assert abs(prob_sum - 100.0) < 1.0


def test_real_prediction_route_with_trained_model(client):
    """Verify /predict route processes real uploaded image and returns valid prediction."""
    # Only run if real model is present
    if not is_model_available():
        pytest.skip("Skipping real prediction test because model is not yet trained")

    img_bytes = create_dummy_image_bytes(width=120, height=120, color=(160, 160, 160))
    data = {
        "image": (img_bytes, "test_face.png")
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data", headers={"X-Requested-With": "XMLHttpRequest"})
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    assert json_data["predicted_label"] in config.CLASS_NAMES
    assert 0.0 <= json_data["confidence"] <= 100.0
    assert len(json_data["probabilities"]) == 7
    assert "quality" in json_data


def test_face_quality_check():
    """Verify face quality check evaluates illumination and blur accurately."""
    from utils.image_processing import check_face_quality

    # 1. Optimal image
    optimal_img = np.full((100, 100, 3), 130, dtype=np.uint8)
    res_opt = check_face_quality(optimal_img)
    assert res_opt["is_usable"] is True
    assert 120 <= res_opt["luminance"] <= 140

    # 2. Low light image
    dark_img = np.full((100, 100, 3), 15, dtype=np.uint8)
    res_dark = check_face_quality(dark_img)
    assert res_dark["status"] == "low_light"
    assert len(res_dark["warnings"]) > 0

    # 3. Overexposed image
    bright_img = np.full((100, 100, 3), 240, dtype=np.uint8)
    res_bright = check_face_quality(bright_img)
    assert res_bright["status"] == "overexposed"


def test_prediction_smoothing_ema():
    """Verify exponential moving average smoothing filters probability fluctuations."""
    mock_model = MagicMock()
    # Frame 1: 80% Happy
    mock_model.predict.return_value = np.array([[0.05, 0.02, 0.03, 0.80, 0.03, 0.03, 0.04]])

    with patch("utils.prediction.get_model", return_value=mock_model):
        sample_tensor = np.zeros((1, 48, 48, 1), dtype=np.float32)
        # Previous frame had 80% Sad
        prev_probs = {
            "Angry": 5.0, "Disgust": 2.0, "Fear": 3.0,
            "Happy": 5.0, "Sad": 80.0, "Surprise": 2.0, "Neutral": 3.0
        }
        result = predict_expression(
            sample_tensor,
            previous_probabilities=prev_probs,
            smooth_alpha=0.65
        )

        # Smoothed Happy should be between 5% and 80%
        assert 45.0 <= result["probabilities"]["Happy"] <= 60.0
        # Smoothed Sad should be decaying from 80%
        assert 20.0 <= result["probabilities"]["Sad"] <= 35.0


def test_uncertainty_handling():
    """Verify low confidence predictions are transparently flagged as uncertain."""
    mock_model = MagicMock()
    # Uniform / ambiguous distribution: highest is only 20%
    mock_model.predict.return_value = np.array([[0.15, 0.14, 0.14, 0.20, 0.13, 0.12, 0.12]])

    with patch("utils.prediction.get_model", return_value=mock_model):
        sample_tensor = np.zeros((1, 48, 48, 1), dtype=np.float32)
        result = predict_expression(sample_tensor, confidence_threshold=35.0)

        assert result["is_uncertain"] is True
        assert "below the 35% threshold" in result["uncertainty_note"]


def test_api_model_info_endpoint(client):
    """Verify /api/model_info returns valid JSON model metadata."""
    response = client.get("/api/model_info")
    assert response.status_code == 200
    data = response.get_json()
    assert "classes" in data
    assert len(data["classes"]) == 7
    assert "input_shape" in data
    assert "framework" in data


def test_phase4_navigation_routes(client):
    """Verify all 5 Phase 4 direct routes load index.html with correct active tab."""
    # 1. Live Analysis
    res = client.get("/live")
    assert res.status_code == 200
    assert "Live Webcam Stream" in res.get_data(as_text=True)

    # 2. Upload Analysis
    res = client.get("/upload")
    assert res.status_code == 200
    assert "Upload Facial Image" in res.get_data(as_text=True)

    # 3. CNN vs LSTM
    res = client.get("/compare")
    assert res.status_code == 200
    assert "CNN vs CNN + LSTM Spatiotemporal Comparison" in res.get_data(as_text=True)

    # 4. Model Performance
    res = client.get("/performance")
    assert res.status_code == 200
    assert "Empirical Model Performance" in res.get_data(as_text=True)

    # 5. Demo Guide (redirects to /about)
    res = client.get("/demo", follow_redirects=True)
    assert res.status_code == 200
    assert "Faculty Examination &amp; Viva Guide" in res.get_data(as_text=True) or "Faculty Examination & Viva Guide" in res.get_data(as_text=True)


def test_api_lstm_info_endpoint(client):
    """Verify /api/lstm_info returns valid JSON data regarding LSTM spatiotemporal model."""
    response = client.get("/api/lstm_info")
    assert response.status_code == 200
    data = response.get_json()
    assert "status" in data
    assert "architecture" in data
    assert "classes" in data
    assert data["architecture"]["sequence_length"] == 16
    assert data["architecture"]["feature_dim"] == 256


def test_api_compare_models_endpoint(client):
    """Verify /api/compare_models returns architecture comparison information."""
    response = client.get("/api/compare_models")
    assert response.status_code == 200
    data = response.get_json()
    assert "modes" in data
    assert "cnn" in data["modes"]
    assert "cnn_lstm" in data["modes"]
    assert "comparison_table" in data




