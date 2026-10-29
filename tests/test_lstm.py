"""
FaceSense AI – Test Suite for CNN + LSTM Architecture
Tests:
1. Intermediate CNN feature extraction (bn_dense 256-D vector).
2. LSTM prediction logic (honest handling when untrained, tensor verification).
3. Flask endpoints for LSTM metadata and model comparison matrix.
4. Multi-mode webcam prediction API (cnn, lstm, compare).
"""

import base64
import json
import cv2
import numpy as np
import pytest

from app import app
import config
from utils.feature_extractor import CNNFeatureExtractor, get_feature_extractor, extract_features_from_face
from utils.lstm_prediction import (
    get_comparison_specs,
    get_lstm_summary_info,
    is_lstm_available,
    predict_temporal_sequence,
)


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_webcam_b64():
    """Create a synthetic 640x480 test image encoded as base64 JPEG."""
    canvas = np.full((480, 640, 3), 128, dtype=np.uint8)
    # Draw simple facial feature proxy
    cv2.circle(canvas, (320, 240), 90, (200, 200, 200), -1)
    _, buffer = cv2.imencode(".jpg", canvas)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


def test_feature_extractor_embedding_shape():
    """Test that the CNN feature extractor extracts a 256-D float32 vector."""
    extractor = get_feature_extractor()
    assert extractor.is_available()

    dummy_tensor = np.zeros((1, 48, 48, 1), dtype=np.float32)
    features = extractor.extract_features(dummy_tensor)
    assert features.shape == (1, config.FEATURE_DIM)
    assert features.dtype == np.float32

    vec = extract_features_from_face(dummy_tensor)
    assert vec.shape == (config.FEATURE_DIM,)
    assert vec.dtype == np.float32


def test_lstm_prediction_untrained_fallback():
    """
    Verify that an untrained LSTM model returns status='not_trained'
    and NEVER returns fabricated emotional predictions.
    """
    dummy_seq = np.zeros((1, config.SEQUENCE_LENGTH, config.FEATURE_DIM), dtype=np.float32)
    res = predict_temporal_sequence(dummy_seq)

    # When untrained, it must honestly declare status
    if not is_lstm_available():
        assert res["status"] == "not_trained"
        assert res["trained"] is False
        assert res["predicted_label"] is None
        assert res["confidence"] == 0.0
        assert res["is_uncertain"] is True
        assert "Awaiting" in res["uncertainty_note"] or "training" in res["message"].lower()


def test_api_lstm_info_endpoint(client):
    """Verify /api/lstm_info returns valid architecture and training metadata."""
    resp = client.get("/api/lstm_info")
    assert resp.status_code == 200
    data = json.loads(resp.data)

    assert "model_name" in data
    assert "status" in data
    assert "architecture" in data
    assert data["architecture"]["sequence_length"] == config.SEQUENCE_LENGTH
    assert data["architecture"]["feature_dim"] == config.FEATURE_DIM
    assert data["architecture"]["intermediate_cnn_layer"] == config.FEATURE_LAYER_NAME
    assert data["classes"] == config.CLASS_NAMES


def test_api_compare_models_endpoint(client):
    """Verify /api/compare_models returns comparison matrix and architecture dimensions."""
    resp = client.get("/api/compare_models")
    assert resp.status_code == 200
    data = json.loads(resp.data)

    assert "modes" in data
    assert "cnn" in data["modes"]
    assert "cnn_lstm" in data["modes"]
    assert "comparison_table" in data
    assert len(data["comparison_table"]) >= 5


def test_api_sequence_buffer_reset(client):
    """Verify /api/sequence_buffer_reset cleanly clears the session buffer."""
    resp = client.post(
        "/api/sequence_buffer_reset",
        json={"session_id": "test_reset_session"}
    )
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "success"
    assert data["session_id"] == "test_reset_session"


def test_predict_webcam_modes(client, sample_webcam_b64):
    """Test /predict_webcam across all three modes: cnn, lstm, compare."""
    for mode in ("cnn", "lstm", "compare"):
        resp = client.post(
            "/predict_webcam",
            json={
                "frame": sample_webcam_b64,
                "mode": mode,
                "session_id": f"test_mode_{mode}"
            }
        )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["mode"] == mode
        assert "temporal_buffer" in data
        assert "latency_ms" in data
