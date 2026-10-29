"""
FaceSense AI – Test Suite for Temporal Sequence Buffer
Tests:
1. FIFO sliding window behavior with configurable sequence length.
2. Buffer ready condition (requires full sequence_length frames).
3. Auto-reset upon consecutive missing face tracking loss (threshold = 4 frames).
4. Multi-session isolation across independent user sessions.
5. Manual buffer reset functionality.
"""

import time
import numpy as np
import pytest

from utils.sequence_buffer import (
    TemporalSequenceBuffer,
    SessionBufferManager,
    get_buffer_manager,
)


def test_buffer_initialization():
    buf = TemporalSequenceBuffer(session_id="test_init", sequence_length=16, feature_dim=256)
    assert buf.sequence_length == 16
    assert buf.feature_dim == 256
    assert not buf.is_ready()
    status = buf.get_status()
    assert status["current_frames"] == 0
    assert status["progress_percent"] == 0.0


def test_buffer_fifo_and_sliding_window():
    seq_len = 16
    feat_dim = 256
    buf = TemporalSequenceBuffer(session_id="test_fifo", sequence_length=seq_len, feature_dim=feat_dim)

    # Add 15 frames: should not be ready
    for i in range(seq_len - 1):
        vec = np.full((feat_dim,), float(i), dtype=np.float32)
        buf.add_frame(vec)
        assert not buf.is_ready()
        assert buf.get_sequence() is None

    # Add 16th frame: should be ready
    vec_15 = np.full((feat_dim,), 15.0, dtype=np.float32)
    buf.add_frame(vec_15)
    assert buf.is_ready()

    seq = buf.get_sequence()
    assert seq is not None
    assert seq.shape == (1, seq_len, feat_dim)
    assert np.isclose(seq[0, 0, 0], 0.0)
    assert np.isclose(seq[0, -1, 0], 15.0)

    # Add 17th frame: sliding window should evict the first frame (0.0)
    vec_16 = np.full((feat_dim,), 16.0, dtype=np.float32)
    buf.add_frame(vec_16)
    assert buf.is_ready()

    seq_after = buf.get_sequence()
    assert seq_after.shape == (1, seq_len, feat_dim)
    assert np.isclose(seq_after[0, 0, 0], 1.0)     # Frame 0 evicted, Frame 1 is now oldest
    assert np.isclose(seq_after[0, -1, 0], 16.0)   # Frame 16 is now newest


def test_buffer_tracking_loss_reset():
    seq_len = 16
    feat_dim = 256
    buf = TemporalSequenceBuffer(
        session_id="test_loss",
        sequence_length=seq_len,
        feature_dim=feat_dim,
        max_missing_faces_to_reset=4
    )

    # Fill 10 frames
    for i in range(10):
        vec = np.zeros((feat_dim,), dtype=np.float32)
        buf.add_frame(vec)
    assert buf.get_status()["current_frames"] == 10

    # 1 to 3 missing frames: buffer retained
    buf.record_missing_frame()
    buf.record_missing_frame()
    buf.record_missing_frame()
    assert buf.get_status()["current_frames"] == 10
    assert buf.get_status()["missing_streak"] == 3

    # 4th missing frame: tracking lost -> buffer reset
    buf.record_missing_frame()
    assert buf.get_status()["current_frames"] == 0
    assert not buf.is_ready()


def test_session_manager_isolation():
    manager = SessionBufferManager(session_ttl_seconds=60)
    buf_a = manager.get_or_create("session_user_A", sequence_length=16)
    buf_b = manager.get_or_create("session_user_B", sequence_length=16)

    assert buf_a is not buf_b

    # Add frames to Session A only
    for _ in range(8):
        buf_a.add_frame(np.zeros((256,), dtype=np.float32))

    assert buf_a.get_status()["current_frames"] == 8
    assert buf_b.get_status()["current_frames"] == 0

    # Reset Session A
    manager.reset_session("session_user_A")
    assert buf_a.get_status()["current_frames"] == 0


def test_invalid_feature_dimension_handling():
    buf = TemporalSequenceBuffer(session_id="test_dim", sequence_length=16, feature_dim=256)
    with pytest.raises(ValueError):
        # Pass vector with wrong dimension (128 instead of 256)
        buf.add_frame(np.zeros((128,), dtype=np.float32))
