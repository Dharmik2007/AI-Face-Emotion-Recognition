"""
FaceSense AI – Temporal Sequence Buffer
Maintains sliding-window buffers of CNN feature vectors for temporal sequence models (LSTM).

Key Features:
- Per-session isolated buffers to prevent temporal cross-talk between different users/streams
- Configurable sequence length (default: 16 frames)
- Sliding window mechanism with rolling FIFO queue
- Automatic tracking-loss reset when face is lost for consecutive frames
- Thread-safe session management with automated garbage collection
"""

import logging
import threading
import time
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)


class TemporalSequenceBuffer:
    """
    Sliding window buffer storing consecutive spatial feature embeddings
    extracted by the CNN for a specific webcam/video session.
    """

    def __init__(
        self,
        session_id: str = "default",
        sequence_length: Optional[int] = None,
        feature_dim: Optional[int] = None,
        max_missing_faces_to_reset: int = 4
    ):
        self.session_id = session_id
        self.sequence_length = int(sequence_length or config.SEQUENCE_LENGTH)
        self.feature_dim = int(feature_dim or config.FEATURE_DIM)
        
        self.buffer = deque(maxlen=self.sequence_length)
        self.bboxes = deque(maxlen=self.sequence_length)
        
        self.consecutive_missing_faces = 0
        self.max_missing_allowed = int(max_missing_faces_to_reset)
        
        self.created_at = time.time()
        self.last_updated = time.time()
        self._lock = threading.Lock()

    def add_frame(self, feature_vector: np.ndarray, bbox: Optional[dict] = None) -> dict:
        """Alias for add_feature."""
        return self.add_feature(feature_vector, bbox)

    def add_feature(self, feature_vector: np.ndarray, bbox: Optional[dict] = None) -> dict:
        """
        Append a single CNN feature vector to the rolling temporal queue.

        Args:
            feature_vector: Array of shape (256,) or (1, 256).
            bbox: Optional face bounding box dict {'x', 'y', 'w', 'h'}.

        Returns:
            Status dictionary describing current buffer progress.
        """
        with self._lock:
            # Flatten vector to 1D
            vec = np.asarray(feature_vector, dtype=np.float32).ravel()
            if len(vec) != self.feature_dim:
                raise ValueError(
                    f"Feature vector dimension mismatch: expected {self.feature_dim}, got {len(vec)}"
                )

            self.buffer.append(vec)
            self.bboxes.append(bbox or {})
            self.consecutive_missing_faces = 0
            self.last_updated = time.time()

            return self._status_unlocked()

    def record_missing_frame(self) -> dict:
        """
        Record that no face was detected in the current incoming frame.
        If face is absent for more than max_missing_allowed consecutive frames,
        resets the sequence buffer to avoid corrupted non-continuous sequences.
        """
        with self._lock:
            self.consecutive_missing_faces += 1
            if self.consecutive_missing_faces >= self.max_missing_allowed:
                if len(self.buffer) > 0:
                    logger.debug(
                        "Tracking lost for %d frames. Resetting sequence buffer for session %s.",
                        self.consecutive_missing_faces, self.session_id
                    )
                    self.buffer.clear()
                    self.bboxes.clear()

            self.last_updated = time.time()
            return self._status_unlocked()

    def is_ready(self) -> bool:
        """Return True when the buffer holds a full temporal sequence."""
        with self._lock:
            return len(self.buffer) == self.sequence_length

    def get_sequence(self) -> Optional[np.ndarray]:
        """Alias for get_sequence_array."""
        return self.get_sequence_array()

    def get_sequence_array(self) -> Optional[np.ndarray]:
        """
        Retrieve the current temporal sequence as a NumPy tensor ready for LSTM inference.

        Returns:
            Tensor of shape (1, sequence_length, feature_dim), float32, or None if not full.
        """
        with self._lock:
            if len(self.buffer) < self.sequence_length:
                return None
            seq_matrix = np.array(list(self.buffer), dtype=np.float32)
            # Add batch dimension: (1, sequence_length, feature_dim)
            return np.expand_dims(seq_matrix, axis=0)

    def reset(self) -> None:
        """Clear all buffered frames."""
        with self._lock:
            self.buffer.clear()
            self.bboxes.clear()
            self.consecutive_missing_faces = 0
            self.last_updated = time.time()

    def get_status(self) -> dict:
        """Thread-safe status snapshot."""
        with self._lock:
            return self._status_unlocked()

    def _status_unlocked(self) -> dict:
        cur_len = len(self.buffer)
        pct = round((cur_len / self.sequence_length) * 100, 1)
        return {
            "session_id": self.session_id,
            "current_frames": cur_len,
            "required_frames": self.sequence_length,
            "sequence_length": self.sequence_length,
            "is_ready": (cur_len == self.sequence_length),
            "progress_percent": pct,
            "feature_dim": self.feature_dim,
            "missing_streak": self.consecutive_missing_faces,
            "last_updated": self.last_updated
        }

    def get_progress_status(self) -> dict:
        """Alias for get_status() returning sequence buffering progress."""
        return self.get_status()


class SessionBufferManager:
    """
    Central registry managing isolated sequence buffers for concurrent browser sessions.
    Automatically garbage-collects idle sessions to prevent unbounded memory usage.
    """

    def __init__(self, session_ttl_seconds: int = 300):
        self._buffers: Dict[str, TemporalSequenceBuffer] = {}
        self._ttl = session_ttl_seconds
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str = "default", sequence_length: Optional[int] = None) -> TemporalSequenceBuffer:
        """Retrieve existing buffer or initialize a new one for session_id."""
        with self._lock:
            self._cleanup_stale_sessions()
            if session_id not in self._buffers:
                self._buffers[session_id] = TemporalSequenceBuffer(
                    session_id=session_id,
                    sequence_length=sequence_length or config.SEQUENCE_LENGTH,
                    feature_dim=config.FEATURE_DIM
                )
            return self._buffers[session_id]

    def get_buffer(self, session_id: str = "default", sequence_length: Optional[int] = None) -> TemporalSequenceBuffer:
        """Alias for get_or_create()."""
        return self.get_or_create(session_id, sequence_length)

    def reset_session(self, session_id: str) -> bool:
        """Reset buffer for a specific session."""
        with self._lock:
            if session_id in self._buffers:
                self._buffers[session_id].reset()
                return True
            return False

    def remove_session(self, session_id: str) -> bool:
        """Evict session buffer from memory."""
        with self._lock:
            if session_id in self._buffers:
                del self._buffers[session_id]
                return True
            return False

    def active_session_count(self) -> int:
        with self._lock:
            return len(self._buffers)

    def _cleanup_stale_sessions(self) -> None:
        """Evict sessions idle for longer than TTL."""
        now = time.time()
        stale_keys = [
            sid for sid, buf in self._buffers.items()
            if (now - buf.last_updated) > self._ttl
        ]
        for sid in stale_keys:
            del self._buffers[sid]


# Global buffer manager singleton
_BUFFER_MANAGER: Optional[SessionBufferManager] = None
_MANAGER_LOCK = threading.Lock()


def get_buffer_manager() -> SessionBufferManager:
    """Return the global SessionBufferManager instance."""
    global _BUFFER_MANAGER
    if _BUFFER_MANAGER is None:
        with _MANAGER_LOCK:
            if _BUFFER_MANAGER is None:
                _BUFFER_MANAGER = SessionBufferManager()
    return _BUFFER_MANAGER


# Alias for compatibility
get_session_manager = get_buffer_manager

