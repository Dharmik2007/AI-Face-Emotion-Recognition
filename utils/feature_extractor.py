"""
FaceSense AI – CNN Feature Extractor
Extracts fixed-dimension spatial feature embeddings from facial image tensors
using the intermediate dense representation of the trained FER CNN model.

Layer used: 'bn_dense' (256-dimensional normalized spatial feature vector)
Architecture: Input (48, 48, 1) -> Conv Blocks 1-3 -> Flatten (4608) -> Dense(256) -> BatchNorm -> Feature Embedding (256-D)
"""

import logging
import os
import threading
from typing import Optional, Tuple

import keras
import numpy as np

import config

logger = logging.getLogger(__name__)

# Thread-safe global feature extractor singleton
_FEATURE_EXTRACTOR: Optional["CNNFeatureExtractor"] = None
_EXTRACTOR_LOCK = threading.Lock()


class CNNFeatureExtractor:
    """
    Sub-model wrapper extracting intermediate 256-D spatial embeddings
    from the trained FER-2013 CNN.
    """

    def __init__(self, model_path: Optional[str] = None, layer_name: Optional[str] = None):
        self.model_path = str(model_path or config.MODEL_PATH)
        self.layer_name = str(layer_name or config.FEATURE_LAYER_NAME)
        self.feature_dim = config.FEATURE_DIM
        self.sub_model: Optional[keras.Model] = None
        self._lock = threading.Lock()
        self._load_feature_model()

    def _load_feature_model(self) -> None:
        """Construct the intermediate feature extraction model from the base CNN."""
        if not os.path.exists(self.model_path):
            logger.warning(
                "CNN model file not found at %s. Feature extractor in standby.",
                self.model_path
            )
            self.sub_model = None
            return

        try:
            full_cnn = keras.models.load_model(self.model_path, compile=False)
            
            # Trace the Sequential model with a dummy tensor to establish input symbolic tensors in Keras 3
            dummy = np.zeros((1, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS), dtype=np.float32)
            _ = full_cnn(dummy, training=False)

            # Verify the target intermediate layer exists
            layer_names = [l.name for l in full_cnn.layers]
            if self.layer_name not in layer_names:
                # Fallback to dense_features if bn_dense not explicitly found
                if "dense_features" in layer_names:
                    self.layer_name = "dense_features"
                elif "flatten" in layer_names:
                    self.layer_name = "flatten"
                else:
                    self.layer_name = layer_names[-2]
                logger.warning(
                    "Target layer not found; fallback to intermediate layer: %s",
                    self.layer_name
                )

            intermediate_layer = full_cnn.get_layer(self.layer_name)
            
            # Select proper input symbol
            inp_tensor = full_cnn.inputs[0] if isinstance(full_cnn.inputs, list) else full_cnn.inputs
            self.sub_model = keras.Model(
                inputs=inp_tensor,
                outputs=intermediate_layer.output,
                name="cnn_spatial_feature_extractor"
            )
            
            # Warm-up sub-model inference
            _ = self.sub_model(dummy, training=False)
            self.feature_dim = int(self.sub_model.output_shape[-1])
            logger.info(
                "CNN Feature Extractor initialized: Layer '%s', Embedding Dim = %d",
                self.layer_name, self.feature_dim
            )
        except Exception as e:
            logger.error("Failed to initialize CNN Feature Extractor: %s", str(e))
            self.sub_model = None

    def is_available(self) -> bool:
        """Return True if the underlying sub-model is loaded and ready for inference."""
        return self.sub_model is not None

    def extract_features(self, face_tensor: np.ndarray) -> np.ndarray:
        """
        Extract normalized spatial feature embedding from a preprocessed face tensor.

        Args:
            face_tensor: Array of shape (1, 48, 48, 1) or (N, 48, 48, 1) with values in [0.0, 1.0].

        Returns:
            NumPy array of shape (N, feature_dim), float32.
        """
        if self.sub_model is None:
            with self._lock:
                if self.sub_model is None:
                    self._load_feature_model()

        if self.sub_model is None:
            raise RuntimeError(
                "Feature extractor unavailable. Please train or provide model/fer_cnn_model.keras"
            )

        # Shape validation
        if not isinstance(face_tensor, np.ndarray):
            face_tensor = np.asarray(face_tensor, dtype=np.float32)

        if face_tensor.ndim == 3:
            face_tensor = np.expand_dims(face_tensor, axis=0)

        if face_tensor.ndim != 4 or face_tensor.shape[1:3] != (config.IMAGE_HEIGHT, config.IMAGE_WIDTH):
            raise ValueError(
                f"Expected input tensor with shape (batch, 48, 48, 1), received {face_tensor.shape}"
            )

        with self._lock:
            # inference mode (training=False) to use stable moving statistics
            features = self.sub_model(face_tensor, training=False)
            if hasattr(features, "numpy"):
                features = features.numpy()
            return np.asarray(features, dtype=np.float32)

    def extract_single_vector(self, face_tensor: np.ndarray) -> np.ndarray:
        """
        Convenience helper to extract a single 1D feature vector of shape (feature_dim,).
        """
        feat = self.extract_features(face_tensor)
        return feat[0]


def get_feature_extractor() -> CNNFeatureExtractor:
    """Return the global CNNFeatureExtractor singleton instance."""
    global _FEATURE_EXTRACTOR
    if _FEATURE_EXTRACTOR is None:
        with _EXTRACTOR_LOCK:
            if _FEATURE_EXTRACTOR is None:
                _FEATURE_EXTRACTOR = CNNFeatureExtractor()
    return _FEATURE_EXTRACTOR


def extract_features_from_face(face_tensor: np.ndarray) -> np.ndarray:
    """
    Convenience helper to extract a single 1D feature vector of shape (feature_dim,)
    from a preprocessed face tensor.
    """
    extractor = get_feature_extractor()
    return extractor.extract_single_vector(face_tensor)

