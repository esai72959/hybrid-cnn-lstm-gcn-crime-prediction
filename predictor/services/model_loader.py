"""
model_loader.py

Production-grade singleton loader for the Hybrid CNN-LSTM Spatio-Temporal
Crime Prediction system.

This module is responsible exclusively for loading and caching all
deployment artifacts required at inference time:

    - The trained hybrid CNN-LSTM Keras model
    - The pretrained CNN feature extractor (spatial embedding)
    - The pretrained LSTM feature extractor (temporal embedding)
    - State/UT and District label encoders
    - The feature scaler
    - The ordered list of feature columns
    - Model/training metadata

All artifacts are loaded lazily and only once per process lifetime
(singleton pattern), avoiding redundant and expensive I/O / model
deserialization on every prediction request.

Author: Final Year B.Tech Project - Hybrid CNN-LSTM Crime Prediction
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import joblib
from tensorflow import keras

import hashlib

try:
    from src.gcn_model import GraphConvLayer
except ImportError:
    try:
        from gcn_model import GraphConvLayer
    except ImportError:
        GraphConvLayer = None

# --------------------------------------------------------------------------
# Logger configuration
# --------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


class ModelLoaderError(Exception):
    """Raised when a required model artifact cannot be located or loaded."""


class ModelLoader:
    """
    Singleton responsible for loading and caching all ML artifacts required
    for crime prediction inference.

    The class locates the project root automatically (relative to this
    file's location) and resolves all artifact paths from there, so it
    remains portable across operating systems and deployment machines.
    """

    _instance: Optional["ModelLoader"] = None
    _instance_lock: Lock = Lock()

    # Directory names relative to the project root.
    _MODELS_DIR = "models"
    _ARTIFACTS_DIR = "artifacts"

    # Artifact file names.
    _HYBRID_MODEL_FILE = "hybrid_model.keras"
    _HYBRID_GCN_MODEL_FILE = "hybrid_gcn_model.keras"
    _CNN_FEATURE_EXTRACTOR_FILE = "cnn_feature_extractor.keras"
    _LSTM_MODEL_FILE = "lstm_model.keras"
    _GCN_FEATURE_EXTRACTOR_FILE = "gcn_feature_extractor.keras"
    _GCN_FULL_MODEL_FILE = "gcn_model.keras"
    _CNN_EMBEDDING_LAYER = "embedding_bn"    # post-BatchNorm 64-dim spatial embedding
    _LSTM_EMBEDDING_LAYER = "lstm_embedding"  # 32-dim temporal embedding
    _GCN_EMBEDDING_LAYER = "gcn_embedding_bn" # 32-dim spatial-adjacency embedding
    _STATE_ENCODER_FILE = "state_ut_encoder.pkl"
    _DISTRICT_ENCODER_FILE = "district_encoder.pkl"
    _SCALER_FILE = "scaler.pkl"
    _FEATURE_COLUMNS_FILE = "feature_columns.json"
    _METADATA_FILE = "metadata.json"
    _ADJACENCY_FILE = "adjacency.pkl"
    _MANIFEST_FILE = "model_manifest.json"

    def __new__(cls) -> "ModelLoader":
        """Ensure only a single instance of ModelLoader ever exists."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """
        Initialize instance attributes exactly once.

        Because __new__ always returns the same instance, __init__ would
        otherwise reset already-loaded artifacts on every ModelLoader()
        call. The `_initialized` guard prevents that.
        """
        if self._initialized:
            return

        self._load_lock: Lock = Lock()

        # Cached artifacts (None until loaded).
        self.model: Optional[keras.Model] = None
        self.hybrid_gcn_model: Optional[keras.Model] = None
        self.cnn_feature_extractor: Optional[keras.Model] = None
        self.lstm_feature_extractor: Optional[keras.Model] = None
        self.gcn_feature_extractor: Optional[keras.Model] = None
        self.adjacency_data: Optional[Dict[str, Any]] = None
        self.state_encoder: Optional[Any] = None
        self.district_encoder: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.feature_columns: Optional[List[str]] = None
        self.metadata: Optional[Dict[str, Any]] = None

        # Resolve project root: predictor/services/model_loader.py -> root.
        self.project_root: Path = Path(__file__).resolve().parents[2]
        self.models_dir: Path = self.project_root / self._MODELS_DIR
        self.artifacts_dir: Path = self.project_root / self._ARTIFACTS_DIR

        logger.info("ModelLoader initialized. Project root resolved to: %s",
                    self.project_root)

        self._initialized = True

    # ----------------------------------------------------------------
    # Internal helpers & Checksum Verification
    # ----------------------------------------------------------------
    @staticmethod
    def _compute_sha256(path: Path) -> str:
        """Computes SHA-256 hash of a file on disk."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_manifest(self) -> Optional[Dict[str, Any]]:
        """Loads and returns model_manifest.json if available."""
        manifest_path = self.models_dir / self._MANIFEST_FILE
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning("Could not parse model_manifest.json: %s", exc)
        return None

    def _verify_and_log_artifact(self, model_key: str, model_path: Path) -> None:
        """Verifies checkpoint against model_manifest.json and logs exact checksum."""
        sha = self._compute_sha256(model_path)
        size_kb = model_path.stat().st_size / 1024.0
        manifest = self.get_manifest()
        
        status_msg = "verified"
        if manifest and "models" in manifest and model_key in manifest["models"]:
            expected_sha = manifest["models"][model_key].get("sha256")
            if expected_sha and expected_sha != sha:
                status_msg = f"MISMATCH (expected {expected_sha[:12]}..., got {sha[:12]}...)"
                logger.warning("Checksum MISMATCH for '%s'! Disk file does not match manifest!", model_path.name)
            else:
                status_msg = "verified against manifest"
        
        logger.info(
            "Loaded checkpoint: %s [SHA256: %s... | Size: %.1f KB | Checksum Status: %s]",
            model_path.name, sha[:12], size_kb, status_msg
        )

    @staticmethod
    def _require_file(path: Path) -> None:
        """
        Verify that a required artifact file exists on disk.

        Parameters
        ----------
        path : Path
            Path to the required file.

        Raises
        ------
        ModelLoaderError
            If the file does not exist at the given path.
        """
        if not path.exists() or not path.is_file():
            raise ModelLoaderError(
                f"Required artifact not found: '{path}'. "
                "Please verify the project directory structure."
            )

    # ----------------------------------------------------------------
    # Individual artifact loaders
    # ----------------------------------------------------------------
    def load_model(self) -> keras.Model:
        """
        Load the trained hybrid CNN-LSTM Keras model (lazy singleton).

        Returns
        -------
        keras.Model
            The loaded hybrid model, ready for inference.

        Raises
        ------
        ModelLoaderError
            If the model file is missing or fails to load.
        """
        if self.model is None:
            model_path = self.models_dir / self._HYBRID_MODEL_FILE
            try:
                self._require_file(model_path)
                self.model = keras.models.load_model(model_path)
                self._verify_and_log_artifact("hybrid_2way", model_path)
            except ModelLoaderError:
                logger.error("Model file missing at: %s", model_path)
                raise
            except Exception as exc:
                logger.error("Failed to load hybrid model from '%s': %s",
                              model_path, exc)
                raise ModelLoaderError(
                    f"Failed to load hybrid model: {exc}"
                ) from exc
        return self.model

    def load_cnn_feature_extractor(self) -> keras.Model:
        """
        Load the pretrained CNN spatial branch (lazy singleton) and wrap
        it in a frozen sub-model that outputs the `_CNN_EMBEDDING_LAYER`
        embedding instead of the branch's own auxiliary crime-count
        prediction.
        """
        if self.cnn_feature_extractor is None:
            model_path = self.models_dir / self._CNN_FEATURE_EXTRACTOR_FILE
            try:
                self._require_file(model_path)
                try:
                    cnn_model = keras.models.load_model(model_path, compile=False)
                    embedding_layer = cnn_model.get_layer(self._CNN_EMBEDDING_LAYER)
                    feature_extractor = keras.Model(
                        inputs=cnn_model.input,
                        outputs=embedding_layer.output,
                        name="cnn_feature_extractor",
                    )
                except Exception as exc:
                    logger.warning(
                        "Direct CNN load failed (%s); building explicit canonical architecture and loading weights...",
                        exc,
                    )
                    from tensorflow.keras import layers, Input, Model
                    feature_cols = self.load_feature_columns()
                    non_spatial = {"TOTAL IPC CRIMES", "YEAR", "YEAR_INDEX", "Id", "ID", "id", "STATE/UT", "DISTRICT", "State/UT", "District"}
                    spatial_cols = [c for c in feature_cols if c not in non_spatial]
                    n_features = len(spatial_cols) if len(spatial_cols) > 0 else 33

                    inputs = Input(shape=(n_features, 1), name="spatial_input")
                    x = layers.Conv1D(64, kernel_size=3, padding="same", name="conv1d_1")(inputs)
                    x = layers.BatchNormalization(name="bn_1")(x)
                    x = layers.ReLU(name="relu_1")(x)
                    x = layers.MaxPooling1D(pool_size=2, name="pool_1")(x)
                    x = layers.Conv1D(128, kernel_size=3, padding="same", name="conv1d_2")(x)
                    x = layers.BatchNormalization(name="bn_2")(x)
                    x = layers.ReLU(name="relu_2")(x)
                    x = layers.GlobalAveragePooling1D(name="gap")(x)
                    x = layers.Dense(128, name="dense_1")(x)
                    x = layers.Dropout(0.3, name="dropout_1")(x)
                    x = layers.Dense(64, activation="relu", name="spatial_embedding")(x)
                    emb = layers.BatchNormalization(name="embedding_bn")(x)
                    pred = layers.Dense(1, name="crime_count_prediction")(emb)

                    cnn_full = Model(inputs=inputs, outputs=pred, name="CrimeCNN_Trainer")
                    try:
                        cnn_full.load_weights(model_path, by_name=True, skip_mismatch=True)
                    except Exception as wexc:
                        logger.warning("Weight loading with skip_mismatch: %s", wexc)

                    feature_extractor = Model(inputs=inputs, outputs=emb, name="cnn_feature_extractor")

                feature_extractor.trainable = False
                self.cnn_feature_extractor = feature_extractor
                logger.info(
                    "CNN feature extractor loaded successfully from: %s",
                    model_path,
                )
            except ModelLoaderError:
                logger.error("CNN feature extractor file missing at: %s",
                              model_path)
                raise
            except Exception as exc:
                logger.error(
                    "Failed to load CNN feature extractor from '%s': %s",
                    model_path, exc,
                )
                raise ModelLoaderError(
                    f"Failed to load CNN feature extractor: {exc}"
                ) from exc
        return self.cnn_feature_extractor

    def load_lstm_feature_extractor(self) -> keras.Model:
        """
        Load the pretrained LSTM temporal branch (lazy singleton) and
        wrap it in a frozen sub-model that outputs the
        `_LSTM_EMBEDDING_LAYER` embedding instead of the branch's own
        crime-count prediction. Mirrors create_lstm_feature_extractor()
        in hybrid_model.py exactly, so the embedding this returns
        matches the one the hybrid model was trained on.

        Returns
        -------
        keras.Model
            Frozen LSTM feature extractor sub-model, ready for inference.

        Raises
        ------
        ModelLoaderError
            If the LSTM model file is missing, fails to load, or does
            not contain a layer named `_LSTM_EMBEDDING_LAYER`.
        """
        if self.lstm_feature_extractor is None:
            model_path = self.models_dir / self._LSTM_MODEL_FILE
            try:
                self._require_file(model_path)
                lstm_model = keras.models.load_model(model_path)
                lstm_model.trainable = False

                embedding_layer = lstm_model.get_layer(self._LSTM_EMBEDDING_LAYER)
                feature_extractor = keras.Model(
                    inputs=lstm_model.input,
                    outputs=embedding_layer.output,
                    name="lstm_feature_extractor",
                )
                feature_extractor.trainable = False

                self.lstm_feature_extractor = feature_extractor
                logger.info(
                    "LSTM feature extractor loaded successfully from: %s "
                    "(embedding dim: %d)", model_path,
                    embedding_layer.output_shape[-1],
                )
            except ModelLoaderError:
                logger.error("LSTM feature extractor file missing at: %s",
                              model_path)
                raise
            except Exception as exc:
                logger.error(
                    "Failed to load LSTM feature extractor from '%s': %s",
                    model_path, exc,
                )
                raise ModelLoaderError(
                    f"Failed to load LSTM feature extractor: {exc}"
                ) from exc
        return self.lstm_feature_extractor

    def load_encoders(self) -> None:
        """
        Load the State/UT and District label encoders (lazy singleton).

        Raises
        ------
        ModelLoaderError
            If either encoder file is missing or fails to load.
        """
        if self.state_encoder is None:
            state_path = self.artifacts_dir / self._STATE_ENCODER_FILE
            try:
                self._require_file(state_path)
                self.state_encoder = joblib.load(state_path)
                logger.info("State/UT encoder loaded successfully from: %s",
                            state_path)
            except ModelLoaderError:
                logger.error("State/UT encoder file missing at: %s",
                              state_path)
                raise
            except Exception as exc:
                logger.error("Failed to load State/UT encoder from '%s': %s",
                              state_path, exc)
                raise ModelLoaderError(
                    f"Failed to load State/UT encoder: {exc}"
                ) from exc

        if self.district_encoder is None:
            district_path = self.artifacts_dir / self._DISTRICT_ENCODER_FILE
            try:
                self._require_file(district_path)
                self.district_encoder = joblib.load(district_path)
                logger.info("District encoder loaded successfully from: %s",
                            district_path)
            except ModelLoaderError:
                logger.error("District encoder file missing at: %s",
                              district_path)
                raise
            except Exception as exc:
                logger.error(
                    "Failed to load District encoder from '%s': %s",
                    district_path, exc
                )
                raise ModelLoaderError(
                    f"Failed to load District encoder: {exc}"
                ) from exc

    def load_scaler(self) -> Any:
        """
        Load the feature scaler used to normalize model inputs.

        Returns
        -------
        Any
            The loaded scaler object (e.g. sklearn StandardScaler /
            MinMaxScaler).

        Raises
        ------
        ModelLoaderError
            If the scaler file is missing or fails to load.
        """
        if self.scaler is None:
            scaler_path = self.artifacts_dir / self._SCALER_FILE
            try:
                self._require_file(scaler_path)
                self.scaler = joblib.load(scaler_path)
                logger.info("Scaler loaded successfully from: %s",
                            scaler_path)
            except ModelLoaderError:
                logger.error("Scaler file missing at: %s", scaler_path)
                raise
            except Exception as exc:
                logger.error("Failed to load scaler from '%s': %s",
                              scaler_path, exc)
                raise ModelLoaderError(
                    f"Failed to load scaler: {exc}"
                ) from exc
        return self.scaler

    def load_feature_columns(self) -> List[str]:
        """
        Load the ordered list of feature columns expected by the model.

        Returns
        -------
        List[str]
            Ordered feature column names.

        Raises
        ------
        ModelLoaderError
            If the feature columns file is missing, malformed, or fails
            to load.
        """
        if self.feature_columns is None:
            columns_path = self.artifacts_dir / self._FEATURE_COLUMNS_FILE
            try:
                self._require_file(columns_path)
                with columns_path.open("r", encoding="utf-8") as file_obj:
                    data = json.load(file_obj)

                if isinstance(data, dict) and "feature_columns" in data:
                    self.feature_columns = data["feature_columns"]
                elif isinstance(data, list):
                    self.feature_columns = data
                else:
                    raise ModelLoaderError(
                        "Unexpected format in feature_columns.json. "
                        "Expected a list or a dict with key "
                        "'feature_columns'."
                    )

                logger.info("Feature columns loaded successfully from: %s "
                            "(%d columns)", columns_path,
                            len(self.feature_columns))
            except ModelLoaderError:
                logger.error("Feature columns file missing or invalid at: "
                              "%s", columns_path)
                raise
            except json.JSONDecodeError as exc:
                logger.error("Malformed JSON in feature columns file '%s': "
                              "%s", columns_path, exc)
                raise ModelLoaderError(
                    f"Malformed feature_columns.json: {exc}"
                ) from exc
            except Exception as exc:
                logger.error(
                    "Failed to load feature columns from '%s': %s",
                    columns_path, exc
                )
                raise ModelLoaderError(
                    f"Failed to load feature columns: {exc}"
                ) from exc
        return self.feature_columns

    def load_metadata(self) -> Dict[str, Any]:
        """
        Load model/training metadata (e.g. training date, metrics,
        version info).

        Returns
        -------
        Dict[str, Any]
            Parsed metadata dictionary.

        Raises
        ------
        ModelLoaderError
            If the metadata file is missing, malformed, or fails to load.
        """
        if self.metadata is None:
            metadata_path = self.artifacts_dir / self._METADATA_FILE
            try:
                self._require_file(metadata_path)
                with metadata_path.open("r", encoding="utf-8") as file_obj:
                    self.metadata = json.load(file_obj)
                logger.info("Metadata loaded successfully from: %s",
                            metadata_path)
            except ModelLoaderError:
                logger.error("Metadata file missing at: %s", metadata_path)
                raise
            except json.JSONDecodeError as exc:
                logger.error("Malformed JSON in metadata file '%s': %s",
                              metadata_path, exc)
                raise ModelLoaderError(
                    f"Malformed metadata.json: {exc}"
                ) from exc
            except Exception as exc:
                logger.error("Failed to load metadata from '%s': %s",
                              metadata_path, exc)
                raise ModelLoaderError(
                    f"Failed to load metadata: {exc}"
                ) from exc
        return self.metadata

    # ----------------------------------------------------------------
    # Public orchestration method
    # ----------------------------------------------------------------
    def load_all(self) -> None:
        """
        Load every required artifact (model, encoders, scaler, feature
        columns, and metadata) exactly once.

        This is the single entry point that should be called during
        application startup (e.g. Django AppConfig.ready(), or on first
        prediction request).

        Raises
        ------
        ModelLoaderError
            If any required artifact fails to load. The exception message
            identifies which artifact caused the failure.
        """
        with self._load_lock:
            logger.info("Starting full artifact load sequence...")
            try:
                self.load_model()
                self.load_cnn_feature_extractor()
                self.load_lstm_feature_extractor()
                self.load_encoders()
                self.load_scaler()
                self.load_feature_columns()
                self.load_metadata()
                logger.info("All artifacts loaded successfully. "
                            "ModelLoader is ready for inference.")
            except ModelLoaderError:
                logger.error("Artifact load sequence failed. "
                              "ModelLoader is NOT ready for inference.")
                raise

    # ----------------------------------------------------------------
    # Getter methods
    # ----------------------------------------------------------------
    def get_model(self) -> keras.Model:
        """
        Return the loaded hybrid CNN-LSTM model, loading it if necessary.

        Returns
        -------
        keras.Model
            The trained hybrid model.
        """
        return self.model if self.model is not None else self.load_model()

    def get_cnn_feature_extractor(self) -> keras.Model:
        """
        Return the loaded CNN feature extractor, loading it if necessary.

        Returns
        -------
        keras.Model
            Frozen CNN feature extractor sub-model producing the
            spatial embedding.
        """
        return (
            self.cnn_feature_extractor
            if self.cnn_feature_extractor is not None
            else self.load_cnn_feature_extractor()
        )

    def get_lstm_feature_extractor(self) -> keras.Model:
        """
        Return the loaded LSTM feature extractor, loading it if
        necessary.

        Returns
        -------
        keras.Model
            Frozen LSTM feature extractor sub-model producing the
            temporal embedding.
        """
        return (
            self.lstm_feature_extractor
            if self.lstm_feature_extractor is not None
            else self.load_lstm_feature_extractor()
        )

    def get_scaler(self) -> Any:
        """
        Return the loaded feature scaler, loading it if necessary.

        Returns
        -------
        Any
            The fitted scaler object.
        """
        return self.scaler if self.scaler is not None else self.load_scaler()

    def get_state_encoder(self) -> Any:
        """
        Return the loaded State/UT label encoder, loading it if necessary.

        Returns
        -------
        Any
            The fitted State/UT encoder.
        """
        if self.state_encoder is None:
            self.load_encoders()
        return self.state_encoder

    def get_district_encoder(self) -> Any:
        """
        Return the loaded District label encoder, loading it if necessary.

        Returns
        -------
        Any
            The fitted District encoder.
        """
        if self.district_encoder is None:
            self.load_encoders()
        return self.district_encoder

    def get_feature_columns(self) -> List[str]:
        """
        Return the ordered list of model feature columns, loading it if
        necessary.

        Returns
        -------
        List[str]
            Ordered feature column names.
        """
        if self.feature_columns is None:
            return self.load_feature_columns()
        return self.feature_columns

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return the model/training metadata, loading it if necessary.

        Returns
        -------
        Dict[str, Any]
            Metadata dictionary.
        """
        if self.metadata is None:
            return self.load_metadata()
        return self.metadata

    def load_hybrid_gcn_model(self) -> keras.Model:
        """
        Load the trained 3-Way Hybrid CNN-LSTM-GCN Keras model (128-dim input).
        """
        if self.hybrid_gcn_model is None:
            model_path = self.models_dir / self._HYBRID_GCN_MODEL_FILE
            try:
                self._require_file(model_path)
                self.hybrid_gcn_model = keras.models.load_model(model_path)
                self._verify_and_log_artifact("hybrid_3way_gcn", model_path)
            except ModelLoaderError:
                logger.error("Hybrid CNN-LSTM-GCN model file missing at: %s", model_path)
                raise
            except Exception as exc:
                logger.error("Failed to load Hybrid CNN-LSTM-GCN model from '%s': %s", model_path, exc)
                raise ModelLoaderError(f"Failed to load Hybrid CNN-LSTM-GCN model: {exc}") from exc
        return self.hybrid_gcn_model

    def load_gcn_feature_extractor(self) -> keras.Model:
        """
        Load the pretrained GCN spatial-adjacency branch feature extractor (32-dim output).
        """
        if self.gcn_feature_extractor is None:
            extractor_path = self.models_dir / self._GCN_FEATURE_EXTRACTOR_FILE
            full_path = self.models_dir / self._GCN_FULL_MODEL_FILE
            custom_objects = {"GraphConvLayer": GraphConvLayer} if GraphConvLayer else {}
            try:
                if extractor_path.exists():
                    self.gcn_feature_extractor = keras.models.load_model(extractor_path, custom_objects=custom_objects)
                elif full_path.exists():
                    full_gcn = keras.models.load_model(full_path, custom_objects=custom_objects)
                    layer = full_gcn.get_layer(self._GCN_EMBEDDING_LAYER)
                    self.gcn_feature_extractor = keras.Model(inputs=full_gcn.input, outputs=layer.output, name="gcn_feature_extractor")
                else:
                    raise ModelLoaderError(f"GCN model file not found at '{extractor_path}' or '{full_path}'")
                self.gcn_feature_extractor.trainable = False
                logger.info("GCN feature extractor loaded successfully.")
            except Exception as exc:
                logger.error("Failed to load GCN feature extractor: %s", exc)
                raise ModelLoaderError(f"Failed to load GCN feature extractor: {exc}") from exc
        return self.gcn_feature_extractor

    def load_adjacency(self) -> Dict[str, Any]:
        """
        Load the precomputed spatial graph adjacency artifacts from artifacts/adjacency.pkl.
        """
        if self.adjacency_data is None:
            adj_path = self.artifacts_dir / self._ADJACENCY_FILE
            try:
                self._require_file(adj_path)
                self.adjacency_data = joblib.load(adj_path)
                logger.info("Spatial graph adjacency artifacts loaded successfully from: %s", adj_path)
            except ModelLoaderError:
                logger.error("Adjacency artifact file missing at: %s", adj_path)
                raise
            except Exception as exc:
                logger.error("Failed to load adjacency artifacts: %s", exc)
                raise ModelLoaderError(f"Failed to load adjacency artifacts: {exc}") from exc
        return self.adjacency_data

    def get_hybrid_gcn_model(self) -> keras.Model:
        """Return the loaded Hybrid CNN-LSTM-GCN model, loading it if necessary."""
        return self.hybrid_gcn_model if self.hybrid_gcn_model is not None else self.load_hybrid_gcn_model()

    def get_gcn_feature_extractor(self) -> keras.Model:
        """Return the loaded GCN feature extractor, loading it if necessary."""
        return (
            self.gcn_feature_extractor
            if self.gcn_feature_extractor is not None
            else self.load_gcn_feature_extractor()
        )

    def get_adjacency(self) -> Dict[str, Any]:
        """Return the loaded spatial graph adjacency artifacts, loading if necessary."""
        return self.adjacency_data if self.adjacency_data is not None else self.load_adjacency()

    def warm_up_models(self) -> None:
        """
        Pre-loads all models and runs 1 dummy forward pass through each to compile
        graph execution kernels ahead of time, ensuring instant user predictions.
        """
        import numpy as np
        logger.info("Starting model warm-up...")
        try:
            cnn = self.get_cnn_feature_extractor()
            shape = tuple(dim if dim is not None else 1 for dim in cnn.input_shape)
            _ = cnn(np.zeros(shape, dtype=np.float32), training=False)
        except Exception as e:
            logger.warning("CNN warm-up failed: %s", e)

        try:
            lstm = self.get_lstm_feature_extractor()
            shape = tuple(dim if dim is not None else 1 for dim in lstm.input_shape)
            _ = lstm(np.zeros(shape, dtype=np.float32), training=False)
        except Exception as e:
            logger.warning("LSTM warm-up failed: %s", e)

        try:
            gcn = self.get_gcn_feature_extractor()
            shape = tuple(dim if dim is not None else 1 for dim in gcn.input_shape)
            _ = gcn(np.zeros(shape, dtype=np.float32), training=False)
        except Exception as e:
            logger.warning("GCN warm-up failed: %s", e)

        try:
            hybrid = self.get_model()
            shape = tuple(dim if dim is not None else 1 for dim in hybrid.input_shape)
            _ = hybrid(np.zeros(shape, dtype=np.float32), training=False)
        except Exception as e:
            logger.warning("Hybrid 2-Way warm-up failed: %s", e)

        try:
            hybrid_gcn = self.get_hybrid_gcn_model()
            shape = tuple(dim if dim is not None else 1 for dim in hybrid_gcn.input_shape)
            _ = hybrid_gcn(np.zeros(shape, dtype=np.float32), training=False)
        except Exception as e:
            logger.warning("Hybrid 3-Way warm-up failed: %s", e)

        logger.info("Model warm-up completed successfully. All models pre-compiled in memory.")

# ------------------------------------------------------------------
# Singleton instance
# ------------------------------------------------------------------

model_loader = ModelLoader()

