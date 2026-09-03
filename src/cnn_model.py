"""
cnn_model.py
============

Spatial feature extraction module for the Hybrid CNN-LSTM Crime Prediction
framework (B.Tech Final Year Project).

This module implements ``CrimeCNN``, a 1D-Convolutional Neural Network that
learns a dense spatial embedding from district/state-level engineered crime
features. This CNN is deliberately NOT a final predictor -- it terminates in
a 64-dimensional Dense embedding layer so that the learned representation can
later be concatenated / fed into an LSTM branch inside ``hybrid_model.py``
to build the final Hybrid CNN-LSTM architecture.

Usage
-----
    python src/cnn_model.py

Author: .
"""
from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend, safe for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# Deterministic seeding
RANDOM_SEED = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
tf.keras.utils.set_random_seed(RANDOM_SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import Input, Model, layers, optimizers, losses, metrics
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CrimeCNN")


# --------------------------------------------------------------------------- #
# Configuration dataclass
# --------------------------------------------------------------------------- #
@dataclass
class CNNConfig:
    """Configuration container for :class:`CrimeCNN`.

    Centralising the hyperparameters here keeps ``CrimeCNN`` importable and
    reusable (e.g. from ``hybrid_model.py``) without hardcoding values inside
    the class body.
    """

    dataset_path: Path = Path("dataset/Crimes_in_india_2001-2013_features.csv")
    model_dir: Path = Path("models")
    results_dir: Path = Path("results")
    model_filename: str = "cnn_feature_extractor.keras"
    report_filename: str = "cnn_training_report.md"
    loss_curve_filename: str = "cnn_loss_curve.png"

    target_column: str = "TOTAL IPC CRIMES"
    columns_to_exclude: List[str] = field(
        default_factory=lambda: [
            "TOTAL IPC CRIMES",
            "YEAR",
            "YEAR_INDEX",
            "Id",
            "ID",
            "id",
        ]
    )

    embedding_dim: int = 64
    validation_split: float = 0.2
    test_split: float = 0.1
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001
    random_state: int = 42


# --------------------------------------------------------------------------- #
# Main class
# --------------------------------------------------------------------------- #
class CrimeCNN:
    """1D-CNN spatial feature extractor for crime data.

    The network consumes a vector of engineered spatial/numerical crime
    features per record, reshapes it into a pseudo-sequence suitable for
    ``Conv1D`` layers, and learns a compact 64-dimensional embedding that
    summarises spatial structure in the data. The target column
    (``TOTAL IPC CRIMES``) is used only as a training-time supervisory
    signal via an auxiliary regression head that is discarded at inference
    time -- ``extract_features`` always returns the embedding, never the
    prediction.

    Parameters
    ----------
    config:
        A :class:`CNNConfig` instance. If omitted, sensible project defaults
        are used (paths are relative, so no hardcoded absolute paths are
        introduced).
    """

    def __init__(self, config: Optional[CNNConfig] = None) -> None:
        self.config = config or CNNConfig()

        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        self.config.results_dir.mkdir(parents=True, exist_ok=True)

        self.df: Optional[pd.DataFrame] = None
        self.feature_columns: List[str] = []

        self.X_train: Optional[np.ndarray] = None
        self.X_val: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        # Full model (features -> embedding -> auxiliary regression head).
        # Used for training only.
        self.training_model: Optional[Model] = None

        # Sub-model that outputs only the embedding. Used for inference /
        # downstream integration with the LSTM branch.
        self.feature_extractor: Optional[Model] = None

        self.history: Optional[tf.keras.callbacks.History] = None

        logger.info("CrimeCNN initialised with config: %s", self.config)

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #
    def load_dataset(self) -> pd.DataFrame:
        """Load the engineered feature dataset from disk.

        Returns
        -------
        pd.DataFrame
            The loaded dataset.

        Raises
        ------
        FileNotFoundError
            If the dataset CSV does not exist at ``self.config.dataset_path``.
        ValueError
            If the target column is missing from the dataset.
        """
        dataset_path = self.config.dataset_path
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{dataset_path}'. "
                "Ensure feature_engineering.py has been run first."
            )

        logger.info("Loading dataset from '%s'", dataset_path)
        self.df = pd.read_csv(dataset_path)

        if self.config.target_column not in self.df.columns:
            raise ValueError(
                f"Target column '{self.config.target_column}' not found in "
                f"dataset columns: {list(self.df.columns)}"
            )

        logger.info(
            "Dataset loaded successfully. Shape: %s", str(self.df.shape)
        )
        return self.df

    # ------------------------------------------------------------------ #
    # Input preparation
    # ------------------------------------------------------------------ #
    def prepare_spatial_input(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Select spatial/numerical predictor columns and reshape for Conv1D.

        Excludes the target column and non-predictive identifier/time
        columns (``YEAR``, ``YEAR_INDEX``, ``Id``). All remaining numeric
        columns (including ``STATE_ENCODED``, ``DISTRICT_ENCODED``,
        ``LATITUDE``, ``LONGITUDE`` and other engineered crime features) are
        treated as the spatial feature vector.

        The resulting 2D feature matrix ``(n_samples, n_features)`` is
        reshaped to 3D ``(n_samples, n_features, 1)`` as required by
        ``Conv1D``, treating each feature as one timestep of a
        single-channel pseudo-sequence.

        Returns
        -------
        Tuple[np.ndarray, ...]
            ``(X_train, X_val, X_test, y_train, y_val, y_test)``

        Raises
        ------
        RuntimeError
            If called before :meth:`load_dataset`.
        """
        if self.df is None:
            raise RuntimeError("Call load_dataset() before prepare_spatial_input().")

        exclude = set(self.config.columns_to_exclude)
        numeric_df = self.df.select_dtypes(include=[np.number])
        self.feature_columns = [
            col for col in numeric_df.columns if col not in exclude
        ]

        if not self.feature_columns:
            raise ValueError(
                "No feature columns remain after exclusion. "
                "Check columns_to_exclude against the dataset schema."
            )

        logger.info(
            "Selected %d spatial/numerical feature columns.",
            len(self.feature_columns),
        )
        logger.debug("Feature columns: %s", self.feature_columns)

        X = self.df[self.feature_columns].to_numpy(dtype=np.float32)
        y = self.df[self.config.target_column].to_numpy(dtype=np.float32)

        # Guard against residual NaNs from upstream preprocessing.
        if np.isnan(X).any():
            logger.warning("NaNs detected in feature matrix; imputing with 0.0")
            X = np.nan_to_num(X, nan=0.0)
        if np.isnan(y).any():
            logger.warning("NaNs detected in target vector; imputing with 0.0")
            y = np.nan_to_num(y, nan=0.0)

        # Reshape to (samples, timesteps=n_features, channels=1) for Conv1D.
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # First split off the test set, then split remaining into train/val.
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_split,
            random_state=self.config.random_state,
        )
        relative_val_split = self.config.validation_split / (1 - self.config.test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=relative_val_split,
            random_state=self.config.random_state,
        )

        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

        logger.info(
            "Data split -> train: %s, val: %s, test: %s",
            self.X_train.shape,
            self.X_val.shape,
            self.X_test.shape,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    # ------------------------------------------------------------------ #
    # Model architecture
    # ------------------------------------------------------------------ #
    def build_model(self) -> Model:
        """Construct the CNN spatial feature extractor.

        Architecture
        ------------
        Input -> Conv1D(64, He-init, L2) -> BatchNorm -> ReLU -> MaxPooling1D ->
        Conv1D(128, He-init, L2) -> BatchNorm -> ReLU -> GlobalAveragePooling1D ->
        Dense(128, He-init, L2) -> Dropout(0.3) ->
        Dense(64, He-init, L2, 'relu') -> BatchNorm  [== embedding output]
        -> Dense(1, L2) [auxiliary regression head, training only]

        Regularization / stability additions:
        - L2 kernel regularization (1e-4) on every Conv1D/Dense layer to
          discourage overfitting on the ~9.3k-row dataset.
        - HeNormal initialization on all ReLU-activated Conv1D/Dense layers,
          matched to ReLU's activation statistics.
        - A BatchNormalization layer directly after the 64-dim embedding,
          so the embedding fed to the auxiliary head (and later consumed by
          hybrid_model.py) sits on a stable, consistently-scaled distribution.
        - Gradient clipping (clipnorm=1.0) on the Adam optimizer, set during
          ``compile()`` below.

        The embedding layer (post-BatchNorm ``Dense(64)``) is exposed
        separately via ``self.feature_extractor`` so downstream code (e.g.
        ``hybrid_model.py``) can consume pure 64-dim embeddings without any
        dependency on the auxiliary regression head.

        Returns
        -------
        keras.Model
            The full trainable model (features -> embedding -> prediction).
        """
        if self.X_train is None:
            raise RuntimeError("Call prepare_spatial_input() before build_model().")

        n_timesteps = self.X_train.shape[1]
        n_channels = self.X_train.shape[2]

        # L2 penalty applied to every Conv1D/Dense kernel. A small value
        # (1e-4) is used so it acts as a mild regularizer that discourages
        # large weights / overfitting without meaningfully slowing
        # convergence -- appropriate given the dataset is only ~9.3k rows.
        l2_reg = regularizers.l2(0.0001)

        # He (Kaiming) normal initialization is the standard choice for
        # layers followed by ReLU: it accounts for the fact that ReLU zeroes
        # out roughly half its inputs, keeping activation variance stable
        # across depth and reducing the risk of vanishing/exploding
        # gradients at the start of training.
        he_init = initializers.HeNormal(seed=self.config.random_state)

        inputs = Input(shape=(n_timesteps, n_channels), name="spatial_input")

        # --- Conv Block 1: local spatial pattern extraction ---
        x = layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding="same",
            kernel_initializer=he_init,
            kernel_regularizer=l2_reg,
            name="conv1d_1",
        )(inputs)
        x = layers.BatchNormalization(name="bn_1")(x)  # stabilizes/accelerates training
        x = layers.ReLU(name="relu_1")(x)
        x = layers.MaxPooling1D(pool_size=2, padding="same", name="maxpool_1")(x)

        # --- Conv Block 2: higher-level spatial feature composition ---
        x = layers.Conv1D(
            filters=128,
            kernel_size=3,
            padding="same",
            kernel_initializer=he_init,
            kernel_regularizer=l2_reg,
            name="conv1d_2",
        )(x)
        x = layers.BatchNormalization(name="bn_2")(x)
        x = layers.ReLU(name="relu_2")(x)

        # Global average pooling collapses the feature-map sequence into a
        # fixed-size vector regardless of input feature count, which keeps
        # the CNN reusable if the upstream feature set grows/shrinks later.
        x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

        x = layers.Dense(
            128,
            activation="relu",
            kernel_initializer=he_init,
            kernel_regularizer=l2_reg,
            name="dense_128",
        )(x)
        # Dropout(0.3) retained: with only ~9.3k rows and 39 engineered
        # features, this rate gives meaningful regularization without
        # starving the 128-unit layer of enough active units to learn from.
        x = layers.Dropout(0.3, name="dropout")(x)

        embedding_raw = layers.Dense(
            self.config.embedding_dim,
            activation="relu",
            kernel_initializer=he_init,
            kernel_regularizer=l2_reg,
            name="spatial_embedding",
        )(x)

        # BatchNormalization on the embedding itself keeps the 64-dim
        # vector on a consistent, well-scaled distribution. This matters
        # a lot for hybrid_model.py: an LSTM branch that later consumes
        # this embedding (possibly concatenated with other scaled inputs)
        # trains more stably when the incoming features aren't arbitrarily
        # scaled/shifted batch-to-batch. Dimensionality is unchanged (64).
        embedding = layers.BatchNormalization(name="embedding_bn")(embedding_raw)

        # Auxiliary regression head: exists purely to supervise embedding
        # learning during training. It is NOT used by extract_features().
        # L2 regularization is applied here too per the spatial-embedding
        # requirement; HeNormal is not used since this output is linear,
        # not ReLU (He init is specifically derived for ReLU variance).
        prediction = layers.Dense(
            1,
            activation="linear",
            kernel_regularizer=l2_reg,
            name="aux_prediction",
        )(embedding)

        self.training_model = Model(
            inputs=inputs, outputs=prediction, name="CrimeCNN_Trainer"
        )
        # feature_extractor exposes the BatchNorm'd embedding (post
        # embedding_bn) rather than the raw Dense(64) output, so downstream
        # consumers (hybrid_model.py) get the same well-scaled vector the
        # network was actually optimized against. Shape stays (n, 64).
        self.feature_extractor = Model(
            inputs=inputs, outputs=embedding, name="CrimeCNN_FeatureExtractor"
        )

        self.training_model.compile(
            # clipnorm=1.0 rescales the gradient vector whenever its L2 norm
            # exceeds 1.0. Combined with BatchNorm at two extra points
            # (post-embedding) and L2-regularized weights, this guards
            # against the occasional large/unstable gradient step that can
            # otherwise destabilize training on a fairly small dataset.
            optimizer=optimizers.Adam(
                learning_rate=self.config.learning_rate, clipnorm=1.0
            ),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanAbsoluteError(name="mae")],
        )

        logger.info("Model built and compiled successfully.")
        self.training_model.summary(print_fn=logger.info)

        return self.training_model

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self) -> tf.keras.callbacks.History:
        """Train the CNN using the configured hyperparameters.

        Uses ``EarlyStopping``, ``ReduceLROnPlateau`` and ``ModelCheckpoint``
        callbacks. The best checkpoint (lowest validation loss) is restored
        automatically via ``EarlyStopping(restore_best_weights=True)``.

        Returns
        -------
        keras.callbacks.History
            Training history object.
        """
        if self.training_model is None:
            raise RuntimeError("Call build_model() before train().")
        if self.X_train is None:
            raise RuntimeError("Call prepare_spatial_input() before train().")

        checkpoint_path = self.config.model_dir / "cnn_checkpoint_best.keras"

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=7,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
        ]

        logger.info(
            "Starting training for up to %d epochs (batch_size=%d)...",
            self.config.epochs,
            self.config.batch_size,
        )

        self.history = self.training_model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=2,
        )

        logger.info("Training complete.")

        test_loss, test_mae = self.training_model.evaluate(
            self.X_test, self.y_test, verbose=0
        )
        logger.info(
            "Held-out test set -> loss (MSE): %.4f | MAE: %.4f",
            test_loss,
            test_mae,
        )

        self._report_extended_metrics()

        self._plot_loss_curve()
        self._write_training_report(test_loss=test_loss, test_mae=test_mae)

        return self.history

    # ------------------------------------------------------------------ #
    # Extended evaluation metrics (MAE / RMSE / R^2)
    # ------------------------------------------------------------------ #
    def _report_extended_metrics(self) -> None:
        """Compute MAE, RMSE and R^2 on the held-out test set and persist
        raw predictions, so the CNN branch reports the same metrics as
        the LSTM and Hybrid models for side-by-side comparison.

        Raises
        ------
        RuntimeError
            If called before the model has been trained.
        """
        if self.training_model is None or self.X_test is None:
            raise RuntimeError(
                "Call prepare_spatial_input(), build_model() and train() "
                "before _report_extended_metrics()."
            )

        predictions = self.training_model.predict(self.X_test, verbose=0).flatten()
        actuals = self.y_test

        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)

        logger.info("=" * 60)
        logger.info("Test MAE  : %.4f", mae)
        logger.info("Test RMSE : %.4f", rmse)
        logger.info("Test R\u00b2   : %.4f", r2)
        logger.info("=" * 60)

        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = self.config.results_dir / "cnn_predictions.csv"
        pd.DataFrame({"Actual": actuals, "Predicted": predictions}).to_csv(
            predictions_path, index=False
        )
        logger.info("Predictions saved to: %s", predictions_path)

    # ------------------------------------------------------------------ #
    # Feature extraction (primary integration point for hybrid_model.py)
    # ------------------------------------------------------------------ #
    def extract_features(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Return learned spatial embeddings for the given input.

        Parameters
        ----------
        X:
            Optional input array of shape ``(n_samples, n_features, 1)``.
            If ``None``, embeddings are computed for the full dataset
            (train + val + test, in that order) using the same feature
            columns selected in :meth:`prepare_spatial_input`.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_samples, embedding_dim)`` containing the
            64-dimensional spatial embeddings. This is the sole artifact
            ``hybrid_model.py`` should consume from this class -- never the
            auxiliary regression output.

        Raises
        ------
        RuntimeError
            If the model has not been built yet.
        """
        if self.feature_extractor is None:
            raise RuntimeError("Call build_model() (and train()) before extract_features().")

        if X is None:
            if self.X_train is None:
                raise RuntimeError(
                    "No cached input available. Call prepare_spatial_input() "
                    "first or pass X explicitly."
                )
            X = np.concatenate([self.X_train, self.X_val, self.X_test], axis=0)

        embeddings = self.feature_extractor.predict(X, verbose=0)
        logger.info("Extracted embeddings with shape: %s", str(embeddings.shape))
        return embeddings

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_model(self) -> Path:
        """Save the full trainer model to ``models/cnn_feature_extractor.keras``.

        Returns
        -------
        Path
            Path to the saved model file.
        """
        if self.training_model is None:
            raise RuntimeError("Call build_model() before save_model().")

        save_path = self.config.model_dir / self.config.model_filename
        self.training_model.save(save_path)
        logger.info("Model saved to '%s'", save_path)
        return save_path

    # ------------------------------------------------------------------ #
    # Reporting utilities
    # ------------------------------------------------------------------ #
    def _plot_loss_curve(self) -> None:
        """Plot and save training/validation loss curves."""
        if self.history is None:
            logger.warning("No training history available; skipping loss plot.")
            return

        history_dict = self.history.history
        epochs_range = range(1, len(history_dict["loss"]) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs_range, history_dict["loss"], label="Training Loss")
        plt.plot(epochs_range, history_dict["val_loss"], label="Validation Loss")
        plt.title("CrimeCNN Training vs Validation Loss (MSE)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        out_path = self.config.results_dir / self.config.loss_curve_filename
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info("Loss curve saved to '%s'", out_path)

    def _write_training_report(self, test_loss: float, test_mae: float) -> None:
        """Write a Markdown training report to ``results/cnn_training_report.md``."""
        if self.history is None or self.df is None:
            logger.warning("Incomplete state; skipping report generation.")
            return

        history_dict = self.history.history
        final_epoch = len(history_dict["loss"])
        final_train_loss = history_dict["loss"][-1]
        final_val_loss = history_dict["val_loss"][-1]
        best_val_loss = min(history_dict["val_loss"])

        model_path = self.config.model_dir / self.config.model_filename

        report_lines = [
            "# CrimeCNN Training Report",
            "",
            "## Dataset",
            f"- Source file: `{self.config.dataset_path}`",
            f"- Dataset shape: {self.df.shape[0]} rows x {self.df.shape[1]} columns",
            f"- Input feature count: {len(self.feature_columns)}",
            f"- Target column: `{self.config.target_column}`",
            f"- Train / Val / Test samples: "
            f"{self.X_train.shape[0]} / {self.X_val.shape[0]} / {self.X_test.shape[0]}",
            "",
            "## Feature Columns Used",
            "```",
            ", ".join(self.feature_columns),
            "```",
            "",
            "## Architecture",
            "```",
            "Input(n_features, 1)",
            "  -> Conv1D(64, kernel=3) -> BatchNorm -> ReLU -> MaxPooling1D(2)",
            "  -> Conv1D(128, kernel=3) -> BatchNorm -> ReLU",
            "  -> GlobalAveragePooling1D",
            "  -> Dense(128, relu) -> Dropout(0.3)",
            "  -> Dense(64, relu)   [Spatial Embedding Output]",
            "  -> Dense(1, linear)  [Auxiliary head, training only]",
            "```",
            "",
            "## Training Configuration",
            f"- Optimizer: Adam (learning_rate={self.config.learning_rate})",
            "- Loss: MeanSquaredError",
            "- Metric: MAE",
            f"- Epochs (max): {self.config.epochs}",
            f"- Epochs run (early stopping): {final_epoch}",
            f"- Batch size: {self.config.batch_size}",
            "",
            "## Results",
            f"- Final training loss (MSE): {final_train_loss:.4f}",
            f"- Final validation loss (MSE): {final_val_loss:.4f}",
            f"- Best validation loss (MSE): {best_val_loss:.4f}",
            f"- Test loss (MSE): {test_loss:.4f}",
            f"- Test MAE: {test_mae:.4f}",
            "",
            "## Embedding",
            f"- Embedding dimension: {self.config.embedding_dim}",
            "",
            "## Artifacts",
            f"- Trained model: `{model_path}`",
            f"- Loss curve: `{self.config.results_dir / self.config.loss_curve_filename}`",
            "",
        ]

        report_path = self.config.results_dir / self.config.report_filename
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        logger.info("Training report written to '%s'", report_path)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def summary(self) -> None:
        """Print model summaries (trainer + feature extractor) via logger."""
        if self.training_model is None:
            logger.warning("Model not built yet. Call build_model() first.")
            return

        logger.info("=== Trainer model summary ===")
        self.training_model.summary(print_fn=logger.info)
        logger.info("=== Feature extractor sub-model summary ===")
        self.feature_extractor.summary(print_fn=logger.info)

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def run(self) -> np.ndarray:
        """Execute the full pipeline end-to-end.

        Steps: load_dataset -> prepare_spatial_input -> build_model ->
        train -> save_model -> extract_features -> summary.

        Returns
        -------
        np.ndarray
            The learned embeddings for the full dataset
            (train + val + test), ready for consumption by
            ``hybrid_model.py``.
        """
        try:
            self.load_dataset()
            self.prepare_spatial_input()
            self.build_model()
            self.train()
            self.save_model()
            embeddings = self.extract_features()
            self.summary()
            logger.info("CrimeCNN pipeline completed successfully.")
            return embeddings
        except Exception:
            logger.exception("CrimeCNN pipeline failed.")
            raise


# --------------------------------------------------------------------------- #
# Script entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cnn = CrimeCNN()
    learned_embeddings = cnn.run()
    logger.info("Final embedding matrix shape: %s", str(learned_embeddings.shape))