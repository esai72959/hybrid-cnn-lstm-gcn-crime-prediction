"""
=========================================================
Project : A Hybrid CNN-LSTM-GCN Framework for Spatio-Temporal Crime Prediction
Module  : Graph Convolutional Network (GCN) Spatial-Adjacency Branch
File    : src/gcn_model.py

Description:
Implements a spectral Graph Convolutional Network (GCN) branch in native
TensorFlow/Keras (Kipf-Welling formulation Z = A_hat * X * W + b).
Learns spatial crime propagation embeddings across the 850 composite district
nodes using the static normalized adjacency matrix from artifacts/adjacency.pkl.
Outputs a 32-dimensional spatial-graph embedding (matching LSTM embedding size)
to be fused in hybrid_model_v2.py.

Author: B.Tech Final Year Project
=========================================================
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")
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

from tensorflow.keras import Input, Model, layers, optimizers, losses, metrics
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.graph_utils import CrimeGraphBuilder, load_adjacency_artifacts
except ImportError:
    from graph_utils import CrimeGraphBuilder, load_adjacency_artifacts

# --------------------------------------------------------------------------- #
# Logging Configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CrimeGCN")


# --------------------------------------------------------------------------- #
# Custom Native Keras Spectral Graph Convolution Layer
# --------------------------------------------------------------------------- #
@tf.keras.utils.register_keras_serializable(package="CrimeGCN")
class GraphConvLayer(layers.Layer):
    """
    Spectral Graph Convolutional Layer implementing Kipf & Welling (ICLR 2017):
        Z = A_hat * X * W + b

    Supports both 2D input (N, F_in) and 3D batched input (batch, N, F_in).
    """

    def __init__(
        self,
        units: int,
        adj_matrix: Optional[np.ndarray] = None,
        kernel_initializer: str = "he_normal",
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        use_bias: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.units = int(units)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.use_bias = use_bias

        if adj_matrix is not None:
            self.adj_matrix = tf.constant(adj_matrix, dtype=tf.float32)
        else:
            try:
                graph_data = load_adjacency_artifacts()
                self.adj_matrix = tf.constant(graph_data["norm_adj_matrix"], dtype=tf.float32)
            except Exception:
                self.adj_matrix = None

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.kernel = self.add_weight(
            name="kernel",
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.bias = None
        super().build(input_shape)

    def call(self, inputs, adj: Optional[tf.Tensor] = None):
        """
        inputs: Tensor of shape (batch, N, F) or (N, F)
        adj: Optional normalized adjacency tensor (N, N). If None, uses internal self.adj_matrix.
        """
        a_hat = adj if adj is not None else self.adj_matrix
        if a_hat is None:
            raise ValueError("Adjacency matrix A_hat must be provided to GraphConvLayer.")

        # X * W
        xw = tf.matmul(inputs, self.kernel)

        # A_hat * (X * W)
        # If 3D batched input (batch, N, units), broadcast a_hat (N, N) across batch
        if len(inputs.shape) == 3:
            # tf.matmul broadcasts (N, N) against (batch, N, units)
            output = tf.matmul(tf.cast(a_hat, inputs.dtype), xw)
        else:
            output = tf.matmul(tf.cast(a_hat, inputs.dtype), xw)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias,
        })
        return config


# --------------------------------------------------------------------------- #
# Configuration Dataclass
# --------------------------------------------------------------------------- #
@dataclass
class GCNConfig:
    """Configuration container for CrimeGCN."""
    dataset_path: Path = Path("dataset/Crimes_in_india_2001-2013_features.csv")
    adjacency_path: Path = Path("artifacts/adjacency.pkl")
    model_dir: Path = Path("models")
    results_dir: Path = Path("results")
    model_filename: str = "gcn_model.keras"
    feature_extractor_filename: str = "gcn_feature_extractor.keras"
    report_filename: str = "gcn_training_report.md"
    loss_curve_filename: str = "gcn_loss_curve.png"

    embedding_dim: int = 32  # 32-dim to match LSTM branch embedding size
    epochs: int = 100
    learning_rate: float = 0.001
    l2_lambda: float = 1e-4
    random_state: int = 42


# --------------------------------------------------------------------------- #
# Main CrimeGCN Class
# --------------------------------------------------------------------------- #
class CrimeGCN:
    """
    Graph Convolutional Network spatial-adjacency branch.

    Learns a 32-dimensional spatial propagation embedding per district node
    over annual feature snapshots, utilizing the static normalized adjacency
    graph A_hat (850 x 850).
    """

    def __init__(self, config: Optional[GCNConfig] = None) -> None:
        self.config = config or GCNConfig()
        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        self.config.results_dir.mkdir(parents=True, exist_ok=True)

        self.graph_data: Optional[Dict[str, Any]] = None
        self.norm_adj: Optional[np.ndarray] = None
        self.node_df: Optional[pd.DataFrame] = None
        self.num_nodes: int = 850
        self.num_features: int = 33

        self.df: Optional[pd.DataFrame] = None
        self.annual_features: Optional[np.ndarray] = None  # (13, 850, 33)
        self.annual_targets: Optional[np.ndarray] = None   # (13, 850, 1)

        self.training_model: Optional[Model] = None
        self.feature_extractor: Optional[Model] = None
        self.history: Optional[tf.keras.callbacks.History] = None

    # ------------------------------------------------------------------ #
    # Step 1: Load Graph & Dataset
    # ------------------------------------------------------------------ #
    def load_data(self) -> None:
        """Loads static adjacency artifacts and prepares 13 annual snapshots."""
        logger.info("Loading graph artifacts from '%s'...", self.config.adjacency_path)
        if not self.config.adjacency_path.exists():
            builder = CrimeGraphBuilder(k_neighbors=5)
            builder.run()

        self.graph_data = load_adjacency_artifacts(str(self.config.adjacency_path))
        self.norm_adj = self.graph_data["norm_adj_matrix"].astype(np.float32)
        self.node_df = self.graph_data["node_df"]
        self.num_nodes = self.norm_adj.shape[0]

        logger.info("Loading features dataset from '%s'...", self.config.dataset_path)
        self.df = pd.read_csv(self.config.dataset_path)

        feature_cols = self.graph_data["feature_columns"]
        self.num_features = len(feature_cols)

        # Build (13, 850, 33) annual node feature snapshots and (13, 850, 1) targets
        unique_years = sorted(self.df["YEAR_INDEX"].unique())
        num_years = len(unique_years)

        x_annual = np.zeros((num_years, self.num_nodes, self.num_features), dtype=np.float32)
        y_annual = np.zeros((num_years, self.num_nodes, 1), dtype=np.float32)

        node_to_idx = self.graph_data["node_to_idx"]

        for y_idx in unique_years:
            year_df = self.df[self.df["YEAR_INDEX"] == y_idx]
            for _, row in year_df.iterrows():
                state = str(row["STATE/UT"]).strip().upper()
                district = str(row["DISTRICT"]).strip().upper()
                node_idx = node_to_idx.get((state, district))

                if node_idx is not None:
                    x_annual[y_idx, node_idx] = row[feature_cols].to_numpy(dtype=np.float32)
                    y_annual[y_idx, node_idx, 0] = float(row["TOTAL IPC CRIMES"])

        self.annual_features = x_annual
        self.annual_targets = y_annual

        logger.info(
            "Prepared %d annual graph snapshots: Feature tensor %s, Target tensor %s.",
            num_years,
            self.annual_features.shape,
            self.annual_targets.shape,
        )

    # ------------------------------------------------------------------ #
    # Step 2: Build GCN Architecture
    # ------------------------------------------------------------------ #
    def build_model(self) -> Model:
        """
        Constructs the spectral GCN model:
            Input(850, 33)
              -> GraphConvLayer(64, HeNormal, L2) -> BatchNorm -> ReLU -> Dropout(0.3)
              -> GraphConvLayer(32, HeNormal, L2) -> BatchNorm -> ReLU -> Dropout(0.3)
              -> Dense(32, HeNormal, L2, "gcn_embedding") -> BatchNorm
              -> Dense(1, Linear) [Auxiliary head for pretraining]
        """
        if self.norm_adj is None:
            self.load_data()

        he_init = initializers.HeNormal(seed=self.config.random_state)
        l2_reg = regularizers.l2(self.config.l2_lambda)

        inputs = Input(shape=(self.num_nodes, self.num_features), name="graph_node_features_input")

        # GCN Block 1
        x = GraphConvLayer(
            units=64,
            adj_matrix=self.norm_adj,
            kernel_initializer=he_init,
            kernel_regularizer=l2_reg,
            name="gcn_conv1",
        )(inputs)
        x = layers.BatchNormalization(name="gcn_bn1")(x)
        x = layers.ReLU(name="gcn_relu1")(x)
        x = layers.Dropout(0.3, name="gcn_drop1")(x)

        # GCN Block 2
        x = GraphConvLayer(
            units=32,
            adj_matrix=self.norm_adj,
            kernel_initializer=he_init,
            kernel_regularizer=l2_reg,
            name="gcn_conv2",
        )(inputs)
        x = layers.BatchNormalization(name="gcn_bn2")(x)
        x = layers.ReLU(name="gcn_relu2")(x)
        x = layers.Dropout(0.3, name="gcn_drop2")(x)

        # Node Embedding Layer (32-dim)
        embed_raw = layers.Dense(
            self.config.embedding_dim,
            activation="relu",
            kernel_initializer=he_init,
            kernel_regularizer=l2_reg,
            name="gcn_embedding",
        )(x)
        embed_bn = layers.BatchNormalization(name="gcn_embedding_bn")(embed_raw)

        # Auxiliary Crime Count Prediction Head (used only for pretraining)
        aux_pred = layers.Dense(1, activation="linear", name="gcn_aux_output")(embed_bn)

        self.training_model = Model(inputs=inputs, outputs=aux_pred, name="CrimeGCN_Trainer")
        self.feature_extractor = Model(inputs=inputs, outputs=embed_bn, name="CrimeGCN_FeatureExtractor")

        optimizer = optimizers.Adam(
            learning_rate=self.config.learning_rate, clipnorm=1.0
        )
        self.training_model.compile(
            optimizer=optimizer,
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanAbsoluteError(name="mae")],
        )

        logger.info("CrimeGCN model built successfully.")
        self.training_model.summary(print_fn=logger.info)
        return self.training_model

    # ------------------------------------------------------------------ #
    # Step 3: Auxiliary Pretraining
    # ------------------------------------------------------------------ #
    def train(self) -> tf.keras.callbacks.History:
        """
        Pretrains the GCN branch on annual graph snapshots to force the embedding
        layer to learn informative spatial propagation representations before fusion.
        """
        if self.training_model is None:
            self.build_model()

        # 10 years train (2001-2010), 3 years val (2011-2013)
        x_train, y_train = self.annual_features[:10], self.annual_targets[:10]
        x_val, y_val = self.annual_features[10:], self.annual_targets[10:]

        checkpoint_path = self.config.model_dir / "gcn_checkpoint_best.keras"

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1),
            ModelCheckpoint(filepath=str(checkpoint_path), monitor="val_loss", save_best_only=True, verbose=1),
        ]

        logger.info(
            "Starting auxiliary pretraining on %d annual snapshots for up to %d epochs...",
            len(x_train),
            self.config.epochs,
        )

        self.history = self.training_model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=self.config.epochs,
            batch_size=1,  # 1 full graph per step
            callbacks=callbacks,
            verbose=2,
        )

        logger.info("GCN auxiliary pretraining completed.")
        self._plot_loss_curve()
        self._write_report()
        return self.history

    # ------------------------------------------------------------------ #
    # Step 4: Feature Extraction for Hybrid Fusion
    # ------------------------------------------------------------------ #
    def extract_annual_embeddings(self) -> np.ndarray:
        """
        Runs feature extractor on all 13 annual snapshots.
        Returns:
            np.ndarray of shape (13, 850, 32)
        """
        if self.feature_extractor is None:
            self.build_model()

        embeddings = self.feature_extractor.predict(self.annual_features, verbose=0)
        logger.info("Extracted annual node embeddings with shape: %s", embeddings.shape)
        return embeddings

    # ------------------------------------------------------------------ #
    # Step 5: Save Models & Artifacts
    # ------------------------------------------------------------------ #
    def save_models(self) -> Tuple[Path, Path]:
        """Saves full GCN model and standalone feature extractor."""
        if self.training_model is None or self.feature_extractor is None:
            raise RuntimeError("Build and train model before saving.")

        model_path = self.config.model_dir / self.config.model_filename
        extractor_path = self.config.model_dir / self.config.feature_extractor_filename

        self.training_model.save(model_path)
        self.feature_extractor.save(extractor_path)

        logger.info("[Saved] GCN Trainer Model -> %s", model_path)
        logger.info("[Saved] GCN Feature Extractor -> %s", extractor_path)
        return model_path, extractor_path

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def _plot_loss_curve(self) -> None:
        if self.history is None:
            return
        plt.figure(figsize=(8, 5))
        plt.plot(self.history.history["loss"], label="Training Loss (MSE)")
        plt.plot(self.history.history["val_loss"], label="Validation Loss (MSE)")
        plt.title("GCN Auxiliary Pretraining Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        out_path = self.config.results_dir / self.config.loss_curve_filename
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info("Saved GCN loss curve to '%s'", out_path)

    def _write_report(self) -> None:
        if self.history is None:
            return
        final_loss = self.history.history["loss"][-1]
        final_val_loss = self.history.history["val_loss"][-1]
        best_val_loss = min(self.history.history["val_loss"])

        report = [
            "# CrimeGCN Branch Training Report",
            "",
            "## Architecture",
            "- Model Type: Native Spectral Graph Convolutional Network (Keras Layer)",
            "- Total Graph Nodes: 850 composite (STATE/UT, DISTRICT) nodes",
            "- Input Features: 33 engineered predictors per node",
            "- Output Embedding Dimension: 32 (named `gcn_embedding`)",
            "",
            "## Layer Pipeline",
            "```",
            "Input(850, 33)",
            "  -> GraphConvLayer(64, HeNormal, L2(1e-4)) -> BatchNorm -> ReLU -> Dropout(0.3)",
            "  -> GraphConvLayer(32, HeNormal, L2(1e-4)) -> BatchNorm -> ReLU -> Dropout(0.3)",
            "  -> Dense(32, ReLU) -> BatchNorm [gcn_embedding output: (850, 32)]",
            "  -> Dense(1, Linear) [Auxiliary head for pretraining]",
            "```",
            "",
            "## Auxiliary Pretraining Note",
            "- Note: The standalone GCN auxiliary head is trained on 13 annual full-graph snapshots (2001-2013).",
            "- Its purpose is auxiliary feature pretraining to condition the 32-dim spatial-graph embedding.",
            "- Final predictive accuracy is evaluated on the 3-way fused model in `hybrid_model_v2.py`.",
            "",
            "## Auxiliary Training Metrics",
            f"- Final Training Loss (MSE): {final_loss:.4f}",
            f"- Final Validation Loss (MSE): {final_val_loss:.4f}",
            f"- Best Validation Loss (MSE): {best_val_loss:.4f}",
            "",
            "## Artifacts",
            f"- GCN Model: `{self.config.model_dir / self.config.model_filename}`",
            f"- GCN Feature Extractor: `{self.config.model_dir / self.config.feature_extractor_filename}`",
            f"- Loss Curve: `{self.config.results_dir / self.config.loss_curve_filename}`",
            "",
        ]
        out_path = self.config.results_dir / self.config.report_filename
        out_path.write_text("\n".join(report), encoding="utf-8")
        logger.info("Saved GCN training report to '%s'", out_path)

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def run(self) -> np.ndarray:
        """Executes full GCN training and feature extraction pipeline."""
        self.load_data()
        self.build_model()
        self.train()
        self.save_models()
        return self.extract_annual_embeddings()


if __name__ == "__main__":
    gcn = CrimeGCN()
    annual_embeddings = gcn.run()
    logger.info("CrimeGCN pipeline completed successfully. Annual embeddings shape: %s", annual_embeddings.shape)
