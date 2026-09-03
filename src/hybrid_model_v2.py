"""
=========================================================
Project : A Hybrid CNN-LSTM-GCN Framework for Spatio-Temporal Crime Prediction
Module  : 3-Way Hybrid Fusion Model (CNN + LSTM + GCN)
File    : src/hybrid_model_v2.py

Description:
Fuses learned representations from all three pretrained feature branches:
  1. CNN Spatial Branch (64-dim)
  2. LSTM Temporal Branch (32-dim)
  3. GCN Spatial-Adjacency Branch (32-dim)
Total fused embedding dimension = 128 (64 + 32 + 32).

Trains a fusion network (Dense 128 -> Dense 64 -> Dense 32 -> Dense 1)
on top of the concatenated 128-dimensional representations and saves
the final model to models/hybrid_gcn_model.keras without modifying any
existing model or service files.

Author: B.Tech Final Year Project
=========================================================
"""

from __future__ import annotations

import logging
import os
import random
import sys
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

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, callbacks, initializers, layers, optimizers, regularizers

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gcn_model import GraphConvLayer
from src.graph_utils import load_adjacency_artifacts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("HybridV2")

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_PATH = PROJECT_ROOT / "dataset" / "Crimes_in_india_2001-2013_features.csv"
ADJACENCY_PATH = PROJECT_ROOT / "artifacts" / "adjacency.pkl"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

CNN_MODEL_PATH = MODELS_DIR / "cnn_feature_extractor.keras"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.keras"
GCN_MODEL_PATH = MODELS_DIR / "gcn_feature_extractor.keras"
GCN_FULL_MODEL_PATH = MODELS_DIR / "gcn_model.keras"

CNN_EMBEDDING_LAYER = "embedding_bn"      # 64-dim spatial embedding
LSTM_EMBEDDING_LAYER = "lstm_embedding"    # 32-dim temporal embedding
GCN_EMBEDDING_LAYER = "gcn_embedding_bn"   # 32-dim spatial-graph embedding

TARGET_COLUMN = "TOTAL IPC CRIMES"
GROUP_COLUMNS = ["STATE_ENCODED", "DISTRICT_ENCODED"]
YEAR_ORDER_COLUMN = "YEAR_INDEX"
SEQUENCE_LENGTH = 3  # 3 years history to predict next year

CNN_EXCLUDED_COLUMNS = [TARGET_COLUMN, "YEAR", "YEAR_INDEX", "Id", "ID", "id"]
LSTM_EXCLUDED_COLUMNS = ["Id", "STATE/UT", "DISTRICT", "YEAR", TARGET_COLUMN]

FUSION_INPUT_DIM = 128  # 64 (CNN) + 32 (LSTM) + 32 (GCN)
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 1e-3
L2_LAMBDA = 1e-4

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ----------------------------------------------------------------------
# Data Loading & Feature Columns
# ----------------------------------------------------------------------
def load_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Loads engineered dataset sorted by district and year."""
    logger.info("Loading dataset from '%s'...", data_path)
    df = pd.read_csv(data_path)
    df = df.sort_values(by=GROUP_COLUMNS + [YEAR_ORDER_COLUMN]).reset_index(drop=True)
    logger.info("Dataset loaded: %d rows, %d columns.", df.shape[0], df.shape[1])
    return df


def _cnn_feature_columns(df: pd.DataFrame) -> list:
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    return [c for c in numeric_columns if c not in CNN_EXCLUDED_COLUMNS]


def _lstm_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in LSTM_EXCLUDED_COLUMNS]


# ----------------------------------------------------------------------
# Feature Extractors Loader
# ----------------------------------------------------------------------
def create_cnn_feature_extractor(model_path: Path = CNN_MODEL_PATH) -> Model:
    """Loads pretrained CNN and wraps in a frozen 64-dim spatial extractor."""
    logger.info("Loading pretrained CNN from '%s'...", model_path)
    cnn_model = tf.keras.models.load_model(model_path)
    cnn_model.trainable = False

    embedding_layer = cnn_model.get_layer(CNN_EMBEDDING_LAYER)
    extractor = Model(inputs=cnn_model.input, outputs=embedding_layer.output, name="cnn_feature_extractor")
    extractor.trainable = False
    logger.info("CNN feature extractor ready (output dim: %d).", embedding_layer.output_shape[-1])
    return extractor


def create_lstm_feature_extractor(model_path: Path = LSTM_MODEL_PATH) -> Model:
    """Loads pretrained LSTM and wraps in a frozen 32-dim temporal extractor."""
    logger.info("Loading pretrained LSTM from '%s'...", model_path)
    lstm_model = tf.keras.models.load_model(model_path)
    lstm_model.trainable = False

    embedding_layer = lstm_model.get_layer(LSTM_EMBEDDING_LAYER)
    extractor = Model(inputs=lstm_model.input, outputs=embedding_layer.output, name="lstm_feature_extractor")
    extractor.trainable = False
    logger.info("LSTM feature extractor ready (output dim: %d).", embedding_layer.output_shape[-1])
    return extractor


def create_gcn_feature_extractor(
    extractor_path: Path = GCN_MODEL_PATH, full_model_path: Path = GCN_FULL_MODEL_PATH
) -> Model:
    """Loads pretrained GCN feature extractor."""
    logger.info("Loading pretrained GCN from '%s'...", extractor_path)
    custom_objects = {"GraphConvLayer": GraphConvLayer}
    try:
        gcn_model = tf.keras.models.load_model(extractor_path, custom_objects=custom_objects)
    except Exception:
        logger.info("Loading from full GCN model '%s'...", full_model_path)
        full_gcn = tf.keras.models.load_model(full_model_path, custom_objects=custom_objects)
        embedding_layer = full_gcn.get_layer(GCN_EMBEDDING_LAYER)
        gcn_model = Model(inputs=full_gcn.input, outputs=embedding_layer.output, name="gcn_feature_extractor")

    gcn_model.trainable = False
    logger.info("GCN feature extractor ready (output shape: %s).", gcn_model.output_shape)
    return gcn_model


# ----------------------------------------------------------------------
# Temporal Sequences & Sample Alignment
# ----------------------------------------------------------------------
def _build_aligned_sequences(
    df: pd.DataFrame, lstm_feature_cols: list, sequence_length: int = SEQUENCE_LENGTH
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds aligned temporal sliding windows and tracks:
      - X_lstm: 3-year feature windows
      - y: Target crime counts
      - target_row_positions: row indices in df of target year
      - target_years: year index of target year
      - target_state_dist_pairs: (state_enc, dist_enc) of target year
    """
    sequences, targets, target_row_positions, target_years, state_dist_pairs = [], [], [], [], []

    for _, group in df.groupby(GROUP_COLUMNS):
        group = group.sort_values(YEAR_ORDER_COLUMN)
        feature_matrix = group[lstm_feature_cols].to_numpy(dtype=np.float32)
        target_vector = group[TARGET_COLUMN].to_numpy(dtype=np.float32)
        row_positions = group.index.to_numpy()
        year_indices = group[YEAR_ORDER_COLUMN].to_numpy(dtype=int)
        state_enc = float(group["STATE_ENCODED"].iloc[0])
        dist_enc = float(group["DISTRICT_ENCODED"].iloc[0])

        if len(group) <= sequence_length:
            continue

        for start in range(len(group) - sequence_length):
            end = start + sequence_length
            sequences.append(feature_matrix[start:end])
            targets.append(target_vector[end])
            target_row_positions.append(row_positions[end])
            target_years.append(year_indices[end])
            state_dist_pairs.append((state_enc, dist_enc))

    return (
        np.array(sequences, dtype=np.float32),
        np.array(targets, dtype=np.float32),
        np.array(target_row_positions, dtype=np.int64),
        np.array(target_years, dtype=np.int32),
        state_dist_pairs,
    )


# ----------------------------------------------------------------------
# 3-Way Feature Extraction & Concatenation (128-dim)
# ----------------------------------------------------------------------
def extract_3way_features(
    df: pd.DataFrame,
    cnn_extractor: Model,
    lstm_extractor: Model,
    gcn_extractor: Model,
    graph_data: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts and fuses all three embeddings:
        [CNN Spatial (64) | LSTM Temporal (32) | GCN Spatial-Adjacency (32)] = 128-dim
    """
    lstm_cols = _lstm_feature_columns(df)
    cnn_cols = _cnn_feature_columns(df)

    logger.info("Building aligned sequence windows...")
    X_lstm, y, target_positions, target_years, state_dist_pairs = _build_aligned_sequences(
        df, lstm_cols, SEQUENCE_LENGTH
    )

    # 1. Extract LSTM Temporal Embeddings (32-dim)
    logger.info("Extracting LSTM temporal embeddings (32-dim)...")
    lstm_embeddings = lstm_extractor.predict(X_lstm, verbose=0)

    # 2. Extract CNN Spatial Embeddings (64-dim)
    logger.info("Extracting CNN spatial embeddings (64-dim)...")
    cnn_feature_matrix = df[cnn_cols].to_numpy(dtype=np.float32)
    cnn_feature_matrix = cnn_feature_matrix.reshape((cnn_feature_matrix.shape[0], -1, 1))
    X_cnn = cnn_feature_matrix[target_positions]
    cnn_embeddings = cnn_extractor.predict(X_cnn, verbose=0)

    # 3. Extract GCN Spatial-Adjacency Embeddings (32-dim)
    logger.info("Extracting GCN spatial-adjacency embeddings (32-dim)...")
    # Precompute all 13 annual graph feature matrices
    unique_years = sorted(df["YEAR_INDEX"].unique())
    num_nodes = len(graph_data["node_df"])
    feature_cols = graph_data["feature_columns"]
    node_to_idx = graph_data["node_to_idx"]
    enc_pair_to_idx = graph_data["enc_pair_to_idx"]

    annual_node_features = np.zeros((len(unique_years), num_nodes, len(feature_cols)), dtype=np.float32)
    for y_idx in unique_years:
        year_df = df[df["YEAR_INDEX"] == y_idx]
        for _, row in year_df.iterrows():
            st = str(row["STATE/UT"]).strip().upper()
            dt = str(row["DISTRICT"]).strip().upper()
            n_idx = node_to_idx.get((st, dt))
            if n_idx is not None:
                annual_node_features[y_idx, n_idx] = row[feature_cols].to_numpy(dtype=np.float32)

    # Run GCN on all 13 annual graphs -> shape (13, 850, 32)
    annual_gcn_embeddings = gcn_extractor.predict(annual_node_features, verbose=0)

    # Align each sequence sample to its corresponding (target_year, target_node)
    num_samples = len(target_years)
    gcn_embeddings = np.zeros((num_samples, 32), dtype=np.float32)

    for i in range(num_samples):
        y_idx = target_years[i]
        st_enc, dt_enc = state_dist_pairs[i]
        n_idx = enc_pair_to_idx.get((st_enc, dt_enc))
        if n_idx is not None:
            gcn_embeddings[i] = annual_gcn_embeddings[y_idx, n_idx]

    # 4. Concatenate all three branches: 64 + 32 + 32 = 128 dimensions
    fused_features = np.concatenate([cnn_embeddings, lstm_embeddings, gcn_embeddings], axis=1)
    logger.info(
        "3-Way Fused Feature Matrix built: %d samples, %d dimensions (%d CNN + %d LSTM + %d GCN).",
        fused_features.shape[0],
        fused_features.shape[1],
        cnn_embeddings.shape[1],
        lstm_embeddings.shape[1],
        gcn_embeddings.shape[1],
    )
    return fused_features, y


# ----------------------------------------------------------------------
# 3-Way Hybrid Model Architecture
# ----------------------------------------------------------------------
def build_hybrid_gcn_model(input_dim: int = FUSION_INPUT_DIM) -> Model:
    """
    Constructs the 3-Way Hybrid Fusion Network:
        Input(128)
          -> Dense(128, HeNormal, L2) -> BatchNorm -> ReLU -> Dropout(0.3)
          -> Dense(64, HeNormal, L2)  -> BatchNorm -> ReLU -> Dropout(0.3)
          -> Dense(32, HeNormal, L2, "hybrid_gcn_embedding")
          -> Dense(1, Linear, "crime_count_output")
    """
    logger.info("Building 3-Way Hybrid CNN-LSTM-GCN Model (input_dim=%d)...", input_dim)
    he_init = initializers.HeNormal(seed=RANDOM_SEED)
    l2_reg = regularizers.l2(L2_LAMBDA)

    inputs = layers.Input(shape=(input_dim,), name="fused_3way_embedding_input")

    x = layers.Dense(128, kernel_initializer=he_init, kernel_regularizer=l2_reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)

    embedding = layers.Dense(
        32, kernel_initializer=he_init, kernel_regularizer=l2_reg, name="hybrid_gcn_embedding"
    )(x)

    outputs = layers.Dense(1, name="crime_count_output")(embedding)

    model = Model(inputs=inputs, outputs=outputs, name="hybrid_cnn_lstm_gcn_model")
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    model.summary(print_fn=logger.info)
    logger.info("Total Parameters: %s", model.count_params())
    return model


# ----------------------------------------------------------------------
# Training & Evaluation
# ----------------------------------------------------------------------
def train_hybrid_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    checkpoint_path: Path,
) -> tf.keras.callbacks.History:
    """Trains the 3-way fusion network with identical callbacks and hyperparams."""
    logger.info("Training Hybrid CNN-LSTM-GCN Model...")
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    checkpoint = callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path), monitor="val_loss", save_best_only=True, save_weights_only=False
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, checkpoint, reduce_lr],
        verbose=2,
    )
    return history


def evaluate_hybrid_model(
    model: Model,
    history: tf.keras.callbacks.History,
    X_test: np.ndarray,
    y_test: np.ndarray,
    results_dir: Path = RESULTS_DIR,
) -> Dict[str, float]:
    """Computes test metrics (MAE, RMSE, R²) and saves result artifacts."""
    logger.info("Evaluating 3-Way Hybrid CNN-LSTM-GCN Model...")
    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]

    predictions = model.predict(X_test, verbose=0).flatten()
    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))

    logger.info("=" * 60)
    logger.info("HYBRID CNN-LSTM-GCN EVALUATION RESULTS:")
    logger.info("Final Training Loss   : %.4f", final_train_loss)
    logger.info("Final Validation Loss : %.4f", final_val_loss)
    logger.info("Test MAE              : %.4f", mae)
    logger.info("Test RMSE             : %.4f", rmse)
    logger.info("Test R\u00b2               : %.4f (%.2f%%)", r2, r2 * 100)
    logger.info("=" * 60)

    results_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    predictions_path = results_dir / "hybrid_gcn_predictions.csv"
    pd.DataFrame({"Actual": y_test, "Predicted": predictions}).to_csv(predictions_path, index=False)

    # Save training history
    history_path = results_dir / "hybrid_gcn_training_history.csv"
    pd.DataFrame(history.history).to_csv(history_path, index=False)

    # Save Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Hybrid CNN-LSTM-GCN - Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "hybrid_gcn_loss_vs_epoch.png")
    plt.close()

    # Save MAE Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["mae"], label="Training MAE")
    plt.plot(history.history["val_mae"], label="Validation MAE")
    plt.title("Hybrid CNN-LSTM-GCN - MAE vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "hybrid_gcn_mae_vs_epoch.png")
    plt.close()

    # Save Scatter Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, predictions, alpha=0.5, edgecolors="k", linewidths=0.3, color="teal")
    lims = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
    plt.plot(lims, lims, "r--", label="Ideal (y = x)")
    plt.title("Hybrid CNN-LSTM-GCN - Actual vs Predicted Total IPC Crimes")
    plt.xlabel("Actual Crime Count")
    plt.ylabel("Predicted Crime Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "hybrid_gcn_actual_vs_predicted.png")
    plt.close()

    return {
        "train_loss": final_train_loss,
        "val_loss": final_val_loss,
        "test_mae": mae,
        "test_rmse": rmse,
        "test_r2": r2,
    }


def save_hybrid_models(model: Model, models_dir: Path = MODELS_DIR) -> Tuple[Path, Path]:
    """Saves the 3-way hybrid model and its standalone embedding extractor."""
    logger.info("Saving Hybrid CNN-LSTM-GCN Model...")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "hybrid_gcn_model.keras"
    extractor_path = models_dir / "hybrid_gcn_feature_extractor.keras"

    model.save(model_path)

    embedding_layer = model.get_layer("hybrid_gcn_embedding")
    feature_extractor = Model(inputs=model.input, outputs=embedding_layer.output, name="hybrid_gcn_feature_extractor")
    feature_extractor.save(extractor_path)

    logger.info("[Saved] Hybrid GCN Model -> %s", model_path)
    logger.info("[Saved] Hybrid GCN Feature Extractor -> %s", extractor_path)
    return model_path, extractor_path


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def main() -> Dict[str, float]:
    """Main execution entry point."""
    df = load_data(DATA_PATH)
    graph_data = load_adjacency_artifacts(str(ADJACENCY_PATH))

    cnn_extractor = create_cnn_feature_extractor(CNN_MODEL_PATH)
    lstm_extractor = create_lstm_feature_extractor(LSTM_MODEL_PATH)
    gcn_extractor = create_gcn_feature_extractor(GCN_MODEL_PATH, GCN_FULL_MODEL_PATH)

    fused_features, y = extract_3way_features(df, cnn_extractor, lstm_extractor, gcn_extractor, graph_data)

    # 70% train, 15% validation, 15% test with random_state=42 (identical split)
    X_train, X_temp, y_train, y_temp = train_test_split(
        fused_features, y, test_size=0.3, random_state=RANDOM_SEED, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, shuffle=True
    )
    logger.info("Split sizes -> train: %d, val: %d, test: %d", len(X_train), len(X_val), len(X_test))

    model = build_hybrid_gcn_model(input_dim=fused_features.shape[1])

    checkpoint_path = MODELS_DIR / "hybrid_gcn_model_checkpoint.keras"
    history = train_hybrid_model(model, X_train, y_train, X_val, y_val, checkpoint_path)

    metrics_dict = evaluate_hybrid_model(model, history, X_test, y_test, RESULTS_DIR)
    save_hybrid_models(model, MODELS_DIR)

    logger.info("Hybrid CNN-LSTM-GCN pipeline completed successfully.")
    return metrics_dict


if __name__ == "__main__":
    main()
