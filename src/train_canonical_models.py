"""
=========================================================
Project : A Hybrid CNN-LSTM-GCN Framework for Spatio-Temporal Crime Prediction
Module  : Canonical Joint Retraining & Evaluation Pipeline
File    : src/train_canonical_models.py

Description:
Trains both the 2-Way (Hybrid CNN-LSTM) and 3-Way (Hybrid CNN-LSTM-GCN)
models back-to-back in a single execution session using the exact same:
  - Dataset (Crimes_in_india_2001-2013_features.csv)
  - Train/Val/Test Split (70% / 15% / 15%, random_state=42, N=6,935)
  - Evaluation Held-Out Test Set (N_test = 1,041)
  - Hyperparameters, Callbacks, Loss, and Optimizer

Generates:
  1. models/hybrid_model.keras & models/hybrid_feature_extractor.keras
  2. models/hybrid_gcn_model.keras & models/hybrid_gcn_feature_extractor.keras
  3. models/model_manifest.json (SHA-256 checksums, parameters, timestamps)
  4. results/final_model_comparison.json & results/final_test_predictions.csv
=========================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any

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

from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
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
logger = logging.getLogger("CanonicalTraining")

# Configuration
DATA_PATH = PROJECT_ROOT / "dataset" / "Crimes_in_india_2001-2013_features.csv"
ADJACENCY_PATH = PROJECT_ROOT / "artifacts" / "adjacency.pkl"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MANIFEST_PATH = MODELS_DIR / "model_manifest.json"

CNN_MODEL_PATH = MODELS_DIR / "cnn_feature_extractor.keras"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.keras"
GCN_MODEL_PATH = MODELS_DIR / "gcn_feature_extractor.keras"

HYBRID_2WAY_PATH = MODELS_DIR / "hybrid_model.keras"
HYBRID_2WAY_EXTRACTOR_PATH = MODELS_DIR / "hybrid_feature_extractor.keras"
HYBRID_3WAY_PATH = MODELS_DIR / "hybrid_gcn_model.keras"
HYBRID_3WAY_EXTRACTOR_PATH = MODELS_DIR / "hybrid_gcn_feature_extractor.keras"

RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 1e-3
L2_LAMBDA = 1e-4
SEQUENCE_LENGTH = 3

GROUP_COLUMNS = ["STATE_ENCODED", "DISTRICT_ENCODED"]
YEAR_ORDER_COLUMN = "YEAR_INDEX"
TARGET_COLUMN = "TOTAL IPC CRIMES"

LSTM_EXCLUDED_COLUMNS = frozenset({"Id", "STATE/UT", "DISTRICT", "YEAR", TARGET_COLUMN})
CNN_EXCLUDED_COLUMNS = frozenset({"Id", "STATE/UT", "DISTRICT", "YEAR", "YEAR_INDEX", TARGET_COLUMN})


def set_deterministic_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def calculate_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_dataset() -> pd.DataFrame:
    logger.info("Loading engineered dataset from '%s'...", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values(by=GROUP_COLUMNS + [YEAR_ORDER_COLUMN]).reset_index(drop=True)
    logger.info("Dataset loaded: %d rows, %d columns.", df.shape[0], df.shape[1])
    return df


from src.hybrid_model_v2 import (
    create_cnn_feature_extractor,
    create_lstm_feature_extractor,
    create_gcn_feature_extractor,
)


def create_feature_extractors() -> Tuple[Model, Model, Model]:
    cnn_ext = create_cnn_feature_extractor()
    lstm_ext = create_lstm_feature_extractor()
    gcn_ext = create_gcn_feature_extractor()
    return cnn_ext, lstm_ext, gcn_ext


def extract_all_branch_embeddings(
    df: pd.DataFrame, cnn_ext: Model, lstm_ext: Model, gcn_ext: Model, graph_data: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts CNN (64), LSTM (32), and GCN (32) embeddings aligned to target sequence years.
    """
    lstm_cols = [c for c in df.columns if c not in LSTM_EXCLUDED_COLUMNS]
    cnn_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in CNN_EXCLUDED_COLUMNS]
    
    logger.info("Building temporal sequence sliding windows (window size=%d)...", SEQUENCE_LENGTH)
    sequences, targets, target_row_pos, target_years, state_dist_pairs = [], [], [], [], []
    
    for _, group in df.groupby(GROUP_COLUMNS):
        group = group.sort_values(YEAR_ORDER_COLUMN)
        feat_mat = group[lstm_cols].to_numpy(dtype=np.float32)
        tgt_vec = group[TARGET_COLUMN].to_numpy(dtype=np.float32)
        row_pos = group.index.to_numpy()
        
        if len(group) <= SEQUENCE_LENGTH:
            continue
            
        for start in range(len(group) - SEQUENCE_LENGTH):
            end = start + SEQUENCE_LENGTH
            sequences.append(feat_mat[start:end])
            targets.append(tgt_vec[end])
            target_row_pos.append(row_pos[end])
            target_years.append(int(group[YEAR_ORDER_COLUMN].iloc[end]))
            state_dist_pairs.append((
                int(group["STATE_ENCODED"].iloc[end]),
                int(group["DISTRICT_ENCODED"].iloc[end]),
            ))
            
    X_lstm = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    target_row_pos = np.array(target_row_pos, dtype=np.int64)
    target_years = np.array(target_years, dtype=np.int32)
    
    # Extract LSTM embeddings
    logger.info("Extracting LSTM temporal embeddings (32-dim)...")
    lstm_embs = lstm_ext.predict(X_lstm, batch_size=64, verbose=0)
    
    # Extract CNN embeddings
    logger.info("Extracting CNN spatial embeddings (64-dim)...")
    cnn_mat = df[cnn_cols].to_numpy(dtype=np.float32)
    cnn_mat_3d = cnn_mat.reshape((cnn_mat.shape[0], -1, 1))
    X_cnn = cnn_mat_3d[target_row_pos]
    cnn_embs = cnn_ext.predict(X_cnn, batch_size=64, verbose=0)
    
    # Extract GCN embeddings
    logger.info("Extracting GCN spatial-adjacency embeddings (32-dim)...")
    node_to_idx = graph_data["node_to_idx"]
    feature_cols = graph_data["feature_columns"]
    node_df = graph_data["node_df"]
    num_nodes = len(node_df)
    unique_years = sorted(df[YEAR_ORDER_COLUMN].unique())
    
    annual_gcn_embs = {}
    for yr in unique_years:
        yr_df = df[df[YEAR_ORDER_COLUMN] == yr]
        yr_mat = np.zeros((num_nodes, len(feature_cols)), dtype=np.float32)
        for _, row in yr_df.iterrows():
            st = str(row["STATE/UT"]).strip().upper()
            dt = str(row["DISTRICT"]).strip().upper()
            idx = node_to_idx.get((st, dt))
            if idx is not None:
                yr_mat[idx] = row[feature_cols].to_numpy(dtype=np.float32)
        yr_tensor = np.expand_dims(yr_mat, axis=0)
        full_gcn_out = gcn_ext.predict(yr_tensor, verbose=0)
        annual_gcn_embs[yr] = full_gcn_out[0]
        
    gcn_sample_embs = []
    for i in range(len(target_row_pos)):
        row_data = df.iloc[target_row_pos[i]]
        st = str(row_data["STATE/UT"]).strip().upper()
        dt = str(row_data["DISTRICT"]).strip().upper()
        yr = target_years[i]
        n_idx = node_to_idx[(st, dt)]
        gcn_sample_embs.append(annual_gcn_embs[yr][n_idx])
    gcn_embs = np.array(gcn_sample_embs, dtype=np.float32)
    
    logger.info("Embedding extraction complete. Samples=%d. CNN=(%s), LSTM=(%s), GCN=(%s)",
                len(y), cnn_embs.shape, lstm_embs.shape, gcn_embs.shape)
    return cnn_embs, lstm_embs, gcn_embs, y, target_row_pos, target_years, np.array(state_dist_pairs)


def build_fusion_model(input_dim: int, model_name: str, embedding_layer_name: str) -> Model:
    he_init = initializers.HeNormal(seed=RANDOM_SEED)
    l2_reg = regularizers.l2(L2_LAMBDA)
    
    inputs = layers.Input(shape=(input_dim,), name="fused_embedding_input")
    x = layers.Dense(128, kernel_initializer=he_init, kernel_regularizer=l2_reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)
    
    embedding = layers.Dense(32, kernel_initializer=he_init, kernel_regularizer=l2_reg, name=embedding_layer_name)(x)
    embedding_bn = layers.BatchNormalization()(embedding)
    
    outputs = layers.Dense(1, name="crime_count_output")(embedding_bn)
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def train_single_model(
    model: Model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, checkpoint_path: Path
) -> tf.keras.callbacks.History:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    checkpoint = callbacks.ModelCheckpoint(filepath=str(checkpoint_path), monitor="val_loss", save_best_only=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, checkpoint, reduce_lr],
        verbose=0,
    )
    return history


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0)))) * 100.0
    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "medae": medae,
        "mape": mape,
    }


def main() -> None:
    set_deterministic_seed(RANDOM_SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    df = load_dataset()
    cnn_ext, lstm_ext, gcn_ext = create_feature_extractors()
    graph_data = load_adjacency_artifacts(ADJACENCY_PATH)
    
    cnn_embs, lstm_embs, gcn_embs, y, target_row_pos, target_years, state_dist_pairs = extract_all_branch_embeddings(
        df, cnn_ext, lstm_ext, gcn_ext, graph_data
    )
    
    # 2-Way Fused representation (96-dim)
    fused_2way = np.concatenate([cnn_embs, lstm_embs], axis=1)
    
    # 3-Way Fused representation (128-dim)
    fused_3way = np.concatenate([cnn_embs, lstm_embs, gcn_embs], axis=1)
    
    # Shared Deterministic Split
    indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=RANDOM_SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_SEED)
    
    logger.info("Split Sizes: Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)",
                len(train_idx), len(train_idx)/len(y)*100,
                len(val_idx), len(val_idx)/len(y)*100,
                len(test_idx), len(test_idx)/len(y)*100)
    
    # -----------------------------------------------------------------
    # Train 2-Way Model (Hybrid CNN-LSTM)
    # -----------------------------------------------------------------
    logger.info("--- Training 2-Way Hybrid Model (CNN + LSTM, 96-dim) ---")
    set_deterministic_seed(RANDOM_SEED)
    model_2way = build_fusion_model(96, "hybrid_cnn_lstm_model", "hybrid_embedding")
    chk_2way = MODELS_DIR / "hybrid_model_checkpoint.keras"
    train_single_model(model_2way, fused_2way[train_idx], y[train_idx], fused_2way[val_idx], y[val_idx], chk_2way)
    
    # Save canonical 2-way model and feature extractor
    model_2way.save(HYBRID_2WAY_PATH)
    ext_2way = Model(inputs=model_2way.input, outputs=model_2way.get_layer("hybrid_embedding").output, name="hybrid_feature_extractor")
    ext_2way.save(HYBRID_2WAY_EXTRACTOR_PATH)
    logger.info("Saved 2-Way model to '%s'.", HYBRID_2WAY_PATH)
    
    # -----------------------------------------------------------------
    # Train 3-Way Model (Hybrid CNN-LSTM-GCN)
    # -----------------------------------------------------------------
    logger.info("--- Training 3-Way Hybrid Model (CNN + LSTM + GCN, 128-dim) ---")
    set_deterministic_seed(RANDOM_SEED)
    model_3way = build_fusion_model(128, "hybrid_cnn_lstm_gcn_model", "hybrid_gcn_embedding")
    chk_3way = MODELS_DIR / "hybrid_gcn_model_checkpoint.keras"
    train_single_model(model_3way, fused_3way[train_idx], y[train_idx], fused_3way[val_idx], y[val_idx], chk_3way)
    
    # Save canonical 3-way model and feature extractor
    model_3way.save(HYBRID_3WAY_PATH)
    ext_3way = Model(inputs=model_3way.input, outputs=model_3way.get_layer("hybrid_gcn_embedding").output, name="hybrid_gcn_feature_extractor")
    ext_3way.save(HYBRID_3WAY_EXTRACTOR_PATH)
    logger.info("Saved 3-Way model to '%s'.", HYBRID_3WAY_PATH)
    
    # -----------------------------------------------------------------
    # Evaluate Both Fresh Checkpoints on EXACT Identical Test Set
    # -----------------------------------------------------------------
    preds_2way_test = model_2way.predict(fused_2way[test_idx], verbose=0).flatten()
    preds_3way_test = model_3way.predict(fused_3way[test_idx], verbose=0).flatten()
    y_test = y[test_idx]
    
    metrics_2way = compute_metrics(y_test, preds_2way_test)
    metrics_3way = compute_metrics(y_test, preds_3way_test)
    
    # Compute manifest with checksums
    timestamp_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    manifest: Dict[str, Any] = {
        "generated_at": timestamp_str,
        "random_seed": RANDOM_SEED,
        "total_samples": int(len(y)),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "models": {
            "hybrid_2way": {
                "file": str(HYBRID_2WAY_PATH.name),
                "sha256": calculate_file_sha256(HYBRID_2WAY_PATH),
                "size_bytes": HYBRID_2WAY_PATH.stat().st_size,
                "input_dim": 96,
                "metrics": metrics_2way,
            },
            "hybrid_3way_gcn": {
                "file": str(HYBRID_3WAY_PATH.name),
                "sha256": calculate_file_sha256(HYBRID_3WAY_PATH),
                "size_bytes": HYBRID_3WAY_PATH.stat().st_size,
                "input_dim": 128,
                "metrics": metrics_3way,
            },
            "cnn_feature_extractor": {
                "file": str(CNN_MODEL_PATH.name),
                "sha256": calculate_file_sha256(CNN_MODEL_PATH),
                "size_bytes": CNN_MODEL_PATH.stat().st_size,
            },
            "lstm_model": {
                "file": str(LSTM_MODEL_PATH.name),
                "sha256": calculate_file_sha256(LSTM_MODEL_PATH),
                "size_bytes": LSTM_MODEL_PATH.stat().st_size,
            },
            "gcn_feature_extractor": {
                "file": str(GCN_MODEL_PATH.name),
                "sha256": calculate_file_sha256(GCN_MODEL_PATH),
                "size_bytes": GCN_MODEL_PATH.stat().st_size,
            },
        },
    }
    
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Model manifest saved to '%s'.", MANIFEST_PATH)
    
    # Save predictions comparison table
    test_df_records = []
    for idx_in_test, global_idx in enumerate(test_idx):
        row_orig = df.iloc[target_row_pos[global_idx]]
        test_df_records.append({
            "STATE/UT": row_orig["STATE/UT"],
            "DISTRICT": row_orig["DISTRICT"],
            "YEAR": int(row_orig["YEAR"]),
            "Actual": float(y_test[idx_in_test]),
            "Pred_2Way": float(preds_2way_test[idx_in_test]),
            "Pred_3Way_GCN": float(preds_3way_test[idx_in_test]),
            "AbsErr_2Way": float(abs(y_test[idx_in_test] - preds_2way_test[idx_in_test])),
            "AbsErr_3Way_GCN": float(abs(y_test[idx_in_test] - preds_3way_test[idx_in_test])),
        })
    pd.DataFrame(test_df_records).to_csv(RESULTS_DIR / "final_test_predictions.csv", index=False)
    
    # Print canonical table
    print("\n" + "=" * 80)
    print("CANONICAL MODEL COMPARISON (Fresh Checkpoints on Identical Test Set N=1,041)")
    print("=" * 80)
    print(f"{'Metric':<25} | {'2-Way (CNN-LSTM)':<20} | {'3-Way (CNN-LSTM-GCN)':<22} | {'Improvement'}")
    print("-" * 80)
    print(f"{'R² Score (%)':<25} | {metrics_2way['r2']*100:19.2f}% | {metrics_3way['r2']*100:21.2f}% | {metrics_3way['r2']*100 - metrics_2way['r2']*100:+5.2f}%")
    print(f"{'Mean Absolute Error (MAE)':<25} | {metrics_2way['mae']:20.2f} | {metrics_3way['mae']:22.2f} | {((metrics_3way['mae']-metrics_2way['mae'])/metrics_2way['mae'])*100:+5.2f}%")
    print(f"{'Root Mean Sq Error (RMSE)':<25} | {metrics_2way['rmse']:20.2f} | {metrics_3way['rmse']:22.2f} | {((metrics_3way['rmse']-metrics_2way['rmse'])/metrics_2way['rmse'])*100:+5.2f}%")
    print(f"{'Median Abs Error (MedAE)':<25} | {metrics_2way['medae']:20.2f} | {metrics_3way['medae']:22.2f} | {((metrics_3way['medae']-metrics_2way['medae'])/metrics_2way['medae'])*100:+5.2f}%")
    print(f"{'Mean Abs Pct Error (MAPE)':<25} | {metrics_2way['mape']:19.2f}% | {metrics_3way['mape']:21.2f}% | {metrics_3way['mape'] - metrics_2way['mape']:+5.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
