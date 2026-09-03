"""
=========================================================
Project : A Hybrid CNN-LSTM-GCN Framework for Spatio-Temporal Crime Prediction
Module  : 5-Fold Cross-Validation Benchmark Pipeline
File    : src/cross_validate_models.py

Description:
Executes rigorous 5-Fold Cross-Validation (KFold, shuffle=True, random_state=42)
comparing the 2-Way Baseline (CNN + LSTM) vs 3-Way Model (CNN + LSTM + GCN)
on the complete 6,935-sample aligned dataset.

For each fold:
  - Retrains both models from scratch with identical hyperparameters & architectures
  - Evaluates on the held-out test fold
  - Records R^2, RMSE, MAE, MedAE, and MAPE
  - Preserves existing canonical production checkpoints (does NOT touch models/*.keras)

Saves:
  - results/cv_5fold_results.json
  - results/cv_5fold_summary.csv
=========================================================
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras import Model, callbacks, initializers, layers, optimizers, regularizers

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gcn_model import GraphConvLayer
from src.graph_utils import load_adjacency_artifacts
from src.hybrid_model_v2 import (
    create_cnn_feature_extractor,
    create_gcn_feature_extractor,
    create_lstm_feature_extractor,
    load_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CrossValidation")

# Paths & Settings
DATA_PATH = PROJECT_ROOT / "dataset" / "Crimes_in_india_2001-2013_features.csv"
ADJACENCY_PATH = PROJECT_ROOT / "artifacts" / "adjacency.pkl"
RESULTS_DIR = PROJECT_ROOT / "results"
CV_RESULTS_JSON = RESULTS_DIR / "cv_5fold_results.json"
CV_RESULTS_CSV = RESULTS_DIR / "cv_5fold_summary.csv"

N_SPLITS = 5
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 120
LEARNING_RATE = 1e-3
L2_LAMBDA = 1e-4
SEQUENCE_LENGTH = 3

GROUP_COLUMNS = ["STATE_ENCODED", "DISTRICT_ENCODED"]
YEAR_ORDER_COLUMN = "YEAR_INDEX"
TARGET_COLUMN = "TOTAL IPC CRIMES"

LSTM_EXCLUDED_COLUMNS = frozenset({"Id", "STATE/UT", "DISTRICT", "YEAR", TARGET_COLUMN})
CNN_EXCLUDED_COLUMNS = frozenset({"Id", "STATE/UT", "DISTRICT", "YEAR", "YEAR_INDEX", TARGET_COLUMN})


def set_seed(seed: int = RANDOM_SEED) -> None:
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


def extract_embeddings_all(
    df: pd.DataFrame, cnn_ext: Model, lstm_ext: Model, gcn_ext: Model, graph_data: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extracts CNN (64), LSTM (32), and GCN (32) embeddings for all aligned sequence samples."""
    lstm_cols = [c for c in df.columns if c not in LSTM_EXCLUDED_COLUMNS]
    cnn_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in CNN_EXCLUDED_COLUMNS]

    logger.info("Extracting aligned temporal sequence windows (N=%d)...", SEQUENCE_LENGTH)
    sequences, targets, target_row_pos, target_years = [], [], [], []

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

    X_lstm = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    target_row_pos = np.array(target_row_pos, dtype=np.int64)
    target_years = np.array(target_years, dtype=np.int32)

    logger.info("Predicting LSTM temporal embeddings (32-dim)...")
    lstm_embs = lstm_ext.predict(X_lstm, batch_size=64, verbose=0)

    logger.info("Predicting CNN spatial embeddings (64-dim)...")
    cnn_mat = df[cnn_cols].to_numpy(dtype=np.float32)
    cnn_mat_3d = cnn_mat.reshape((cnn_mat.shape[0], -1, 1))
    X_cnn = cnn_mat_3d[target_row_pos]
    cnn_embs = cnn_ext.predict(X_cnn, batch_size=64, verbose=0)

    logger.info("Predicting GCN spatial-adjacency embeddings (32-dim)...")
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

    logger.info("Embeddings extracted for all %d samples.", len(y))
    return cnn_embs, lstm_embs, gcn_embs, y


def build_model(input_dim: int, seed: int = RANDOM_SEED) -> Model:
    he_init = initializers.HeNormal(seed=seed)
    l2_reg = regularizers.l2(L2_LAMBDA)

    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, kernel_initializer=he_init, kernel_regularizer=l2_reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)

    embedding = layers.Dense(32, kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
    embedding_bn = layers.BatchNormalization()(embedding)

    outputs = layers.Dense(1)(embedding_bn)
    model = Model(inputs=inputs, outputs=outputs)

    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0)))) * 100.0
    smape = float(np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-5))) * 100.0
    medape = float(np.median(np.abs(y_true - y_pred) / np.maximum(y_true, 1.0))) * 100.0
    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "medae": medae,
        "mape": mape,
        "smape": smape,
        "medape": medape,
    }


def run_cross_validation() -> None:
    set_seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    cnn_ext = create_cnn_feature_extractor()
    lstm_ext = create_lstm_feature_extractor()
    gcn_ext = create_gcn_feature_extractor()
    graph_data = load_adjacency_artifacts(ADJACENCY_PATH)

    cnn_embs, lstm_embs, gcn_embs, y = extract_embeddings_all(df, cnn_ext, lstm_ext, gcn_ext, graph_data)

    fused_2way = np.concatenate([cnn_embs, lstm_embs], axis=1)  # 96-dim
    fused_3way = np.concatenate([cnn_embs, lstm_embs, gcn_embs], axis=1)  # 128-dim

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    fold_results_2way: List[Dict[str, float]] = []
    fold_results_3way: List[Dict[str, float]] = []

    logger.info("Starting %d-Fold Cross-Validation on %d samples...", N_SPLITS, len(y))

    for fold_num, (train_val_idx, test_idx) in enumerate(kf.split(fused_2way), start=1):
        fold_start = time.time()
        logger.info("=== FOLD %d / %d (TrainVal=%d, Test=%d) ===", fold_num, N_SPLITS, len(train_val_idx), len(test_idx))

        # Split train_val into train (85%) and val (15%) for early stopping
        tr_idx, val_idx = train_test_split(train_val_idx, test_size=0.15, random_state=RANDOM_SEED + fold_num)

        # -------------------------------------------------------------
        # 1. Train 2-Way Model (CNN + LSTM, 96-dim)
        # -------------------------------------------------------------
        set_seed(RANDOM_SEED + fold_num)
        model_2way = build_model(96, seed=RANDOM_SEED + fold_num)
        early_stop_2w = callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
        reduce_lr_2w = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6)

        model_2way.fit(
            fused_2way[tr_idx], y[tr_idx],
            validation_data=(fused_2way[val_idx], y[val_idx]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop_2w, reduce_lr_2w],
            verbose=0,
        )
        preds_2way = model_2way.predict(fused_2way[test_idx], verbose=0).flatten()
        m_2way = compute_metrics(y[test_idx], preds_2way)
        fold_results_2way.append(m_2way)

        # -------------------------------------------------------------
        # 2. Train 3-Way Model (CNN + LSTM + GCN, 128-dim)
        # -------------------------------------------------------------
        set_seed(RANDOM_SEED + fold_num)
        model_3way = build_model(128, seed=RANDOM_SEED + fold_num)
        early_stop_3w = callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
        reduce_lr_3w = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6)

        model_3way.fit(
            fused_3way[tr_idx], y[tr_idx],
            validation_data=(fused_3way[val_idx], y[val_idx]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop_3w, reduce_lr_3w],
            verbose=0,
        )
        preds_3way = model_3way.predict(fused_3way[test_idx], verbose=0).flatten()
        m_3way = compute_metrics(y[test_idx], preds_3way)
        fold_results_3way.append(m_3way)

        fold_elapsed = time.time() - fold_start
        logger.info(
            "Fold %d Finished (%.1fs) | 2-Way: R2=%.4f, RMSE=%.1f, MAE=%.1f | 3-Way: R2=%.4f, RMSE=%.1f, MAE=%.1f",
            fold_num, fold_elapsed,
            m_2way["r2"], m_2way["rmse"], m_2way["mae"],
            m_3way["r2"], m_3way["rmse"], m_3way["mae"]
        )

    # -----------------------------------------------------------------
    # Summary Statistics & Paired Comparison
    # -----------------------------------------------------------------
    metrics_keys = ["r2", "rmse", "mae", "medae", "mape", "smape", "medape"]
    summary: Dict[str, Any] = {
        "n_splits": N_SPLITS,
        "random_seed": RANDOM_SEED,
        "total_samples": len(y),
        "per_fold": {
            f"fold_{i+1}": {
                "2way": fold_results_2way[i],
                "3way_gcn": fold_results_3way[i],
            }
            for i in range(N_SPLITS)
        },
        "aggregate": {},
        "wins": {
            "r2": 0,
            "rmse": 0,
            "mae": 0,
            "medae": 0,
            "smape": 0,
            "mape": 0,
            "medape": 0,
        }
    }

    for k in metrics_keys:
        vals_2w = [f[k] for f in fold_results_2way]
        vals_3w = [f[k] for f in fold_results_3way]
        summary["aggregate"][k] = {
            "2way_mean": float(np.mean(vals_2w)),
            "2way_std": float(np.std(vals_2w)),
            "3way_mean": float(np.mean(vals_3w)),
            "3way_std": float(np.std(vals_3w)),
        }

    # Paired win counts (3-Way vs 2-Way)
    for i in range(N_SPLITS):
        if fold_results_3way[i]["r2"] > fold_results_2way[i]["r2"]:
            summary["wins"]["r2"] += 1
        if fold_results_3way[i]["rmse"] < fold_results_2way[i]["rmse"]:
            summary["wins"]["rmse"] += 1
        if fold_results_3way[i]["mae"] < fold_results_2way[i]["mae"]:
            summary["wins"]["mae"] += 1
        if fold_results_3way[i]["medae"] < fold_results_2way[i]["medae"]:
            summary["wins"]["medae"] += 1
        if fold_results_3way[i]["smape"] < fold_results_2way[i]["smape"]:
            summary["wins"]["smape"] += 1
        if fold_results_3way[i]["mape"] < fold_results_2way[i]["mape"]:
            summary["wins"]["mape"] += 1
        if fold_results_3way[i]["medape"] < fold_results_2way[i]["medape"]:
            summary["wins"]["medape"] += 1

    # Save to JSON
    with open(CV_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("CV results saved to '%s'.", CV_RESULTS_JSON)

    # Save CSV summary
    rows = []
    for i in range(N_SPLITS):
        rows.append({
            "Fold": i + 1,
            "2Way_R2": fold_results_2way[i]["r2"],
            "3Way_R2": fold_results_3way[i]["r2"],
            "2Way_RMSE": fold_results_2way[i]["rmse"],
            "3Way_RMSE": fold_results_3way[i]["rmse"],
            "2Way_MAE": fold_results_2way[i]["mae"],
            "3Way_MAE": fold_results_3way[i]["mae"],
            "2Way_MedAE": fold_results_2way[i]["medae"],
            "3Way_MedAE": fold_results_3way[i]["medae"],
            "2Way_sMAPE": fold_results_2way[i]["smape"],
            "3Way_sMAPE": fold_results_3way[i]["smape"],
            "2Way_MAPE": fold_results_2way[i]["mape"],
            "3Way_MAPE": fold_results_3way[i]["mape"],
        })
    pd.DataFrame(rows).to_csv(CV_RESULTS_CSV, index=False)

    from scipy.stats import ttest_rel, wilcoxon

    # Print Final CV Report
    agg = summary["aggregate"]
    print("\n" + "=" * 95)
    print("5-FOLD CROSS-VALIDATION SUMMARY (Weighted Graph & Non-Geo Filtered)")
    print("=" * 95)
    print(f"{'Metric':<25} | {'2-Way (CNN-LSTM)':<20} | {'3-Way (CNN-LSTM-GCN)':<20} | {'t-test p':<10} | {'Wilcoxon p':<10}")
    print("-" * 95)
    for k, label, is_pct in [
        ("r2", "R² Score (%)", True),
        ("rmse", "Root Mean Sq Error (RMSE)", False),
        ("mae", "Mean Absolute Error (MAE)", False),
        ("medae", "Median Abs Error (MedAE)", False),
        ("smape", "Symmetric MAPE (sMAPE %)", True),
        ("mape", "Mean Abs Pct Error (MAPE)", True),
    ]:
        v2 = [f[k] for f in fold_results_2way]
        v3 = [f[k] for f in fold_results_3way]
        m2, s2 = agg[k]["2way_mean"], agg[k]["2way_std"]
        m3, s3 = agg[k]["3way_mean"], agg[k]["3way_std"]

        # Statistical tests
        try:
            _, p_ttest = ttest_rel(v2, v3)
        except Exception:
            p_ttest = float("nan")
        try:
            _, p_wilcoxon = wilcoxon(v2, v3)
        except Exception:
            p_wilcoxon = float("nan")

        if is_pct and k == "r2":
            str_2w = f"{m2*100:6.2f}% ± {s2*100:4.2f}%"
            str_3w = f"{m3*100:6.2f}% ± {s3*100:4.2f}%"
        elif is_pct:
            str_2w = f"{m2:6.2f}% ± {s2:4.2f}%"
            str_3w = f"{m3:6.2f}% ± {s3:4.2f}%"
        else:
            str_2w = f"{m2:6.2f} ± {s2:6.2f}"
            str_3w = f"{m3:6.2f} ± {s3:6.2f}"

        print(f"{label:<25} | {str_2w:<20} | {str_3w:<20} | {p_ttest:10.4f} | {p_wilcoxon:10.4f}")
    print("=" * 95)


if __name__ == "__main__":
    run_cross_validation()
