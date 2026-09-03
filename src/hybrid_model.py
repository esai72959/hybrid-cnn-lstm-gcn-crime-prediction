"""
=========================================================
Project : A Hybrid CNN-LSTM Framework for Spatio-Temporal Crime Prediction
Module  : Hybrid Fusion Model

Dataset :
Crimes_in_india_2001-2013_features.csv (output of feature_engineering.py)

Description :
Loads the already-trained CNN spatial branch (cnn_model.py) and LSTM
temporal branch (lstm_model.py), freezes both, extracts their learned
embeddings (64-dim spatial + 32-dim temporal), and trains a small
fusion network on top of the concatenated 96-dim representation to
produce the final crime-count prediction.

Alignment note
--------------
The CNN branch treats every (state, district, year) row as an
independent sample. The LSTM branch instead groups rows by
(STATE_ENCODED, DISTRICT_ENCODED) and slides a SEQUENCE_LENGTH-year
window over YEAR_INDEX, so one LSTM sample corresponds to several
consecutive rows and predicts the year immediately after the window.
To fuse the two branches meaningfully, every LSTM sequence is paired
with the CNN embedding of that same "target" row (the year being
predicted) rather than with an arbitrary/unrelated row. This is done
by tracking each sequence's target row position while building the
sequences, and using those positions to pull the matching rows for
the CNN branch. Without this step the two branches would not agree on
what a "sample" is, and their embeddings could not be concatenated.
=========================================================
"""

import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

from tensorflow.keras import Model, callbacks, initializers, layers, optimizers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "dataset" / "Crimes_in_india_2001-2013_features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# These paths match the files actually produced by cnn_model.py and
# lstm_model.py in this project. cnn_model.py saves its trainer model
# (the one whose embedding_bn layer we need) under CNNConfig.model_filename,
# and lstm_model.py saves its full trained model as lstm_model.keras.
CNN_MODEL_PATH = MODELS_DIR / "cnn_feature_extractor.keras"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.keras"

CNN_EMBEDDING_LAYER = "embedding_bn"    # post-BatchNorm 64-dim spatial embedding
LSTM_EMBEDDING_LAYER = "lstm_embedding"  # 32-dim temporal embedding

TARGET_COLUMN = "TOTAL IPC CRIMES"
GROUP_COLUMNS = ["STATE_ENCODED", "DISTRICT_ENCODED"]
YEAR_ORDER_COLUMN = "YEAR_INDEX"
SEQUENCE_LENGTH = 3  # must match SEQUENCE_LENGTH used in lstm_model.py

# Columns excluded from the CNN's feature vector, mirroring
# CNNConfig.columns_to_exclude in cnn_model.py.
CNN_EXCLUDED_COLUMNS = [TARGET_COLUMN, "YEAR", "YEAR_INDEX", "Id", "ID", "id"]

# Columns excluded from the LSTM's feature vector, mirroring
# NON_FEATURE_COLUMNS in lstm_model.py.
LSTM_EXCLUDED_COLUMNS = ["Id", "STATE/UT", "DISTRICT", "YEAR", TARGET_COLUMN]

FUSION_INPUT_DIM = 96  # 64 (CNN) + 32 (LSTM)
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 1e-3
L2_LAMBDA = 1e-4

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Loads the engineered dataset and sorts it so every district's
    records are contiguous and ordered by year. This ordering is what
    lets extract_features() build LSTM windows and locate each
    window's target row correctly - it is not re-derived anywhere
    else, since sorting is a dataset-wide concern that belongs here.
    """
    logger.info("Loading dataset...")
    df = pd.read_csv(data_path)
    df = df.sort_values(by=GROUP_COLUMNS + [YEAR_ORDER_COLUMN]).reset_index(drop=True)
    logger.info("Dataset loaded: %d rows, %d columns.", df.shape[0], df.shape[1])
    return df


def _cnn_feature_columns(df: pd.DataFrame) -> list:
    """Numeric columns the CNN branch was trained on (see cnn_model.py)."""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    return [c for c in numeric_columns if c not in CNN_EXCLUDED_COLUMNS]


def _lstm_feature_columns(df: pd.DataFrame) -> list:
    """Feature columns the LSTM branch was trained on (see lstm_model.py)."""
    return [c for c in df.columns if c not in LSTM_EXCLUDED_COLUMNS]


def create_cnn_feature_extractor(model_path: Path = CNN_MODEL_PATH) -> Model:
    """
    Loads the pretrained CNN and wraps it in a frozen sub-model that
    outputs the 64-dimensional spatial embedding instead of the
    auxiliary crime-count prediction. The layer is located by name
    (embedding_bn) rather than by position, matching the same
    survives-small-edits reasoning used in cnn_model.py.
    """
    logger.info("Loading pretrained CNN...")
    cnn_model = tf.keras.models.load_model(model_path)
    cnn_model.trainable = False

    embedding_layer = cnn_model.get_layer(CNN_EMBEDDING_LAYER)
    feature_extractor = Model(
        inputs=cnn_model.input,
        outputs=embedding_layer.output,
        name="cnn_feature_extractor",
    )
    feature_extractor.trainable = False
    logger.info("CNN feature extractor ready (output dim: %d).", embedding_layer.output_shape[-1])
    return feature_extractor


def create_lstm_feature_extractor(model_path: Path = LSTM_MODEL_PATH) -> Model:
    """
    Loads the pretrained LSTM and wraps it in a frozen sub-model that
    outputs the 32-dimensional temporal embedding instead of the
    crime-count prediction, matching the extract_features() logic in
    lstm_model.py exactly so the fusion network sees the same vectors.
    """
    logger.info("Loading pretrained LSTM...")
    lstm_model = tf.keras.models.load_model(model_path)
    lstm_model.trainable = False

    embedding_layer = lstm_model.get_layer(LSTM_EMBEDDING_LAYER)
    feature_extractor = Model(
        inputs=lstm_model.input,
        outputs=embedding_layer.output,
        name="lstm_feature_extractor",
    )
    feature_extractor.trainable = False
    logger.info("LSTM feature extractor ready (output dim: %d).", embedding_layer.output_shape[-1])
    return feature_extractor


def _build_lstm_sequences(
    df: pd.DataFrame, feature_columns: list, sequence_length: int = SEQUENCE_LENGTH
) -> tuple:
    """
    Rebuilds the same (sequence, target) windows as
    lstm_model.create_sequences(), but additionally records the row
    position (in the sorted, reset-index dataframe) of each window's
    target year. Those positions are the bridge that lets the CNN
    branch be evaluated on the exact same "sample" as the LSTM branch.
    """
    sequences, targets, target_positions = [], [], []

    for _, group in df.groupby(GROUP_COLUMNS):
        group = group.sort_values(YEAR_ORDER_COLUMN)
        feature_matrix = group[feature_columns].to_numpy(dtype=np.float32)
        target_vector = group[TARGET_COLUMN].to_numpy(dtype=np.float32)
        row_positions = group.index.to_numpy()  # positions in the sorted df

        if len(group) <= sequence_length:
            continue

        for start in range(len(group) - sequence_length):
            end = start + sequence_length
            sequences.append(feature_matrix[start:end])
            targets.append(target_vector[end])
            target_positions.append(row_positions[end])

    X_lstm = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    target_positions = np.array(target_positions, dtype=np.int64)
    return X_lstm, y, target_positions


def extract_features(
    df: pd.DataFrame,
    cnn_extractor: Model,
    lstm_extractor: Model,
) -> tuple:
    """
    Produces the fused 96-dimensional feature matrix the hybrid model
    trains on. Each fused row is [CNN spatial embedding of the target
    year's row | LSTM temporal embedding of the preceding
    SEQUENCE_LENGTH years for that same district], paired with that
    target year's TOTAL IPC CRIMES value.
    """
    lstm_feature_columns = _lstm_feature_columns(df)
    cnn_feature_columns = _cnn_feature_columns(df)

    logger.info("Extracting LSTM embeddings...")
    X_lstm, y, target_positions = _build_lstm_sequences(df, lstm_feature_columns)
    lstm_embeddings = lstm_extractor.predict(X_lstm, verbose=0)

    logger.info("Extracting CNN embeddings...")
    cnn_feature_matrix = df[cnn_feature_columns].to_numpy(dtype=np.float32)
    cnn_feature_matrix = cnn_feature_matrix.reshape((cnn_feature_matrix.shape[0], -1, 1))
    X_cnn = cnn_feature_matrix[target_positions]
    cnn_embeddings = cnn_extractor.predict(X_cnn, verbose=0)

    fused_features = np.concatenate([cnn_embeddings, lstm_embeddings], axis=1)
    logger.info(
        "Fused feature matrix built: %d samples, %d dims (%d CNN + %d LSTM).",
        fused_features.shape[0], fused_features.shape[1],
        cnn_embeddings.shape[1], lstm_embeddings.shape[1],
    )
    return fused_features, y


def build_hybrid_model(input_dim: int = FUSION_INPUT_DIM) -> Model:
    """
    Builds the fusion network:

        Input(96) -> Dense(128) -> BatchNorm -> ReLU -> Dropout(0.3)
                  -> Dense(64)  -> BatchNorm -> ReLU -> Dropout(0.3)
                  -> Dense(32, "hybrid_embedding")
                  -> Dense(1)

    Only this network is trainable - the CNN and LSTM branches feeding
    it were frozen back in create_cnn_feature_extractor() /
    create_lstm_feature_extractor() and never re-enter training here.
    The Dense(32) layer is named so a feature extractor for the fused
    representation can be built the same way the CNN/LSTM branches do.
    """
    logger.info("Building Hybrid Model...")
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

    embedding = layers.Dense(
        32, kernel_initializer=he_init, kernel_regularizer=l2_reg, name="hybrid_embedding"
    )(x)

    outputs = layers.Dense(1, name="crime_count_output")(embedding)

    model = Model(inputs=inputs, outputs=outputs, name="hybrid_cnn_lstm_model")

    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    model.summary(print_fn=logger.info)
    logger.info("Total Parameters: %s", model.count_params())
    return model


def train_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    checkpoint_path: Path,
) -> tf.keras.callbacks.History:
    """
    Trains the fusion network with early stopping, LR reduction on
    plateau, and checkpointing of the best-validation-loss weights,
    mirroring the callback configuration used for the CNN and LSTM
    branches so all three modules are trained under comparable
    conditions.
    """
    logger.info("Training Hybrid Model...")

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True,
    )
    checkpoint = callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path), monitor="val_loss",
        save_best_only=True, save_weights_only=False,
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, checkpoint, reduce_lr],
        verbose=2,
    )
    return history


def save_training_history(history: tf.keras.callbacks.History, results_dir: Path = RESULTS_DIR) -> None:
    """
    Saves the per-epoch training history (loss, val_loss, mae, val_mae)
    to CSV so the hybrid run can be compared against the CNN and LSTM
    branches without re-running training.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    history_path = results_dir / "hybrid_training_history.csv"
    pd.DataFrame(history.history).to_csv(history_path, index=False)
    logger.info("Training history saved to: %s", history_path)


def evaluate_model(
    model: Model,
    history: tf.keras.callbacks.History,
    X_test: np.ndarray,
    y_test: np.ndarray,
    results_dir: Path = RESULTS_DIR,
) -> dict:
    """
    Reports final training/validation loss, computes MAE, RMSE and R^2
    on the held-out test set, saves loss/MAE-vs-epoch plots plus an
    actual-vs-predicted scatter plot, and writes the raw predictions
    to CSV so they can be compared directly against the CNN and LSTM
    branches' own results files.
    """
    logger.info("Evaluating...")
    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]

    predictions = model.predict(X_test, verbose=0).flatten()
    mae = float(np.mean(np.abs(predictions - y_test)))
    rmse = float(np.sqrt(np.mean((predictions - y_test) ** 2)))
    r2 = float(r2_score(y_test, predictions))

    logger.info("Training Loss   : %.4f", final_train_loss)
    logger.info("Validation Loss : %.4f", final_val_loss)
    logger.info("Test MAE        : %.4f", mae)
    logger.info("Test RMSE       : %.4f", rmse)
    logger.info("Test R^2        : %.4f", r2)

    results_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = results_dir / "hybrid_predictions.csv"
    pd.DataFrame({"Actual": y_test, "Predicted": predictions}).to_csv(
        predictions_path, index=False
    )
    logger.info("Predictions saved to: %s", predictions_path)

    # Loss vs Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Hybrid Model - Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "hybrid_loss_vs_epoch.png")
    plt.close()

    # MAE vs Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["mae"], label="Training MAE")
    plt.plot(history.history["val_mae"], label="Validation MAE")
    plt.title("Hybrid Model - MAE vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "hybrid_mae_vs_epoch.png")
    plt.close()

    logger.info("Loss and MAE plots saved to: %s", results_dir)

    # Actual vs Predicted scatter
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, predictions, alpha=0.5, edgecolors="k", linewidths=0.3)
    lims = [
        min(y_test.min(), predictions.min()),
        max(y_test.max(), predictions.max()),
    ]
    plt.plot(lims, lims, "r--", label="Ideal (y = x)")
    plt.title("Hybrid Model - Actual vs Predicted Crime Count")
    plt.xlabel("Actual Crime Count")
    plt.ylabel("Predicted Crime Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "hybrid_actual_vs_predicted.png")
    plt.close()

    logger.info("Actual vs predicted scatter plot saved to: %s", results_dir)

    return {
        "train_loss": final_train_loss,
        "val_loss": final_val_loss,
        "test_mae": mae,
        "test_rmse": rmse,
        "test_r2": r2,
    }


def save_models(model: Model, models_dir: Path = MODELS_DIR) -> None:
    """
    Saves the full hybrid model and a standalone feature extractor
    that exposes the fused 32-dimensional hybrid_embedding layer, the
    same pattern cnn_model.py and lstm_model.py use for their own
    branch embeddings.
    """
    logger.info("Saving Hybrid Model...")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "hybrid_model.keras"
    extractor_path = models_dir / "hybrid_feature_extractor.keras"

    model.save(model_path)

    embedding_layer = model.get_layer("hybrid_embedding")
    feature_extractor = Model(
        inputs=model.input, outputs=embedding_layer.output, name="hybrid_feature_extractor",
    )
    feature_extractor.save(extractor_path)

    logger.info("[Saved] Hybrid model -> %s", model_path)
    logger.info("[Saved] Hybrid feature extractor -> %s", extractor_path)


def main() -> None:
    """Entry point: loads pretrained branches, fuses embeddings, trains, evaluates, and saves the hybrid model."""
    df = load_data(DATA_PATH)

    cnn_extractor = create_cnn_feature_extractor(CNN_MODEL_PATH)
    lstm_extractor = create_lstm_feature_extractor(LSTM_MODEL_PATH)

    fused_features, y = extract_features(df, cnn_extractor, lstm_extractor)

    # 70% train, 15% validation, 15% test - samples are shuffled since
    # each fused row is already an independent (district, target-year)
    # sample, not a continuation of the previous one.
    X_train, X_temp, y_train, y_temp = train_test_split(
        fused_features, y, test_size=0.3, random_state=RANDOM_SEED, shuffle=True,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, shuffle=True,
    )
    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d",
        len(X_train), len(X_val), len(X_test),
    )

    model = build_hybrid_model(input_dim=fused_features.shape[1])

    checkpoint_path = MODELS_DIR / "hybrid_model_checkpoint.keras"
    history = train_model(model, X_train, y_train, X_val, y_val, checkpoint_path)
    save_training_history(history, RESULTS_DIR)

    evaluate_model(model, history, X_test, y_test, RESULTS_DIR)

    save_models(model, MODELS_DIR)

    logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()
