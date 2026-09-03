"""
=========================================================
Project : A Hybrid CNN-LSTM Framework for Spatio-Temporal Crime Prediction
Module  : LSTM Temporal Branch

Dataset :
Crimes_in_india_2001-2013_features.csv (output of feature_engineering.py)

Description :
Builds and trains the LSTM branch that learns temporal crime patterns
per state/district over the 2001-2013 window. Sequences are built by
grouping the engineered dataset on (STATE_ENCODED, DISTRICT_ENCODED)
and sliding a fixed-length window across YEAR_INDEX. The trained
model's Dense(32) layer is exposed separately as a feature extractor,
so its embeddings can be concatenated with the CNN branch's embeddings
in hybrid_model.py.
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

TARGET_COLUMN = "TOTAL IPC CRIMES"
GROUP_COLUMNS = ["STATE_ENCODED", "DISTRICT_ENCODED"]
YEAR_ORDER_COLUMN = "YEAR_INDEX"

# Columns that identify a row or duplicate information already carried
# by the encoded/engineered columns - these are never fed to the model
# as predictors.
NON_FEATURE_COLUMNS = ["Id", "STATE/UT", "DISTRICT", "YEAR", TARGET_COLUMN]

SEQUENCE_LENGTH = 3     # years of history used to predict the next year
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 1e-3
L2_LAMBDA = 1e-4

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_data(data_path: Path) -> pd.DataFrame:
    """
    Loads the engineered dataset and sorts it so every district's
    records are contiguous and ordered by year. create_sequences()
    relies on this ordering - it does not re-sort internally, since
    sorting is a dataset-wide concern that belongs here, not repeated
    inside every grouped iteration.
    """
    logger.info("Loading dataset...")
    df = pd.read_csv(data_path)
    df = df.sort_values(by=GROUP_COLUMNS + [YEAR_ORDER_COLUMN]).reset_index(drop=True)
    logger.info("Dataset loaded: %d rows, %d columns.", df.shape[0], df.shape[1])
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Returns every column that is a legitimate model predictor - i.e.
    everything except the identifier, raw text, raw year, and target
    columns. Deriving this list from the DataFrame rather than typing
    it out keeps it correct even if feature_engineering.py's output
    column set changes slightly.
    """
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def create_sequences(
    df: pd.DataFrame,
    feature_columns: list,
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple:
    """
    Builds fixed-length temporal sequences per (state, district) group.
    For a group with N years of records, this produces N - sequence_length
    windows, each mapping `sequence_length` consecutive years of features
    to the TOTAL IPC CRIMES value of the year immediately following the
    window. Groups with fewer years than sequence_length + 1 can't form
    a single valid window and are skipped rather than padded, since
    padding a district's crime history with artificial zeros would
    distort the temporal signal the LSTM is supposed to learn.
    """
    logger.info("Creating sequences...")
    sequences, targets = [], []

    for _, group in df.groupby(GROUP_COLUMNS):
        group = group.sort_values(YEAR_ORDER_COLUMN)
        feature_matrix = group[feature_columns].to_numpy(dtype=np.float32)
        target_vector = group[TARGET_COLUMN].to_numpy(dtype=np.float32)

        if len(group) <= sequence_length:
            continue

        for start in range(len(group) - sequence_length):
            end = start + sequence_length
            sequences.append(feature_matrix[start:end])
            targets.append(target_vector[end])

    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    logger.info(
        "Sequences created: %d samples, window length %d, %d features per step.",
        X.shape[0], sequence_length, X.shape[2],
    )
    return X, y


def build_model(input_shape: tuple) -> Model:
    """
    Builds the LSTM temporal branch:

        Input -> LSTM(128, return_sequences=True) -> BatchNorm -> Dropout(0.3)
              -> LSTM(64) -> BatchNorm
              -> Dense(64, ReLU) -> BatchNorm -> Dropout(0.3)
              -> Dense(32, "lstm_embedding")
              -> Dense(1)

    The Dense(32) layer is named explicitly so build_feature_extractor()
    can locate it by name later, without depending on layer position -
    a name lookup survives small architecture edits, an index lookup
    doesn't.
    """
    he_init = initializers.HeNormal(seed=RANDOM_SEED)
    l2_reg = regularizers.l2(L2_LAMBDA)

    inputs = layers.Input(shape=input_shape, name="lstm_input")

    x = layers.LSTM(
        128, return_sequences=True, kernel_initializer=he_init, kernel_regularizer=l2_reg
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.LSTM(64, kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(64, activation="relu", kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    embedding = layers.Dense(
        32, activation="relu", kernel_initializer=he_init,
        kernel_regularizer=l2_reg, name="lstm_embedding",
    )(x)

    outputs = layers.Dense(1, name="crime_count_output")(embedding)

    model = Model(inputs=inputs, outputs=outputs, name="lstm_temporal_branch")

    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def save_training_history(history: tf.keras.callbacks.History, results_dir: Path) -> None:
    """
    Saves the per-epoch training history (loss, val_loss, mae, val_mae)
    to CSV so experiments can be compared later without re-running
    training, and so results can be dropped straight into a report.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    history_path = results_dir / "lstm_training_history.csv"
    pd.DataFrame(history.history).to_csv(history_path, index=False)
    logger.info("Training history saved to: %s", history_path)


def train_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    checkpoint_path: Path,
) -> tf.keras.callbacks.History:
    """
    Trains the LSTM branch with early stopping, LR reduction on
    plateau, and checkpointing of the best-validation-loss weights.
    Restoring best weights at the end of training (rather than keeping
    whatever the final epoch produced) matters here because the
    dataset is small enough that late epochs can overfit visibly.
    """
    logger.info("Training...")

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


def evaluate_model(
    model: Model,
    history: tf.keras.callbacks.History,
    X_test: np.ndarray,
    y_test: np.ndarray,
    results_dir: Path,
) -> dict:
    """
    Reports final training/validation loss, computes MAE, RMSE and R^2
    on the held-out test sequences, saves loss/MAE-vs-epoch plots plus
    an actual-vs-predicted scatter plot to results_dir, and writes the
    raw predictions to CSV. Metrics are returned as a dict rather than
    only printed, so hybrid_model.py can later log them alongside the
    CNN branch's numbers for a side-by-side comparison.
    """
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

    # Raw predictions vs actuals, for later inspection or reporting
    predictions_path = results_dir / "lstm_predictions.csv"
    pd.DataFrame({"Actual": y_test, "Predicted": predictions}).to_csv(
        predictions_path, index=False
    )
    logger.info("Predictions saved to: %s", predictions_path)

    # Loss vs Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("LSTM Branch - Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "lstm_loss_vs_epoch.png")
    plt.close()

    # MAE vs Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["mae"], label="Training MAE")
    plt.plot(history.history["val_mae"], label="Validation MAE")
    plt.title("LSTM Branch - MAE vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "lstm_mae_vs_epoch.png")
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
    plt.title("LSTM Branch - Actual vs Predicted Crime Count")
    plt.xlabel("Actual Crime Count")
    plt.ylabel("Predicted Crime Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "lstm_actual_vs_predicted.png")
    plt.close()

    logger.info("Actual vs predicted scatter plot saved to: %s", results_dir)

    return {
        "train_loss": final_train_loss,
        "val_loss": final_val_loss,
        "test_mae": mae,
        "test_rmse": rmse,
        "test_r2": r2,
    }


def extract_features(
    model: Model, X: np.ndarray, results_dir: Path = RESULTS_DIR
) -> tuple:
    """
    Builds a feature-extractor model that shares the trained LSTM
    branch's weights and outputs the 32-dimensional "lstm_embedding"
    layer instead of the final crime-count prediction. Reusing the
    already-trained layers (rather than rebuilding and retraining a
    separate network) is what makes these embeddings meaningful for
    fusion in hybrid_model.py. The embeddings are also persisted to
    disk since hybrid_model.py will need to load them later rather
    than recompute them.
    """
    embedding_layer = model.get_layer("lstm_embedding")
    feature_extractor = Model(
        inputs=model.input, outputs=embedding_layer.output, name="lstm_feature_extractor",
    )
    embeddings = feature_extractor.predict(X, verbose=0)
    logger.info("Extracted embeddings shape: %s", embeddings.shape)

    results_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = results_dir / "lstm_embeddings.npy"
    np.save(embeddings_path, embeddings)
    logger.info("LSTM embeddings saved to: %s", embeddings_path)

    return embeddings, feature_extractor


def save_models(model: Model, feature_extractor: Model, models_dir: Path) -> None:
    """Saves the full LSTM model and the standalone feature extractor."""
    logger.info("Saving model...")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "lstm_model.keras"
    extractor_path = models_dir / "lstm_feature_extractor.keras"

    model.save(model_path)
    feature_extractor.save(extractor_path)

    logger.info("[Saved] LSTM model -> %s", model_path)
    logger.info("[Saved] LSTM feature extractor -> %s", extractor_path)


def main() -> None:
    """Entry point: builds, trains, evaluates, and saves the LSTM temporal branch."""
    df = load_data(DATA_PATH)
    feature_columns = get_feature_columns(df)
    X, y = create_sequences(df, feature_columns)

    # 70% train, 15% validation, 15% test - sequences are shuffled since
    # each one is already an independent (district, window) sample, not
    # a continuation of the previous one.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, shuffle=True,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, shuffle=True,
    )
    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d",
        len(X_train), len(X_val), len(X_test),
    )

    logger.info("Building LSTM...")
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary(print_fn=logger.info)
    logger.info("Total Parameters: %s", model.count_params())

    checkpoint_path = MODELS_DIR / "lstm_model_checkpoint.keras"
    history = train_model(model, X_train, y_train, X_val, y_val, checkpoint_path)
    save_training_history(history, RESULTS_DIR)

    evaluate_model(model, history, X_test, y_test, RESULTS_DIR)

    _, feature_extractor = extract_features(model, X_test, RESULTS_DIR)
    save_models(model, feature_extractor, MODELS_DIR)

    logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()
