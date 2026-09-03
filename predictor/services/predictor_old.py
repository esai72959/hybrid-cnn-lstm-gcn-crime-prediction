"""
predictor.py

Production-grade inference service for the Hybrid CNN-LSTM Spatio-Temporal
Crime Prediction system.

This module is responsible exclusively for turning a (state, district,
prediction_year) selection into a model-ready tensor - sourced entirely
from the engineered dataset via DatasetLoader - running inference through
the pre-trained Hybrid CNN-LSTM model (via ModelLoader), and translating
the raw prediction into a risk classification with an actionable
recommendation.

All model / scaler / feature-column artifacts come from ModelLoader, and
all historical feature data comes from DatasetLoader. This module never
loads TensorFlow, the scaler, the encoders, or the CSV directly.

Author: Final Year B.Tech Project - Hybrid CNN-LSTM Crime Prediction
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from predictor.services.dataset_loader import DatasetLoader
from predictor.services.model_loader import ModelLoader

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


class PredictionError(Exception):
    """Raised when input preparation or model inference fails."""


# --------------------------------------------------------------------------
# Columns present in the engineered dataset that are NOT model input
# features (identifiers, human-readable labels, and the training target).
# --------------------------------------------------------------------------
NON_FEATURE_COLUMNS: List[str] = [
    "Id",
    "STATE/UT",
    "DISTRICT",
    "TOTAL IPC CRIMES",
]

# --------------------------------------------------------------------------
# Risk classification thresholds (kept as constants for easy tuning).
# --------------------------------------------------------------------------
LOW_RISK_THRESHOLD: float = 300.0
HIGH_RISK_THRESHOLD: float = 700.0

RISK_LOW: str = "Low"
RISK_MEDIUM: str = "Medium"
RISK_HIGH: str = "High"

RECOMMENDATIONS: Dict[str, str] = {
    RISK_LOW: "Routine monitoring recommended.",
    RISK_MEDIUM: "Increase police patrolling.",
    RISK_HIGH: "Immediate intervention recommended.",
}


class CrimePredictor:
    """
    Inference service for the Hybrid CNN-LSTM crime prediction model.

    Responsibilities
    -----------------
    - Retrieve the latest engineered historical record for a given
      state/district via DatasetLoader, and project it forward to the
      requested prediction year.
    - Strip identifier/label/target columns and reorder the remaining
      features to exactly match feature_columns.json.
    - Scale and reshape the feature vector to match the trained model's
      expected input.
    - Run inference using the model obtained from ModelLoader.
    - Classify the predicted crime count into a risk category and
      generate a human-readable recommendation.

    This class does NOT load, fit, or cache any ML artifact or dataset
    itself - all artifacts come from ModelLoader and DatasetLoader.
    """

    def __init__(self) -> None:
        """
        Initialize the predictor and eagerly load all required artifacts
        and dataset content exactly once, via ModelLoader and
        DatasetLoader.

        Raises
        ------
        PredictionError
            If ModelLoader or DatasetLoader fails to load anything
            required for inference.
        """
        try:
            self.model_loader = ModelLoader()
            self.model_loader.load_all()

            self.dataset_loader = DatasetLoader()
            self.dataset_loader.load_dataset()

            logger.info(
                "CrimePredictor initialized successfully. Model, "
                "artifacts, and dataset are ready for inference."
            )
        except Exception as exc:
            logger.error("CrimePredictor failed to initialize: %s", exc)
            raise PredictionError(
                f"CrimePredictor failed to initialize: {exc}"
            ) from exc

    # ----------------------------------------------------------------
    # Feature preparation
    # ----------------------------------------------------------------
    @staticmethod
    def _coerce_numeric(column: str, value: Any) -> float:
        """
        Convert a feature value to float, raising a meaningful error if
        the value is non-numeric.

        Parameters
        ----------
        column : str
            Name of the feature column (used for error messages).
        value : Any
            Value to convert.

        Returns
        -------
        float
            Numeric representation of the value.

        Raises
        ------
        PredictionError
            If the value cannot be converted to float.
        """
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise PredictionError(
                f"Feature '{column}' expects a numeric value but got "
                f"'{value}'."
            ) from exc

    def _build_feature_vector(
        self, state: str, district: str, prediction_year: int
    ) -> np.ndarray:
        """
        Build the ordered, scaled feature vector for a prediction
        request.

        Steps
        -----
        1. Fetch a projected feature record for `prediction_year` from
           DatasetLoader.prepare_prediction_record().
        2. Drop identifier/label/target columns that are not model
           input (Id, STATE/UT, DISTRICT, TOTAL IPC CRIMES).
        3. Look up the exact feature order from
           ModelLoader.get_feature_columns().
        4. Reorder the record to match feature_columns.json exactly.
        5. Convert to a numpy array and scale it using the saved
           MinMaxScaler (transform only - never re-fitted).

        Parameters
        ----------
        state : str
            State/UT name.
        district : str
            District name.
        prediction_year : int
            Future year to generate a forecast for.

        Returns
        -------
        np.ndarray
            A 2D array of shape (1, num_features), scaled and ordered
            to match the model's training-time feature layout.

        Raises
        ------
        PredictionError
            If the record cannot be prepared, required feature columns
            are missing, or scaling fails.
        """
        try:
            # 1. Project the latest historical record to the target year.
            record: pd.Series = (
                self.dataset_loader.prepare_prediction_record(
                    state, district, prediction_year
                )
            )

            # 2. Drop identifier/label/target columns.
            trimmed_record = record.drop(
                labels=[
                    col for col in NON_FEATURE_COLUMNS
                    if col in record.index
                ],
                errors="ignore",
            )

            # 3. Exact feature order used during training.
            feature_columns: List[str] = (
                self.model_loader.get_feature_columns()
            )
            if not feature_columns:
                raise PredictionError(
                    "feature_columns.json is empty or invalid; cannot "
                    "determine model input order."
                )

            missing_columns = [
                col for col in feature_columns
                if col not in trimmed_record.index
            ]
            if missing_columns:
                raise PredictionError(
                    f"Prepared record is missing required feature "
                    f"column(s): {missing_columns}."
                )

            # 4. Reorder exactly to match feature_columns.json.
            ordered_values = [
                self._coerce_numeric(col, trimmed_record[col])
                for col in feature_columns
            ]

            # 5. Convert to numpy and scale (transform only).
            feature_array = np.array(
                ordered_values, dtype=np.float32
            ).reshape(1, -1)

            scaler = self.model_loader.get_scaler()
            scaled_array = scaler.transform(feature_array)

            logger.info(
                "Feature vector built for state='%s', district='%s', "
                "prediction_year=%s | feature_count=%d",
                state, district, prediction_year, len(feature_columns),
            )
            return scaled_array

        except PredictionError:
            raise
        except Exception as exc:
            logger.error(
                "Failed to build feature vector for state='%s', "
                "district='%s', prediction_year=%s: %s",
                state, district, prediction_year, exc,
            )
            raise PredictionError(
                f"Failed to build feature vector: {exc}"
            ) from exc

    def _reshape_for_model(self, scaled_array: np.ndarray) -> np.ndarray:
        """
        Reshape a 2D (1, num_features) scaled array to match the shape
        expected by the trained Hybrid CNN-LSTM model.

        The model's own `input_shape` is introspected so this works
        regardless of whether the model expects:
            - (batch, num_features)                    -> plain dense input
            - (batch, timesteps, num_features)          -> LSTM-style input
            - (batch, num_features, channels)           -> CNN-style input

        Parameters
        ----------
        scaled_array : np.ndarray
            2D array of shape (1, num_features).

        Returns
        -------
        np.ndarray
            Reshaped array matching the model's expected input rank.

        Raises
        ------
        PredictionError
            If the model's input shape is unsupported or incompatible
            with the number of prepared features.
        """
        model = self.model_loader.get_model()
        input_shape = model.input_shape

        # Some models report a list of input shapes (multi-input models).
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        num_features = scaled_array.shape[1]
        target_rank = len(input_shape)

        try:
            if target_rank == 2:
                # (batch, num_features)
                return scaled_array

            if target_rank == 3:
                _, dim1, dim2 = input_shape
                if dim2 in (None, num_features):
                    # (batch, timesteps, num_features) -> single timestep
                    return scaled_array.reshape(1, 1, num_features)
                if dim1 in (None, num_features):
                    # (batch, num_features, channels) -> single channel
                    return scaled_array.reshape(1, num_features, 1)
                raise PredictionError(
                    f"Model input shape {input_shape} is incompatible "
                    f"with prepared feature count ({num_features})."
                )

            raise PredictionError(
                f"Unsupported model input rank: {target_rank} "
                f"(input_shape={input_shape})."
            )
        except PredictionError:
            raise
        except Exception as exc:
            raise PredictionError(
                f"Failed to reshape input for model inference: {exc}"
            ) from exc

    # ----------------------------------------------------------------
    # Risk classification and recommendation
    # ----------------------------------------------------------------
    def classify_risk(self, predicted_value: float) -> str:
        """
        Classify a predicted crime count into a risk category.

        Parameters
        ----------
        predicted_value : float
            Predicted crime count returned by the model.

        Returns
        -------
        str
            One of "Low", "Medium", "High".
        """
        if predicted_value < LOW_RISK_THRESHOLD:
            return RISK_LOW
        if predicted_value <= HIGH_RISK_THRESHOLD:
            return RISK_MEDIUM
        return RISK_HIGH

    def generate_recommendation(self, risk: str) -> str:
        """
        Generate a human-readable recommendation for a given risk level.

        Parameters
        ----------
        risk : str
            Risk category, one of "Low", "Medium", "High".

        Returns
        -------
        str
            Recommended action for the given risk category.

        Raises
        ------
        PredictionError
            If the risk category is not recognized.
        """
        try:
            return RECOMMENDATIONS[risk]
        except KeyError as exc:
            raise PredictionError(
                f"Unknown risk category: '{risk}'."
            ) from exc

    # ----------------------------------------------------------------
    # Public inference entry point
    # ----------------------------------------------------------------
    def predict(
        self, state: str, district: str, prediction_year: int
    ) -> Dict[str, Any]:
        """
        Run end-to-end crime prediction for a state/district/year
        selection.

        Parameters
        ----------
        state : str
            State/UT name.
        district : str
            District name.
        prediction_year : int
            Future year to generate a forecast for.

        Returns
        -------
        Dict[str, Any]
            {
                "status": "success",
                "prediction": float,
                "risk": str,
                "recommendation": str,
                "model": "Hybrid CNN-LSTM"
            }

        Raises
        ------
        PredictionError
            If feature preparation or model inference fails.
        """
        start_time = time.time()
        try:
            scaled_array = self._build_feature_vector(
                state, district, prediction_year
            )
            model_input = self._reshape_for_model(scaled_array)

            model = self.model_loader.get_model()
            raw_prediction = model.predict(model_input, verbose=0)
            predicted_value = float(np.ravel(raw_prediction)[0])
            # Crime counts cannot be negative.
            predicted_value = max(predicted_value, 0.0)

            risk = self.classify_risk(predicted_value)
            recommendation = self.generate_recommendation(risk)

            elapsed_seconds = time.time() - start_time
            logger.info(
                "Prediction completed in %.4f s | state=%s district=%s "
                "prediction_year=%s -> prediction=%.2f risk=%s",
                elapsed_seconds, state, district, prediction_year,
                predicted_value, risk,
            )

            return {
                "status": "success",
                "prediction": round(predicted_value, 2),
                "risk": risk,
                "recommendation": recommendation,
                "model": "Hybrid CNN-LSTM",
            }

        except PredictionError as exc:
            logger.error("Prediction failed: %s", exc)
            raise
        except Exception as exc:
            logger.error("Unexpected error during prediction: %s", exc)
            raise PredictionError(
                f"Unexpected error during prediction: {exc}"
            ) from exc