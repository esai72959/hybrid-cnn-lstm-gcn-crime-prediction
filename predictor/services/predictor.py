"""
predictor.py

Core prediction service for the Hybrid CNN-LSTM Spatio-Temporal Crime
Prediction system.

This module orchestrates the end-to-end prediction workflow by composing
the already-existing DatasetLoader (historical/engineered data access),
ModelLoader (trained artifacts: hybrid model, encoders, scaler, feature
columns) and the feature engineering utilities. It does NOT reimplement
any of those responsibilities.

Part 1 of this file covers:
    - Imports
    - Logger configuration
    - PredictionError exception
    - CrimePredictor class skeleton (__init__, input validation,
      reusable helper methods)

CNN/LSTM/Hybrid inference, risk-scoring and API-facing logic are
implemented in later parts of this file.

Author: Final Year B.Tech Project - Hybrid CNN-LSTM Crime Prediction
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import time

from predictor.services.dataset_loader import DatasetLoader, DatasetLoaderError
from predictor.services.model_loader import ModelLoader, ModelLoaderError

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
    """
    Raised when a prediction request cannot be validated or fulfilled.

    Used for invalid user input (unknown state/district, malformed or
    out-of-range prediction year) as well as any downstream failure
    surfaced from DatasetLoader or ModelLoader during preparation of a
    prediction request.
    """


class CrimePredictor:
    """
    Orchestrates crime prediction requests for the Hybrid CNN-LSTM
    Spatio-Temporal Crime Prediction system.

    This class is a thin coordination layer: it validates incoming
    requests and delegates all data access and artifact access to the
    existing DatasetLoader and ModelLoader singletons respectively. It
    holds no dataset- or model-loading logic of its own.

    Usage
    -----
        predictor = CrimePredictor()
        predictor.validate_input("TELANGANA", "HYDERABAD", 2027)

    Notes
    -----
    - DatasetLoader and ModelLoader are themselves singletons (enforced
      via their own __new__ implementations), so instantiating them here
      simply binds this instance to the single shared instance already
      used elsewhere in the project.
    - This part of the class intentionally implements no inference
      logic (CNN, LSTM, Hybrid, model.predict) and no risk-scoring
      logic. Those are added in later parts.
    """

    def __init__(
        self,
        dataset_loader: Optional[DatasetLoader] = None,
        model_loader: Optional[ModelLoader] = None,
    ) -> None:
        """
        Initialize the CrimePredictor with the shared DatasetLoader and
        ModelLoader singleton instances.

        Parameters
        ----------
        dataset_loader : Optional[DatasetLoader]
            Existing DatasetLoader singleton to use. If not provided,
            the singleton instance is obtained via `DatasetLoader()`.
        model_loader : Optional[ModelLoader]
            Existing ModelLoader singleton to use. If not provided, the
            singleton instance is obtained via `ModelLoader()`.
        """
        self.dataset_loader: DatasetLoader = dataset_loader or DatasetLoader()
        self.model_loader: ModelLoader = model_loader or ModelLoader()

        # Cache for GCN full-graph annual feature tensor
        self._cached_baseline_features: Optional[np.ndarray] = None
        self._cached_baseline_year: Optional[int] = None

        logger.info(
            "CrimePredictor initialized with DatasetLoader and "
            "ModelLoader singleton instances."
        )

    # ----------------------------------------------------------------
    # Input validation
    # ----------------------------------------------------------------
    def validate_input(
        self, state: str, district: str, prediction_year: Any
    ) -> Dict[str, Any]:
        """
        Validate a prediction request's state, district and year.

        Parameters
        ----------
        state : str
            State/UT name to validate against the dataset.
        district : str
            District name to validate against the dataset (within the
            given state).
        prediction_year : Any
            The year to generate a prediction for. Must be an integer
            (or an integer-valued type) strictly greater than the most
            recent year present in the dataset.

        Returns
        -------
        Dict[str, Any]
            Normalized request payload:
            {
                "state": str,
                "district": str,
                "prediction_year": int,
                "latest_dataset_year": int,
            }

        Raises
        ------
        PredictionError
            If any of the following hold:
                - `state` is missing/empty or not present in the dataset
                - `district` is missing/empty or not present under the
                  given state
                - `prediction_year` is not a valid integer
                - `prediction_year` is not strictly greater than the
                  latest year available in the dataset
        """
        normalized_state = self._validate_state(state)
        normalized_district = self._validate_district(
            normalized_state, district
        )
        validated_year = self._validate_prediction_year(normalized_state, normalized_district)

        prediction_year_int, latest_dataset_year = validated_year(prediction_year)

        logger.info(
            "Input validated successfully: state='%s', district='%s', "
            "prediction_year=%d (latest_dataset_year=%d).",
            normalized_state, normalized_district,
            prediction_year_int, latest_dataset_year,
        )

        return {
            "state": normalized_state,
            "district": normalized_district,
            "prediction_year": prediction_year_int,
            "latest_dataset_year": latest_dataset_year,
        }

    # ----------------------------------------------------------------
    # Validation helpers
    # ----------------------------------------------------------------
    def _validate_state(self, state: Any) -> str:
        """
        Verify that `state` is non-empty and exists in the dataset.

        Parameters
        ----------
        state : Any
            Raw state value supplied by the caller.

        Returns
        -------
        str
            The normalized (stripped, upper-cased) state name.

        Raises
        ------
        PredictionError
            If `state` is empty or not found among the dataset's known
            states.
        """
        if state is None or str(state).strip() == "":
            raise PredictionError("'state' is required and cannot be empty.")

        normalized_state = str(state).strip().upper()

        try:
            known_states = self.dataset_loader.get_states()
        except DatasetLoaderError as exc:
            logger.error("Failed to retrieve known states: %s", exc)
            raise PredictionError(f"Unable to validate state: {exc}") from exc

        known_states_upper = {s.strip().upper() for s in known_states}
        if normalized_state not in known_states_upper:
            raise PredictionError(
                f"Unknown state '{state}'. It was not found in the "
                "dataset."
            )

        return normalized_state

    def _validate_district(self, state: str, district: Any) -> str:
        """
        Verify that `district` is non-empty and exists under `state`.

        Parameters
        ----------
        state : str
            Already-validated, normalized state name.
        district : Any
            Raw district value supplied by the caller.

        Returns
        -------
        str
            The normalized (stripped, upper-cased) district name.

        Raises
        ------
        PredictionError
            If `district` is empty or not found under the given state.
        """
        if district is None or str(district).strip() == "":
            raise PredictionError(
                "'district' is required and cannot be empty."
            )

        normalized_district = str(district).strip().upper()

        try:
            known_districts = self.dataset_loader.get_districts(state)
        except DatasetLoaderError as exc:
            logger.error(
                "Failed to retrieve known districts for state '%s': %s",
                state, exc,
            )
            raise PredictionError(
                f"Unable to validate district: {exc}"
            ) from exc

        known_districts_upper = {d.strip().upper() for d in known_districts}
        if normalized_district not in known_districts_upper:
            raise PredictionError(
                f"Unknown district '{district}' for state '{state}'. It "
                "was not found in the dataset."
            )

        return normalized_district

    def _validate_prediction_year(self, state: str, district: str):
        """
        Build a closure-style validator that checks a prediction year
        is an integer strictly greater than the latest year available
        for the given state/district in the dataset.

        Parameters
        ----------
        state : str
            Already-validated, normalized state name.
        district : str
            Already-validated, normalized district name.

        Returns
        -------
        Callable[[Any], tuple[int, int]]
            A callable that accepts a raw `prediction_year` value and
            returns `(prediction_year_int, latest_dataset_year)`.

        Raises
        ------
        PredictionError
            If the latest dataset year cannot be determined.
        """
        try:
            latest_record = self.dataset_loader.get_latest_record(
                state, district
            )
            latest_year = int(latest_record["YEAR"])
        except DatasetLoaderError as exc:
            logger.error(
                "Failed to determine latest dataset year for "
                "state='%s', district='%s': %s", state, district, exc,
            )
            raise PredictionError(
                f"Unable to determine latest available year: {exc}"
            ) from exc

        def _check(prediction_year: Any) -> "tuple[int, int]":
            year_int = self._coerce_year(prediction_year)
            if year_int <= latest_year:
                raise PredictionError(
                    f"'prediction_year' ({year_int}) must be strictly "
                    f"greater than the latest available dataset year "
                    f"({latest_year})."
                )
            return year_int, latest_year

        return _check

    @staticmethod
    def _coerce_year(prediction_year: Any) -> int:
        """
        Coerce a raw prediction year value into a strict integer.

        Parameters
        ----------
        prediction_year : Any
            Raw value supplied by the caller (may be int, numeric
            string, or float with a whole-number value).

        Returns
        -------
        int
            The coerced integer year.

        Raises
        ------
        PredictionError
            If `prediction_year` cannot be interpreted as an integer.
        """
        if isinstance(prediction_year, bool):
            # bool is a subclass of int; explicitly reject it.
            raise PredictionError(
                f"'prediction_year' must be an integer, got "
                f"'{prediction_year}'."
            )

        if isinstance(prediction_year, int):
            return prediction_year

        try:
            year_float = float(prediction_year)
        except (TypeError, ValueError) as exc:
            raise PredictionError(
                f"'prediction_year' must be an integer, got "
                f"'{prediction_year}'."
            ) from exc

        if not year_float.is_integer():
            raise PredictionError(
                f"'prediction_year' must be a whole number, got "
                f"'{prediction_year}'."
            )

        return int(year_float)

    @staticmethod
    def _current_year() -> int:
        """
        Return the current calendar year.

        Returns
        -------
        int
            The current year (server clock).
        """
        return datetime.now().year
    # ----------------------------------------------------------------
    # Historical data retrieval
    # ----------------------------------------------------------------
    def get_historical_records(self, state: str, district: str) -> pd.DataFrame:
        """
        Retrieve all historical records for a validated state/district,
        sorted chronologically (O(1) fast cache with fallback).
        """
        fast_df = self.dataset_loader.get_historical_records_fast(state, district)
        if not fast_df.empty:
            return fast_df

        try:
            dataframe = self.dataset_loader.get_dataframe()
        except DatasetLoaderError as exc:
            logger.error(
                "Failed to load dataset while retrieving historical "
                "records for state='%s', district='%s': %s",
                state, district, exc,
            )
            raise PredictionError(
                f"Unable to retrieve historical records: {exc}"
            ) from exc

        records = self._filter_state_district(dataframe, state, district)

        if records.empty:
            raise PredictionError(
                f"No historical records found for state='{state}', "
                f"district='{district}'."
            )

        sorted_records = records.sort_values(
            by="YEAR", ascending=True
        ).reset_index(drop=True)

        logger.info(
            "Retrieved %d historical record(s) for state='%s', "
            "district='%s' (years %d-%d).",
            len(sorted_records), state, district,
            int(sorted_records["YEAR"].min()),
            int(sorted_records["YEAR"].max()),
        )
        return sorted_records

    def get_latest_record(self, state: str, district: str) -> "pd.Series":
        """
        Retrieve the single most recent historical record for a
        validated state/district.
        """
        try:
            latest_record = self.dataset_loader.get_latest_record(
                state, district
            )
        except DatasetLoaderError as exc:
            logger.error(
                "Failed to retrieve latest record for state='%s', "
                "district='%s': %s", state, district, exc,
            )
            raise PredictionError(
                f"No historical record available for state='{state}', "
                f"district='{district}': {exc}"
            ) from exc

        if latest_record is None:
            raise PredictionError(
                f"No historical record available for state='{state}', "
                f"district='{district}'."
            )

        return latest_record

    @staticmethod
    def _filter_state_district(
        dataframe: "pd.DataFrame", state: str, district: str
    ) -> "pd.DataFrame":
        """
        Filter a dataframe down to rows matching a state/district,
        case-insensitively and whitespace-tolerant.
        """
        normalized_state = state.strip().upper()
        normalized_district = district.strip().upper()

        mask = (
            (
                dataframe["STATE/UT"].astype(str).str.strip().str.upper()
                == normalized_state
            )
            & (
                dataframe["DISTRICT"].astype(str).str.strip().str.upper()
                == normalized_district
            )
        )
        return dataframe.loc[mask].copy(deep=True)

    # ----------------------------------------------------------------
    # Base feature preparation
    # ----------------------------------------------------------------
    def prepare_base_features(self, latest_record: "pd.Series") -> Dict[str, Any]:
        """
        Build the raw (unscaled, unencoded) base feature dictionary for
        a single historical/prepared record, preserving the exact
        column order expected by the model.

        This method only assembles the raw feature vector. It performs
        NO scaling, NO encoding, and NO model inference â€” those are
        handled by later parts of the pipeline.

        Parameters
        ----------
        latest_record : pd.Series
            A historical record (e.g. from `get_latest_record`) or a
            prepared future-year record (e.g. from
            `DatasetLoader.prepare_prediction_record`) containing raw
            column values.

        Returns
        -------
        Dict[str, Any]
            An ordered mapping of feature name -> raw value, ordered
            exactly as in the model's `feature_columns.json`, with
            identifier columns (Id, STATE/UT, DISTRICT) and the target
            column excluded.

        Raises
        ------
        PredictionError
            If the model's feature column list cannot be loaded, or if
            `latest_record` is missing a value for any required
            feature column.
        """
        ordered_feature_columns = self._get_ordered_feature_columns()

        feature_dict: Dict[str, Any] = {}
        missing_columns: List[str] = []

        for column in ordered_feature_columns:
            if column not in latest_record.index:
                missing_columns.append(column)
                continue
            feature_dict[column] = latest_record[column]

        if missing_columns:
            raise PredictionError(
                "Historical record is missing required feature "
                f"column(s): {missing_columns}."
            )

        logger.info(
            "Prepared %d base feature(s) from historical record "
            "(unscaled, unencoded).", len(feature_dict),
        )
        return feature_dict

    def _get_ordered_feature_columns(self) -> List[str]:
        """
        Resolve the ordered list of model feature columns, with
        identifier and target columns excluded.

        Reads the ordered column list from
        `ModelLoader.get_feature_columns()` and the target column name
        from `ModelLoader.get_metadata()`, then strips out columns that
        are not genuine model input features.

        Returns
        -------
        List[str]
            Ordered feature column names, excluding identifiers
            ('Id', 'STATE/UT', 'DISTRICT') and the target column.

        Raises
        ------
        PredictionError
            If the feature columns or metadata cannot be loaded from
            ModelLoader.
        """
        try:
            all_columns = self.model_loader.get_feature_columns()
        except ModelLoaderError as exc:
            logger.error("Failed to load feature columns: %s", exc)
            raise PredictionError(
                f"Unable to load model feature columns: {exc}"
            ) from exc

        try:
            metadata = self.model_loader.get_metadata()
        except ModelLoaderError as exc:
            logger.error("Failed to load model metadata: %s", exc)
            raise PredictionError(
                f"Unable to load model metadata: {exc}"
            ) from exc

        target_column = metadata.get("target_column")
        excluded_columns = set(self._NON_FEATURE_COLUMNS)
        if target_column:
            excluded_columns.add(target_column)

        ordered_feature_columns = [
            column for column in all_columns if column not in excluded_columns
        ]

        return ordered_feature_columns

    # Identifier columns present in the dataset that are never fed to
    # the model as raw features (their encoded counterparts,
    # STATE_ENCODED / DISTRICT_ENCODED, are used instead).
    _NON_FEATURE_COLUMNS: frozenset = frozenset({"Id", "STATE/UT", "DISTRICT"})

    # ----------------------------------------------------------------
    # CNN input preparation
    # ----------------------------------------------------------------

    # Columns present in the base feature dict (see Part 2) that the CNN
    # spatial branch does NOT consume. This mirrors CNN_EXCLUDED_COLUMNS /
    # CNNConfig.columns_to_exclude used when the CNN branch was trained
    # (YEAR and YEAR_INDEX are temporal, not spatial, signals; Id/target/
    # STATE/UT/DISTRICT are already excluded upstream by
    # `_get_ordered_feature_columns`).
    _CNN_NON_SPATIAL_COLUMNS: frozenset = frozenset({"YEAR", "YEAR_INDEX"})

    # Base feature dict keys that hold LabelEncoder-encoded categorical
    # values, mapped to the ModelLoader getter that returns the matching
    # fitted encoder. The engineered dataset already stores these columns
    # pre-encoded (and pre-scaled) as numeric values; this mapping exists
    # purely as a safety net so that a raw string value (e.g. surfaced
    # from an upstream edge case) is still encoded correctly rather than
    # silently fed downstream as-is.
    _CATEGORICAL_COLUMN_ENCODERS: Dict[str, str] = {
        "STATE_ENCODED": "get_state_encoder",
        "DISTRICT_ENCODED": "get_district_encoder",
    }

    def prepare_cnn_input(self, feature_dict: Dict[str, Any]) -> "np.ndarray":
        """
        Prepare the fully-encoded, ordered CNN input tensor from a raw
        base feature dictionary.

        This is the single public entry point for CNN input preparation.
        It enforces the exact feature ordering used to train the CNN
        spatial branch, defensively encodes any not-yet-encoded
        categorical value, and reshapes the result into the tensor
        format expected by the CNN feature extractor in hybrid_model.py.
        No inference, risk scoring, or model invocation happens here.

        Parameters
        ----------
        feature_dict : Dict[str, Any]
            The raw feature dictionary produced by
            `prepare_base_features()`. Values are sourced from the
            engineered dataset (via `get_latest_record()` /
            `DatasetLoader.prepare_prediction_record()`), which is
            already MinMax-scaled at the source -- see the module-level
            "Scaling note" above.

        Returns
        -------
        np.ndarray
            A float32 tensor of shape (1, n_cnn_features, 1), ready to be
            passed directly to the CNN feature extractor.

        Raises
        ------
        PredictionError
            If required feature columns are missing, categorical
            encoding fails, or a value cannot be converted to a numeric
            type.
        """
        cnn_columns = self._get_cnn_feature_columns()

        ordered_values: List[float] = []
        missing_columns: List[str] = []

        for column in cnn_columns:
            if column not in feature_dict:
                missing_columns.append(column)
                continue
            ordered_values.append(
                self._encode_feature_value(column, feature_dict[column])
            )

        if missing_columns:
            raise PredictionError(
                "Base feature dictionary is missing required CNN feature "
                f"column(s): {missing_columns}."
            )

        # NOTE: No scaler.transform() call here. feature_engineering.py
        # runs scale_numerical_features() BEFORE save_feature_dataset(),
        # so the engineered CSV -- and therefore every value in
        # `feature_dict` -- is already MinMax-scaled. Re-applying the
        # scaler here would scale an already-scaled record a second time
        # and silently corrupt the CNN input.
        try:
            feature_vector = np.array(ordered_values, dtype=np.float32).reshape(1, -1)
        except (TypeError, ValueError) as exc:
            logger.error("Failed to build CNN feature vector: %s", exc)
            raise PredictionError(
                f"Unable to build CNN feature vector: {exc}"
            ) from exc

        cnn_tensor = self.build_cnn_tensor(feature_vector)

        logger.info(
            "Prepared CNN input tensor with shape %s from %d feature(s).",
            cnn_tensor.shape, len(cnn_columns),
        )
        return cnn_tensor

    def build_cnn_tensor(self, feature_vector: "np.ndarray") -> "np.ndarray":
        """
        Reshape a 2D feature vector into the 3D tensor format expected by
        the CNN feature extractor (Conv1D input).

        Parameters
        ----------
        feature_vector : np.ndarray
            A 2D array of shape (n_samples, n_cnn_features), already in
            the model's expected numeric scale.

        Returns
        -------
        np.ndarray
            A float32 tensor of shape (n_samples, n_cnn_features, 1),
            matching the (samples, timesteps, channels=1) layout used by
            the CNN's Conv1D input, as constructed in cnn_model.py /
            hybrid_model.py.

        Raises
        ------
        PredictionError
            If `feature_vector` is not a 2D array.
        """
        if feature_vector.ndim != 2:
            raise PredictionError(
                "Expected a 2D feature vector (n_samples, n_features), "
                f"got array with shape {feature_vector.shape}."
            )

        n_samples, n_features = feature_vector.shape
        return feature_vector.reshape(n_samples, n_features, 1).astype(np.float32)

    # ----------------------------------------------------------------
    # CNN preparation helpers
    # ----------------------------------------------------------------
    def _get_cnn_feature_columns(self) -> List[str]:
        """
        Resolve the ordered list of feature columns the CNN spatial
        branch consumes.

        Starts from `_get_ordered_feature_columns()` (Part 2), which
        already excludes identifier and target columns per
        feature_columns.json order, and further excludes YEAR /
        YEAR_INDEX to mirror CNN_EXCLUDED_COLUMNS in hybrid_model.py /
        CNNConfig.columns_to_exclude in cnn_model.py.

        Returns
        -------
        List[str]
            Ordered CNN feature column names.
        """
        ordered_feature_columns = self._get_ordered_feature_columns()
        return [
            column for column in ordered_feature_columns
            if column not in self._CNN_NON_SPATIAL_COLUMNS
        ]

    def _encode_feature_value(self, column: str, value: Any) -> float:
        """
        Return the numeric value to use for `column`, encoding it with
        the matching fitted LabelEncoder from ModelLoader if `column`
        holds a categorical value that has not already been encoded.

        Parameters
        ----------
        column : str
            Feature column name.
        value : Any
            Raw value from the base feature dictionary. Expected to
            already be numeric (the engineered dataset stores
            STATE_ENCODED / DISTRICT_ENCODED pre-encoded and
            pre-scaled); a string is only encountered in edge cases and
            is encoded here rather than passed through downstream as-is.

        Returns
        -------
        float
            The numeric (encoded, if applicable) feature value.

        Raises
        ------
        PredictionError
            If a categorical value cannot be encoded (e.g. unseen label)
            or a numeric value cannot be coerced to float.
        """
        encoder_getter_name = self._CATEGORICAL_COLUMN_ENCODERS.get(column)

        if encoder_getter_name is not None and isinstance(value, str):
            try:
                encoder = getattr(self.model_loader, encoder_getter_name)()
                encoded_value = encoder.transform([value.strip().upper()])[0]
                return float(encoded_value)
            except Exception as exc:
                logger.error(
                    "Failed to encode categorical value for column "
                    "'%s' (value='%s'): %s", column, value, exc,
                )
                raise PredictionError(
                    f"Unable to encode value for '{column}': {exc}"
                ) from exc

        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            logger.error(
                "Non-numeric, non-categorical value for column '%s': %s",
                column, value,
            )
            raise PredictionError(
                f"Feature column '{column}' has a non-numeric value "
                f"'{value}' that could not be converted for the CNN "
                "input."
            ) from exc

    # ----------------------------------------------------------------
    # LSTM input preparation
    # ----------------------------------------------------------------

    # Number of consecutive years of history the LSTM branch consumes per
    # sample. Must match SEQUENCE_LENGTH in lstm_model.py exactly, since
    # the pretrained LSTM feature extractor's Input layer is fixed to
    # this timestep count.
    _LSTM_SEQUENCE_LENGTH: int = 3

    # Columns present in the base feature-column list (see Part 2's
    # `_get_ordered_feature_columns`) that the LSTM temporal branch does
    # NOT consume. This mirrors NON_FEATURE_COLUMNS in lstm_model.py /
    # LSTM_EXCLUDED_COLUMNS in hybrid_model.py: YEAR is dropped because
    # the raw calendar year carries no information beyond what
    # YEAR_INDEX (which IS kept) already encodes, and Id / STATE/UT /
    # DISTRICT / the target column are already excluded upstream by
    # `_get_ordered_feature_columns`.
    _LSTM_NON_TEMPORAL_COLUMNS: frozenset = frozenset({"YEAR"})

    def prepare_lstm_input(self, state: str, district: str) -> "np.ndarray":
        """
        Prepare the fully-ordered LSTM input tensor for a validated
        state/district, mirroring the sequence construction used to
        train the LSTM temporal branch.

        This is the single public entry point for LSTM input
        preparation. It retrieves historical records via the existing
        `get_historical_records()` (Part 2), builds a chronologically
        ordered feature sequence, and reshapes it into the tensor shape
        expected by the pretrained LSTM feature extractor. No inference,
        risk scoring, or model invocation happens here.

        Parameters
        ----------
        state : str
            Already-validated, normalized state name.
        district : str
            Already-validated, normalized district name.

        Returns
        -------
        np.ndarray
            A float32 tensor of shape
            (1, `_LSTM_SEQUENCE_LENGTH`, n_lstm_features), ready to be
            passed directly to the LSTM feature extractor.

        Raises
        ------
        PredictionError
            If historical records cannot be retrieved, there are fewer
            than `_LSTM_SEQUENCE_LENGTH` historical records available,
            or any required feature column is missing / non-numeric.
        """
        historical_records = self.get_historical_records(state, district)

        sequence = self.build_lstm_sequence(historical_records)
        lstm_tensor = self.build_lstm_tensor(sequence)

        logger.info(
            "Prepared LSTM input tensor with shape %s for state='%s', "
            "district='%s'.", lstm_tensor.shape, state, district,
        )
        return lstm_tensor

    def build_lstm_sequence(self, historical_records: "pd.DataFrame") -> "np.ndarray":
        """
        Build the chronologically ordered 2D feature sequence the LSTM
        branch expects, from a state/district's historical records.

        Takes the most recent `_LSTM_SEQUENCE_LENGTH` records (oldest
        first, newest last, matching `create_sequences()` in
        lstm_model.py, which sorts each group by YEAR_INDEX ascending
        before slicing windows) and selects exactly the LSTM feature
        columns, in the LSTM branch's training column order.

        Parameters
        ----------
        historical_records : pd.DataFrame
            Historical records for a single state/district, as returned
            by `get_historical_records()` (Part 2), already sorted by
            YEAR ascending.

        Returns
        -------
        np.ndarray
            A float32 array of shape (`_LSTM_SEQUENCE_LENGTH`,
            n_lstm_features), oldest record first.

        Raises
        ------
        PredictionError
            If fewer than `_LSTM_SEQUENCE_LENGTH` historical records are
            available, or a required LSTM feature column is missing or
            cannot be converted to a numeric type.
        """
        if len(historical_records) < self._LSTM_SEQUENCE_LENGTH:
            raise PredictionError(
                "Insufficient historical data to build an LSTM input "
                f"sequence: found {len(historical_records)} record(s), "
                f"need at least {self._LSTM_SEQUENCE_LENGTH}."
            )

        # Oldest-first ordering is already guaranteed by
        # get_historical_records() (sorted by YEAR ascending); take the
        # most recent window of the required length.
        window = historical_records.tail(self._LSTM_SEQUENCE_LENGTH)

        lstm_columns = self._get_lstm_feature_columns()
        missing_columns = [
            column for column in lstm_columns if column not in window.columns
        ]
        if missing_columns:
            raise PredictionError(
                "Historical records are missing required LSTM feature "
                f"column(s): {missing_columns}."
            )

        try:
            sequence = window[lstm_columns].to_numpy(dtype=np.float32)
        except (TypeError, ValueError) as exc:
            logger.error("Failed to build LSTM sequence array: %s", exc)
            raise PredictionError(
                f"Unable to build LSTM input sequence: {exc}"
            ) from exc

        logger.info(
            "Built LSTM sequence with shape %s (%d timesteps, %d "
            "feature(s) each).", sequence.shape, sequence.shape[0],
            sequence.shape[1],
        )
        return sequence

    def build_lstm_tensor(self, sequence: "np.ndarray") -> "np.ndarray":
        """
        Reshape a 2D chronological feature sequence into the 3D batch
        tensor format expected by the LSTM feature extractor.

        Parameters
        ----------
        sequence : np.ndarray
            A 2D array of shape (`_LSTM_SEQUENCE_LENGTH`,
            n_lstm_features), oldest record first, already in the
            model's expected numeric scale.

        Returns
        -------
        np.ndarray
            A float32 tensor of shape (1, `_LSTM_SEQUENCE_LENGTH`,
            n_lstm_features), matching the (batch, timesteps, features)
            input layout `build_model()` in lstm_model.py was trained
            on.

        Raises
        ------
        PredictionError
            If `sequence` is not a 2D array, or its first dimension does
            not equal `_LSTM_SEQUENCE_LENGTH`.
        """
        if sequence.ndim != 2:
            raise PredictionError(
                "Expected a 2D LSTM sequence (timesteps, n_features), "
                f"got array with shape {sequence.shape}."
            )

        if sequence.shape[0] != self._LSTM_SEQUENCE_LENGTH:
            raise PredictionError(
                "Expected an LSTM sequence with "
                f"{self._LSTM_SEQUENCE_LENGTH} timesteps, got "
                f"{sequence.shape[0]}."
            )

        n_timesteps, n_features = sequence.shape
        return sequence.reshape(1, n_timesteps, n_features).astype(np.float32)

    # ----------------------------------------------------------------
    # LSTM preparation helpers
    # ----------------------------------------------------------------
    def _get_lstm_feature_columns(self) -> List[str]:
        """
        Resolve the ordered list of feature columns the LSTM temporal
        branch consumes.

        Starts from `_get_ordered_feature_columns()` (Part 2), which
        already excludes identifier and target columns per
        feature_columns.json order, and further excludes YEAR to mirror
        NON_FEATURE_COLUMNS in lstm_model.py / LSTM_EXCLUDED_COLUMNS in
        hybrid_model.py. YEAR_INDEX is intentionally kept, matching both
        training files.

        Returns
        -------
        List[str]
            Ordered LSTM feature column names.
        """
        ordered_feature_columns = self._get_ordered_feature_columns()
        return [
            column for column in ordered_feature_columns
            if column not in self._LSTM_NON_TEMPORAL_COLUMNS
        ]

    # ----------------------------------------------------------------
    # Embedding dimensions (must match hybrid_model.py exactly)
    # ----------------------------------------------------------------
    _CNN_EMBEDDING_DIM: int = 64
    _LSTM_EMBEDDING_DIM: int = 32
    _FUSED_EMBEDDING_DIM: int = _CNN_EMBEDDING_DIM + _LSTM_EMBEDDING_DIM  # 96

    # ----------------------------------------------------------------
    # Embedding extraction
    # ----------------------------------------------------------------
    def extract_cnn_embedding(self, cnn_tensor: "np.ndarray") -> "np.ndarray":
        """
        Run the pretrained CNN spatial branch's feature extractor on a
        prepared CNN input tensor and return its embedding.

        Parameters
        ----------
        cnn_tensor : np.ndarray
            The tensor produced by `prepare_cnn_input()` (Part 3), of
            shape (1, n_cnn_features, 1).

        Returns
        -------
        np.ndarray
            The CNN spatial embedding, of shape
            (1, `_CNN_EMBEDDING_DIM`).

        Raises
        ------
        PredictionError
            If the CNN feature extractor cannot be obtained or run, or
            if the resulting embedding does not have the expected
            shape.
        """
        try:
            cnn_feature_extractor = self.model_loader.get_cnn_feature_extractor()
        except ModelLoaderError as exc:
            logger.error("Failed to obtain CNN feature extractor: %s", exc)
            raise PredictionError(
                f"Unable to obtain CNN feature extractor: {exc}"
            ) from exc

        try:
            cnn_embedding = cnn_feature_extractor(cnn_tensor, training=False).numpy()
        except Exception:
            try:
                cnn_embedding = cnn_feature_extractor.predict(cnn_tensor, verbose=0)
            except Exception as exc:
                logger.error("CNN feature extractor inference failed: %s", exc)
                raise PredictionError(
                    f"CNN feature extractor inference failed: {exc}"
                ) from exc

        self._validate_embedding_shape(
            cnn_embedding, expected_dim=self._CNN_EMBEDDING_DIM, name="CNN"
        )

        logger.info(
            "Extracted CNN embedding with shape %s.", cnn_embedding.shape
        )
        return cnn_embedding

    def extract_lstm_embedding(self, lstm_tensor: "np.ndarray") -> "np.ndarray":
        """
        Run the pretrained LSTM temporal branch's feature extractor on
        a prepared LSTM input tensor and return its embedding.

        Parameters
        ----------
        lstm_tensor : np.ndarray
            The tensor produced by `prepare_lstm_input()` (Part 4), of
            shape (1, `_LSTM_SEQUENCE_LENGTH`, n_lstm_features).

        Returns
        -------
        np.ndarray
            The LSTM temporal embedding, of shape
            (1, `_LSTM_EMBEDDING_DIM`).

        Raises
        ------
        PredictionError
            If the LSTM feature extractor cannot be obtained or run, or
            if the resulting embedding does not have the expected
            shape.
        """
        try:
            lstm_feature_extractor = self.model_loader.get_lstm_feature_extractor()
        except ModelLoaderError as exc:
            logger.error("Failed to obtain LSTM feature extractor: %s", exc)
            raise PredictionError(
                f"Unable to obtain LSTM feature extractor: {exc}"
            ) from exc

        try:
            lstm_embedding = lstm_feature_extractor(lstm_tensor, training=False).numpy()
        except Exception:
            try:
                lstm_embedding = lstm_feature_extractor.predict(lstm_tensor, verbose=0)
            except Exception as exc:
                logger.error("LSTM feature extractor inference failed: %s", exc)
                raise PredictionError(
                    f"LSTM feature extractor inference failed: {exc}"
                ) from exc

        self._validate_embedding_shape(
            lstm_embedding, expected_dim=self._LSTM_EMBEDDING_DIM, name="LSTM"
        )

        logger.info(
            "Extracted LSTM embedding with shape %s.", lstm_embedding.shape
        )
        return lstm_embedding

    # ----------------------------------------------------------------
    # Embedding fusion
    # ----------------------------------------------------------------
    def fuse_embeddings(
        self, cnn_embedding: "np.ndarray", lstm_embedding: "np.ndarray"
    ) -> "np.ndarray":
        """
        Concatenate the CNN spatial embedding and LSTM temporal
        embedding into the single fused representation the hybrid
        model consumes.

        Mirrors `np.concatenate([cnn_embeddings, lstm_embeddings],
        axis=1)` in `extract_features()` (hybrid_model.py) exactly:
        both inputs must be batch-first 2D arrays sharing the same
        batch size, and the CNN embedding must come first.

        Parameters
        ----------
        cnn_embedding : np.ndarray
            CNN spatial embedding, shape
            (batch_size, `_CNN_EMBEDDING_DIM`).
        lstm_embedding : np.ndarray
            LSTM temporal embedding, shape
            (batch_size, `_LSTM_EMBEDDING_DIM`).

        Returns
        -------
        np.ndarray
            The fused embedding, shape
            (batch_size, `_FUSED_EMBEDDING_DIM`), with the CNN
            embedding occupying the first `_CNN_EMBEDDING_DIM` columns
            and the LSTM embedding occupying the remaining
            `_LSTM_EMBEDDING_DIM` columns.

        Raises
        ------
        PredictionError
            If either embedding is not a batch-first 2D array of the
            expected dimensionality, if their batch sizes disagree, or
            if concatenation fails.
        """
        self._validate_embedding_shape(
            cnn_embedding, expected_dim=self._CNN_EMBEDDING_DIM, name="CNN"
        )
        self._validate_embedding_shape(
            lstm_embedding, expected_dim=self._LSTM_EMBEDDING_DIM, name="LSTM"
        )

        if cnn_embedding.shape[0] != lstm_embedding.shape[0]:
            raise PredictionError(
                "CNN and LSTM embeddings have mismatched batch sizes: "
                f"{cnn_embedding.shape[0]} vs {lstm_embedding.shape[0]}."
            )

        try:
            fused_embedding = np.concatenate(
                [cnn_embedding, lstm_embedding], axis=1
            ).astype(np.float32)
        except Exception as exc:
            logger.error("Failed to fuse CNN and LSTM embeddings: %s", exc)
            raise PredictionError(
                f"Unable to fuse CNN and LSTM embeddings: {exc}"
            ) from exc

        if fused_embedding.shape[1] != self._FUSED_EMBEDDING_DIM:
            raise PredictionError(
                "Fused embedding has unexpected dimensionality: got "
                f"{fused_embedding.shape[1]}, expected "
                f"{self._FUSED_EMBEDDING_DIM} "
                f"({self._CNN_EMBEDDING_DIM} CNN + "
                f"{self._LSTM_EMBEDDING_DIM} LSTM)."
            )

        logger.info(
            "Fused embeddings into shape %s (%d CNN + %d LSTM).",
            fused_embedding.shape, self._CNN_EMBEDDING_DIM,
            self._LSTM_EMBEDDING_DIM,
        )
        return fused_embedding

    # ----------------------------------------------------------------
    # Hybrid model inference
    # ----------------------------------------------------------------
    def predict_hybrid(self, fused_embedding: "np.ndarray") -> float:
        """
        Run the pretrained hybrid fusion model on a fused embedding and
        return the predicted crime count.

        Parameters
        ----------
        fused_embedding : np.ndarray
            The fused embedding produced by `fuse_embeddings()`, of
            shape (1, `_FUSED_EMBEDDING_DIM`).

        Returns
        -------
        float
            The predicted crime count, as a plain Python float. No
            risk scoring, confidence calculation, or response
            formatting is performed here.

        Raises
        ------
        PredictionError
            If `fused_embedding` has an unexpected shape, if the
            hybrid model cannot be obtained or run, or if its output
            cannot be interpreted as a single scalar prediction.
        """
        if (
            not isinstance(fused_embedding, np.ndarray)
            or fused_embedding.ndim != 2
            or fused_embedding.shape[1] != self._FUSED_EMBEDDING_DIM
        ):
            raise PredictionError(
                "Expected a fused embedding of shape "
                f"(batch_size, {self._FUSED_EMBEDDING_DIM}), got "
                f"{getattr(fused_embedding, 'shape', type(fused_embedding))}."
            )

        try:
            hybrid_model = self.model_loader.get_model()
        except ModelLoaderError as exc:
            logger.error("Failed to obtain hybrid model: %s", exc)
            raise PredictionError(
                f"Unable to obtain hybrid model: {exc}"
            ) from exc

        try:
            raw_prediction = hybrid_model(fused_embedding, training=False).numpy()
        except Exception:
            try:
                raw_prediction = hybrid_model.predict(fused_embedding, verbose=0)
            except Exception as exc:
                logger.error("Hybrid model inference failed: %s", exc)
                raise PredictionError(
                    f"Hybrid model inference failed: {exc}"
                ) from exc

        try:
            predicted_value = float(np.asarray(raw_prediction).reshape(-1)[0])
        except (TypeError, ValueError, IndexError) as exc:
            logger.error(
                "Unable to extract a scalar prediction from hybrid "
                "model output %s: %s", raw_prediction, exc,
            )
            raise PredictionError(
                "Unable to interpret hybrid model output as a single "
                f"predicted crime count: {exc}"
            ) from exc

        logger.info(
            "Hybrid model predicted crime count: %.4f", predicted_value
        )
        return predicted_value

    # ----------------------------------------------------------------
    # GCN Spatial-Adjacency branch & 3-Way Fusion methods
    # ----------------------------------------------------------------
    _GCN_EMBEDDING_DIM: int = 32
    _FUSED_3WAY_DIMENSION: int = 128
    _MODEL_NAME_GCN: str = "Hybrid CNN-LSTM-GCN"

    def prepare_gcn_input(
        self,
        state: str,
        district: str,
        base_features: Dict[str, Any],
        prediction_year: int,
    ) -> Tuple["np.ndarray", int]:
        """
        Construct the full R^(1 x 850 x 33) annual node feature matrix for GCN inference.
        Populates all 850 nodes with the latest available baseline year (2013) features
        across India, and updates the target district's slice with the user's prediction
        inputs and temporal indicator (YEAR_INDEX).

        Returns
        -------
        Tuple[np.ndarray, int]
            - Float32 tensor of shape (1, 850, n_features)
            - Target district node integer index (0 <= idx < 850)
        """
        adjacency_data = self.model_loader.get_adjacency()
        node_df = adjacency_data["node_df"]
        feature_cols = adjacency_data["feature_columns"]
        node_to_idx = adjacency_data["node_to_idx"]

        target_key = (state.strip().upper(), district.strip().upper())
        if target_key not in node_to_idx:
            raise PredictionError(
                f"District '{district}' in state '{state}' was not found in the spatial graph nodes."
            )
        target_node_idx = node_to_idx[target_key]

        # Lazy cache the 2013 all-district baseline feature matrix (850 x 33)
        if self._cached_baseline_features is None:
            df = self.dataset_loader.get_dataframe()
            if df is None or df.empty:
                raise PredictionError("Dataset not available to construct GCN baseline features.")

            latest_year_idx = int(df["YEAR_INDEX"].max())
            latest_df = df[df["YEAR_INDEX"] == latest_year_idx]

            num_nodes = len(node_df)
            num_features = len(feature_cols)
            baseline_mat = np.zeros((num_nodes, num_features), dtype=np.float32)

            for _, row in latest_df.iterrows():
                st = str(row["STATE/UT"]).strip().upper()
                dt = str(row["DISTRICT"]).strip().upper()
                n_idx = node_to_idx.get((st, dt))
                if n_idx is not None:
                    baseline_mat[n_idx] = row[feature_cols].to_numpy(dtype=np.float32)

            self._cached_baseline_features = baseline_mat
            self._cached_baseline_year = latest_year_idx

        # Copy baseline (850, 33)
        gcn_tensor = np.copy(self._cached_baseline_features)

        # Update target node's feature slice with current prepared base_features
        target_vector = np.array(
            [self._encode_feature_value(col, base_features.get(col, 0.0)) for col in feature_cols],
            dtype=np.float32,
        )
        gcn_tensor[target_node_idx] = target_vector

        # Reshape to batch-first (1, 850, 33)
        gcn_tensor_batched = np.expand_dims(gcn_tensor, axis=0)
        return gcn_tensor_batched, target_node_idx

    def extract_gcn_embedding(
        self, gcn_input: "np.ndarray", target_node_idx: int
    ) -> "np.ndarray":
        """
        Passes the full 850-node graph through the GCN feature extractor and extracts
        the target district's 32-dim spatial-adjacency embedding.
        """
        gcn_extractor = self.model_loader.get_gcn_feature_extractor()
        try:
            full_embeddings = gcn_extractor(gcn_input, training=False).numpy()
        except Exception:
            full_embeddings = gcn_extractor.predict(gcn_input, verbose=0)
        target_embedding = full_embeddings[0, target_node_idx : target_node_idx + 1, :]
        return target_embedding.astype(np.float32)

    def fuse_3way_embeddings(
        self,
        cnn_embedding: "np.ndarray",
        lstm_embedding: "np.ndarray",
        gcn_embedding: "np.ndarray",
    ) -> "np.ndarray":
        """
        Concatenates CNN (64), LSTM (32), and GCN (32) embeddings into a 128-dim fused vector.
        """
        self._validate_embedding_shape(cnn_embedding, self._CNN_EMBEDDING_DIM, "CNN")
        self._validate_embedding_shape(lstm_embedding, self._LSTM_EMBEDDING_DIM, "LSTM")
        self._validate_embedding_shape(gcn_embedding, self._GCN_EMBEDDING_DIM, "GCN")

        fused = np.concatenate([cnn_embedding, lstm_embedding, gcn_embedding], axis=1)
        return fused.astype(np.float32)

    def predict_hybrid_gcn(self, fused_embedding: "np.ndarray") -> float:
        """
        Runs inference through the 3-Way Hybrid CNN-LSTM-GCN model (128-dim input).
        """
        self._validate_embedding_shape(fused_embedding, self._FUSED_3WAY_DIMENSION, "3-Way Fused")
        model = self.model_loader.get_hybrid_gcn_model()
        try:
            raw_pred = model(fused_embedding, training=False).numpy()
        except Exception:
            raw_pred = model.predict(fused_embedding, verbose=0)
        return float(np.asarray(raw_pred).reshape(-1)[0])

    # ----------------------------------------------------------------
    # Embedding validation helper
    # ----------------------------------------------------------------
    @staticmethod
    def _validate_embedding_shape(
        embedding: "np.ndarray", expected_dim: int, name: str
    ) -> None:
        """
        Validate that an embedding is a batch-first 2D array with the
        expected feature dimensionality.

        Parameters
        ----------
        embedding : np.ndarray
            The embedding array to validate.
        expected_dim : int
            The expected size of the embedding's second axis (feature
            dimension).
        name : str
            Human-readable name of the embedding (e.g. "CNN", "LSTM"),
            used in error messages.

        Raises
        ------
        PredictionError
            If `embedding` is not a 2D array, or its second axis does
            not equal `expected_dim`.
        """
        if not isinstance(embedding, np.ndarray) or embedding.ndim != 2:
            actual = (
                embedding.shape if isinstance(embedding, np.ndarray)
                else type(embedding)
            )
            raise PredictionError(
                f"Expected a batch-first 2D {name} embedding of shape "
                f"(batch_size, {expected_dim}), got {actual}."
            )

        if embedding.shape[1] != expected_dim:
            raise PredictionError(
                f"{name} embedding has unexpected dimensionality: got "
                f"{embedding.shape[1]}, expected {expected_dim}."
            )

    # ----------------------------------------------------------------
    # Risk-level thresholds (class constants â€” see calculate_risk_level)
    # ----------------------------------------------------------------
    # Predicted crime count is compared against these upper bounds, in
    # ascending order, to classify risk. A predicted count <=
    # _RISK_LOW_MAX is "Low"; <= _RISK_MODERATE_MAX is "Moderate"; <=
    # _RISK_HIGH_MAX is "High"; anything above that is "Very High".
    #
    # These are deliberately kept as tunable class constants (rather
    # than inlined in calculate_risk_level) so they can be recalibrated
    # against the dataset's actual crime-count distribution without
    # touching any method body.
    _RISK_LOW_MAX: float = 50.0
    _RISK_MODERATE_MAX: float = 150.0
    _RISK_HIGH_MAX: float = 300.0

    _RISK_LEVEL_LOW: str = "Low"
    _RISK_LEVEL_MODERATE: str = "Moderate"
    _RISK_LEVEL_HIGH: str = "High"
    _RISK_LEVEL_VERY_HIGH: str = "Very High"

    # ----------------------------------------------------------------
    # Confidence heuristic constants (class constants â€” see
    # calculate_confidence)
    # ----------------------------------------------------------------
    # The heuristic is deliberately simple and fully deterministic: it
    # is NOT a model-derived probability (the hybrid model has no
    # softmax/uncertainty head), but a transparent proxy for how much
    # the pipeline is being asked to extrapolate.
    #
    #   confidence = _CONFIDENCE_BASE
    #                - (years_ahead * _CONFIDENCE_YEAR_PENALTY)
    #                + min(extra_history * _CONFIDENCE_HISTORY_BONUS_PER_RECORD,
    #                      _CONFIDENCE_HISTORY_BONUS_CAP)
    #
    # clamped to [_CONFIDENCE_MIN, _CONFIDENCE_MAX].
    #
    # - years_ahead: prediction_year - latest_dataset_year. Every extra
    #   year beyond the last observed data point is a year of pure
    #   extrapolation with no ground truth to anchor it, so confidence
    #   is penalized per year.
    # - extra_history: historical records available beyond the minimum
    #   the LSTM branch requires (_LSTM_SEQUENCE_LENGTH, Part 4). More
    #   history for a state/district means the LSTM sequence is drawn
    #   from a more established trend, so a small bonus is awarded
    #   (capped, so a district with 13 years of history isn't treated
    #   as dramatically more reliable than one with 6).
    _CONFIDENCE_BASE: float = 95.0
    _CONFIDENCE_YEAR_PENALTY: float = 6.0
    _CONFIDENCE_HISTORY_BONUS_PER_RECORD: float = 0.5
    _CONFIDENCE_HISTORY_BONUS_CAP: float = 5.0
    _CONFIDENCE_MIN: float = 40.0
    _CONFIDENCE_MAX: float = 99.0

    _MODEL_NAME: str = "Hybrid CNN-LSTM"

    # ----------------------------------------------------------------
    # Risk-level classification
    # ----------------------------------------------------------------
    def calculate_risk_level(self, predicted_crime_count: float) -> str:
        """
        Classify a predicted crime count into a discrete risk level.

        Uses the ascending, mutually-exclusive thresholds defined as
        class constants (`_RISK_LOW_MAX`, `_RISK_MODERATE_MAX`,
        `_RISK_HIGH_MAX`) rather than hardcoded values, so thresholds
        can be recalibrated without touching this method.

        Parameters
        ----------
        predicted_crime_count : float
            The predicted crime count returned by `predict_hybrid()`
            (Part 5).

        Returns
        -------
        str
            One of "Low", "Moderate", "High", "Very High".

        Raises
        ------
        PredictionError
            If `predicted_crime_count` is negative or not a real
            number.
        """
        try:
            count = max(0.0, float(predicted_crime_count))
        except (TypeError, ValueError) as exc:
            raise PredictionError(
                f"Unable to classify risk level: invalid predicted "
                f"crime count '{predicted_crime_count}': {exc}"
            ) from exc

        if count <= self._RISK_LOW_MAX:
            risk_level = self._RISK_LEVEL_LOW
        elif count <= self._RISK_MODERATE_MAX:
            risk_level = self._RISK_LEVEL_MODERATE
        elif count <= self._RISK_HIGH_MAX:
            risk_level = self._RISK_LEVEL_HIGH
        else:
            risk_level = self._RISK_LEVEL_VERY_HIGH

        logger.info(
            "Classified predicted crime count %.4f as risk level '%s'.",
            count, risk_level,
        )
        return risk_level

    # ----------------------------------------------------------------
    # Confidence estimation
    # ----------------------------------------------------------------
    def calculate_confidence(
        self,
        prediction_year: int,
        latest_dataset_year: int,
        historical_record_count: int,
    ) -> float:
        """
        Estimate a deterministic confidence score for a prediction.

        This is a transparent heuristic over the prediction pipeline's
        own inputs (how far ahead the request extrapolates, and how
        much historical data backed the LSTM sequence) â€” it is NOT a
        model-reported probability, since the hybrid model exposes no
        uncertainty head, and it uses no randomness.

        Parameters
        ----------
        prediction_year : int
            The validated target year being predicted for.
        latest_dataset_year : int
            The most recent year present in the dataset for the
            requested state/district (from `validate_input()`, Part 1).
        historical_record_count : int
            Number of historical records available for the requested
            state/district (from `get_historical_records()`, Part 2).

        Returns
        -------
        float
            Confidence score in the inclusive range
            [`_CONFIDENCE_MIN`, `_CONFIDENCE_MAX`].

        Raises
        ------
        PredictionError
            If `prediction_year` is not strictly greater than
            `latest_dataset_year`, or `historical_record_count` is
            negative.
        """
        years_ahead = prediction_year - latest_dataset_year
        if years_ahead <= 0:
            raise PredictionError(
                "Unable to calculate confidence: 'prediction_year' "
                f"({prediction_year}) must be strictly greater than "
                f"'latest_dataset_year' ({latest_dataset_year})."
            )

        if historical_record_count < 0:
            raise PredictionError(
                "Unable to calculate confidence: "
                "'historical_record_count' cannot be negative "
                f"(got {historical_record_count})."
            )

        year_penalty = years_ahead * self._CONFIDENCE_YEAR_PENALTY

        extra_history = max(
            0, historical_record_count - self._LSTM_SEQUENCE_LENGTH
        )
        history_bonus = min(
            extra_history * self._CONFIDENCE_HISTORY_BONUS_PER_RECORD,
            self._CONFIDENCE_HISTORY_BONUS_CAP,
        )

        raw_confidence = self._CONFIDENCE_BASE - year_penalty + history_bonus
        confidence = max(
            self._CONFIDENCE_MIN, min(self._CONFIDENCE_MAX, raw_confidence)
        )

        logger.info(
            "Calculated confidence %.2f (years_ahead=%d, "
            "history_records=%d, extra_history=%d).",
            confidence, years_ahead, historical_record_count, extra_history,
        )
        return float(confidence)

    # ----------------------------------------------------------------
    # Main orchestration entry point
    # ----------------------------------------------------------------
    def predict(
        self,
        state: str,
        district: str,
        prediction_year: Any,
        model_type: str = "hybrid_gcn",
    ) -> Dict[str, Any]:
        """
        Run the full end-to-end crime prediction pipeline for a single
        state/district/year request.

        Supports both:
          - "hybrid_gcn" (Default 3-Way: CNN 64 + LSTM 32 + GCN 32 = 128-dim fused embedding)
          - "hybrid" / "hybrid_2way" (Legacy 2-Way: CNN 64 + LSTM 32 = 96-dim fused embedding)

        Parameters
        ----------
        state : str
            State/UT name to predict for.
        district : str
            District name to predict for.
        prediction_year : Any
            The year to generate a prediction for.
        model_type : str, default "hybrid_gcn"
            Model architecture to use ("hybrid_gcn" or "hybrid").

        Returns
        -------
        Dict[str, Any]
            {
                "state": str,
                "district": str,
                "prediction_year": int,
                "predicted_crime_count": float,
                "risk_level": str,
                "confidence": float,
                "latest_dataset_year": int,
                "model": str,
                "status": "success",
            }
        """
        start_time = time.perf_counter()
        logger.info(
            "Prediction started: state='%s', district='%s', "
            "prediction_year=%s, model_type='%s'.",
            state, district, prediction_year, model_type,
        )

        try:
            # 1. Validate input.
            validated = self.validate_input(state, district, prediction_year)
            validated_state = validated["state"]
            validated_district = validated["district"]
            validated_year = validated["prediction_year"]
            latest_dataset_year = validated["latest_dataset_year"]

            logger.info(
                "Validation successful for state='%s', district='%s', "
                "prediction_year=%d.",
                validated_state, validated_district, validated_year,
            )

            # 2. Retrieve the latest historical record.
            latest_record = self.get_latest_record(
                validated_state, validated_district
            )

            # 3. Build the raw base feature dictionary.
            base_features = self.prepare_base_features(latest_record)

            # 4. Prepare CNN input tensor.
            cnn_input = self.prepare_cnn_input(base_features)

            # 5. Prepare LSTM input tensor.
            lstm_input = self.prepare_lstm_input(
                validated_state, validated_district
            )

            # 6. Extract CNN spatial embedding (64-dim).
            cnn_embedding = self.extract_cnn_embedding(cnn_input)

            # 7. Extract LSTM temporal embedding (32-dim).
            lstm_embedding = self.extract_lstm_embedding(lstm_input)

            predicted_crime_count: Optional[float] = None
            model_name = self._MODEL_NAME

            # 8. Dual-Mode Branch Selection (3-Way GCN vs Legacy 2-Way)
            if model_type.lower() in ("hybrid_gcn", "hybrid_v2", "gcn"):
                try:
                    # 8a. Prepare full 850-node feature tensor and extract GCN embedding (32-dim)
                    gcn_input, target_node_idx = self.prepare_gcn_input(
                        validated_state, validated_district, base_features, validated_year
                    )
                    gcn_embedding = self.extract_gcn_embedding(gcn_input, target_node_idx)

                    # 8b. Fuse 3-way embeddings: 64 (CNN) + 32 (LSTM) + 32 (GCN) = 128-dim
                    fused_128 = self.fuse_3way_embeddings(cnn_embedding, lstm_embedding, gcn_embedding)

                    # 8c. Run 3-way hybrid inference
                    predicted_crime_count = self.predict_hybrid_gcn(fused_128)
                    model_name = self._MODEL_NAME_GCN
                except Exception as gcn_exc:
                    logger.warning(
                        "GCN 3-way inference encountered an issue (%s); falling back to 2-way Hybrid CNN-LSTM.",
                        gcn_exc,
                    )
                    predicted_crime_count = None

            if predicted_crime_count is None:
                # 2-Way Baseline Path: 64 (CNN) + 32 (LSTM) = 96-dim
                fused_embedding = self.fuse_embeddings(cnn_embedding, lstm_embedding)
                predicted_crime_count = self.predict_hybrid(fused_embedding)
                model_name = self._MODEL_NAME

            predicted_crime_count = max(0.0, float(predicted_crime_count))

            # 10. Classify risk level.
            risk_level = self.calculate_risk_level(predicted_crime_count)

            # 11. Estimate confidence.
            historical_record_count = len(
                self.get_historical_records(validated_state, validated_district)
            )
            confidence = self.calculate_confidence(
                validated_year, latest_dataset_year, historical_record_count
            )

        except PredictionError as exc:
            logger.error(
                "Prediction failed for state='%s', district='%s', "
                "prediction_year=%s: %s",
                state, district, prediction_year, exc,
            )
            raise
        except (DatasetLoaderError, ModelLoaderError) as exc:
            logger.error(
                "Prediction failed for state='%s', district='%s', "
                "prediction_year=%s due to an underlying data/model "
                "error: %s", state, district, prediction_year, exc,
            )
            raise PredictionError(f"Prediction failed: {exc}") from exc
        except Exception as exc:
            logger.error(
                "Prediction failed for state='%s', district='%s', "
                "prediction_year=%s due to an unexpected error: %s",
                state, district, prediction_year, exc,
            )
            raise PredictionError(
                f"An unexpected error occurred during prediction: {exc}"
            ) from exc

        elapsed_seconds = time.perf_counter() - start_time

        response: Dict[str, Any] = {
            "state": validated_state,
            "district": validated_district,
            "prediction_year": validated_year,
            "predicted_crime_count": float(predicted_crime_count),
            "risk_level": risk_level,
            "confidence": confidence,
            "latest_dataset_year": latest_dataset_year,
            "model": model_name,
            "status": "success",
        }

        logger.info(
            "Prediction completed: state='%s', district='%s', "
            "prediction_year=%d, predicted_crime_count=%.4f, "
            "risk_level='%s', confidence=%.2f, model='%s', elapsed=%.3fs.",
            validated_state, validated_district, validated_year,
            predicted_crime_count, risk_level, confidence, model_name, elapsed_seconds,
        )

        return response
