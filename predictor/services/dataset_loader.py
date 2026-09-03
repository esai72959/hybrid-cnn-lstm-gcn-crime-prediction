"""
dataset_loader.py

Production-grade singleton loader for the engineered crime dataset used by
the Hybrid CNN-LSTM Spatio-Temporal Crime Prediction system.

This module is responsible exclusively for loading and serving the
engineered dataset (state / district / year-level historical crime
features). It does NOT perform any prediction, encoding, scaling, or
model inference - those responsibilities belong to ModelLoader and
CrimePredictor respectively.

Author: Final Year B.Tech Project - Hybrid CNN-LSTM Crime Prediction
"""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import pandas as pd

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


class DatasetLoaderError(Exception):
    """Raised when the engineered dataset cannot be located, loaded, or
    queried as requested."""


class DatasetLoader:
    """
    Singleton responsible for loading and serving the engineered crime
    dataset used to auto-populate model features for a given
    State / District / Prediction Year selection.

    The class locates the project root automatically (relative to this
    file's location) and resolves the dataset path from there, so it
    remains portable across operating systems and deployment machines.

    Usage
    -----
        loader = DatasetLoader()
        loader.load_dataset()
        states = loader.get_states()
        districts = loader.get_districts("TELANGANA")
        latest = loader.get_latest_record("TELANGANA", "HYDERABAD")
        record = loader.prepare_prediction_record(
            "TELANGANA", "HYDERABAD", 2027
        )

    Notes
    -----
    - This class implements the singleton design pattern: only one
      instance is ever created per process, and the dataset is loaded
      at most once (lazy initialization).
    - Thread-safe: a lock guards instance creation and the loading
      routine to avoid race conditions under concurrent request
      handling (e.g. Django + multiple worker threads).
    - This class strictly serves data. It never loads TensorFlow, never
      loads encoders/scalers, and never performs prediction.
    """

    _instance: Optional["DatasetLoader"] = None
    _instance_lock: Lock = Lock()

    # Directory / file names relative to the project root.
    _DATASET_DIR = "dataset"
    _DATASET_FILE = "Crimes_in_india_2001-2013_features.csv"

    # Column names expected in the engineered dataset.
    _COL_STATE = "STATE/UT"
    _COL_DISTRICT = "DISTRICT"
    _COL_YEAR = "YEAR"
    _COL_YEAR_INDEX = "YEAR_INDEX"

    def __new__(cls) -> "DatasetLoader":
        """Ensure only a single instance of DatasetLoader ever exists."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """
        Initialize instance attributes exactly once.

        Because __new__ always returns the same instance, __init__ would
        otherwise reset an already-loaded dataset on every
        DatasetLoader() call. The `_initialized` guard prevents that.
        """
        if self._initialized:
            return

        self._load_lock: Lock = Lock()

        # Cached dataset (None until loaded).
        self.dataframe: Optional[pd.DataFrame] = None

        # O(1) in-memory lookup caches
        self._cached_states: Optional[List[str]] = None
        self._cached_districts_by_state: Dict[str, List[str]] = {}
        self._cached_history_by_key: Dict[Tuple[str, str], pd.DataFrame] = {}
        self._cached_latest_by_key: Dict[Tuple[str, str], pd.Series] = {}

        # Resolve project root: predictor/services/dataset_loader.py -> root.
        self.project_root: Path = Path(__file__).resolve().parents[2]
        self.dataset_dir: Path = self.project_root / self._DATASET_DIR
        self.dataset_path: Path = self.dataset_dir / self._DATASET_FILE

        logger.info(
            "DatasetLoader initialized. Project root resolved to: %s",
            self.project_root,
        )

        self._initialized = True

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------
    @staticmethod
    def _require_file(path: Path) -> None:
        """
        Verify that a required dataset file exists on disk.

        Parameters
        ----------
        path : Path
            Path to the required file.

        Raises
        ------
        DatasetLoaderError
            If the file does not exist at the given path.
        """
        if not path.exists() or not path.is_file():
            raise DatasetLoaderError(
                f"Required dataset file not found: '{path}'. "
                "Please verify the project directory structure."
            )

    def _build_indexes(self) -> None:
        """Build O(1) lookup dictionaries for states, districts, and historical records."""
        if self.dataframe is None or self.dataframe.empty:
            return

        df = self.dataframe
        # Cache unique states
        self._cached_states = sorted(
            df[self._COL_STATE].dropna().astype(str).str.strip().unique().tolist()
        )

        self._cached_districts_by_state = {}
        self._cached_history_by_key = {}
        self._cached_latest_by_key = {}

        df_work = df.copy(deep=False)
        df_work["_NORM_STATE"] = df_work[self._COL_STATE].astype(str).str.strip().str.upper()
        df_work["_NORM_DISTRICT"] = df_work[self._COL_DISTRICT].astype(str).str.strip().str.upper()

        for state_norm, state_group in df_work.groupby("_NORM_STATE"):
            dists = sorted(
                state_group[self._COL_DISTRICT].dropna().astype(str).str.strip().unique().tolist()
            )
            self._cached_districts_by_state[state_norm] = dists

            for dist_norm, group in state_group.groupby("_NORM_DISTRICT"):
                clean_group = group.drop(columns=["_NORM_STATE", "_NORM_DISTRICT"]).sort_values(
                    by=self._COL_YEAR, ascending=True
                ).reset_index(drop=True)
                key = (state_norm, dist_norm)
                self._cached_history_by_key[key] = clean_group
                self._cached_latest_by_key[key] = clean_group.iloc[-1].copy(deep=True)

    def _ensure_loaded(self) -> pd.DataFrame:
        """
        Return the cached dataset, loading it first if necessary.

        Returns
        -------
        pd.DataFrame
            The loaded engineered dataset.
        """
        if self.dataframe is None:
            self.load_dataset()
        return self.dataframe

    def _require_columns(self, columns: List[str]) -> None:
        """
        Verify that the given columns exist in the loaded dataset.

        Parameters
        ----------
        columns : List[str]
            Column names that must be present.

        Raises
        ------
        DatasetLoaderError
            If any required column is missing.
        """
        dataframe = self._ensure_loaded()
        missing = [col for col in columns if col not in dataframe.columns]
        if missing:
            raise DatasetLoaderError(
                f"Required column(s) missing from dataset: {missing}. "
                f"Available columns: {list(dataframe.columns)}."
            )

    # ----------------------------------------------------------------
    # Dataset loading
    # ----------------------------------------------------------------
    def load_dataset(self) -> pd.DataFrame:
        """
        Load the engineered crime dataset from disk (lazy singleton).

        Returns
        -------
        pd.DataFrame
            The loaded engineered dataset.

        Raises
        ------
        DatasetLoaderError
            If the dataset file is missing or fails to load.
        """
        with self._load_lock:
            if self.dataframe is None:
                try:
                    self._require_file(self.dataset_path)
                    self.dataframe = pd.read_csv(self.dataset_path)
                    self._build_indexes()
                    logger.info(
                        "Dataset loaded successfully from: %s "
                        "(%d rows, %d columns, %d indexed districts)",
                        self.dataset_path,
                        self.dataframe.shape[0],
                        self.dataframe.shape[1],
                        len(self._cached_latest_by_key),
                    )
                except DatasetLoaderError:
                    logger.error(
                        "Dataset file missing at: %s", self.dataset_path
                    )
                    raise
                except Exception as exc:
                    logger.error(
                        "Failed to load dataset from '%s': %s",
                        self.dataset_path, exc,
                    )
                    raise DatasetLoaderError(
                        f"Failed to load dataset: {exc}"
                    ) from exc
        return self.dataframe

    def get_dataframe(self) -> pd.DataFrame:
        """
        Return the loaded engineered dataset, loading it if necessary.

        Returns
        -------
        pd.DataFrame
            The full engineered dataset.
        """
        return self._ensure_loaded()

    # ----------------------------------------------------------------
    # Query methods
    # ----------------------------------------------------------------
    def get_states(self) -> List[str]:
        """
        Return the sorted list of unique states/UTs present in the
        dataset (O(1) indexed lookup).
        """
        try:
            self._ensure_loaded()
            if self._cached_states is not None:
                return list(self._cached_states)
            self._require_columns([self._COL_STATE])
            dataframe = self._ensure_loaded()
            states = (
                dataframe[self._COL_STATE]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )
            return sorted(states)
        except DatasetLoaderError:
            raise
        except Exception as exc:
            logger.error("Failed to retrieve states: %s", exc)
            raise DatasetLoaderError(
                f"Failed to retrieve states: {exc}"
            ) from exc

    def get_districts(self, state: str) -> List[str]:
        """
        Return the sorted list of districts belonging to a given state (O(1) lookup).
        """
        if state is None or str(state).strip() == "":
            raise DatasetLoaderError("Missing required argument: 'state'.")

        try:
            self._ensure_loaded()
            normalized_state = str(state).strip().upper()
            if normalized_state in self._cached_districts_by_state:
                return list(self._cached_districts_by_state[normalized_state])

            self._require_columns([self._COL_STATE, self._COL_DISTRICT])
            dataframe = self._ensure_loaded()

            state_mask = (
                dataframe[self._COL_STATE]
                .astype(str)
                .str.strip()
                .str.upper()
                == normalized_state
            )

            districts = (
                dataframe.loc[state_mask, self._COL_DISTRICT]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )
            return sorted(districts)
        except DatasetLoaderError:
            raise
        except Exception as exc:
            logger.error(
                "Failed to retrieve districts for state '%s': %s",
                state, exc,
            )
            raise DatasetLoaderError(
                f"Failed to retrieve districts for state '{state}': {exc}"
            ) from exc

    def get_historical_records_fast(self, state: str, district: str) -> pd.DataFrame:
        """
        Return all historical records for (state, district) in O(1) time.
        """
        self._ensure_loaded()
        key = (str(state).strip().upper(), str(district).strip().upper())
        if key in self._cached_history_by_key:
            return self._cached_history_by_key[key].copy(deep=True)
        return pd.DataFrame()

    def get_latest_record(self, state: str, district: str) -> pd.Series:
        """
        Return the latest available record (highest YEAR) for the given
        state and district (O(1) lookup).
        """
        if state is None or str(state).strip() == "":
            raise DatasetLoaderError("Missing required argument: 'state'.")
        if district is None or str(district).strip() == "":
            raise DatasetLoaderError(
                "Missing required argument: 'district'."
            )

        try:
            self._ensure_loaded()
            key = (str(state).strip().upper(), str(district).strip().upper())
            if key in self._cached_latest_by_key:
                return self._cached_latest_by_key[key].copy(deep=True)

            raise DatasetLoaderError(
                f"No historical record found for state='{state}', "
                f"district='{district}'."
            )
        except DatasetLoaderError:
            raise
        except Exception as exc:
            logger.error(
                "Failed to retrieve latest record for state='%s', "
                "district='%s': %s",
                state, district, exc,
            )
            raise DatasetLoaderError(
                f"Failed to retrieve latest record for state='{state}', "
                f"district='{district}': {exc}"
            ) from exc

    # ----------------------------------------------------------------
    # Prediction record preparation
    # ----------------------------------------------------------------
    def prepare_prediction_record(
        self, state: str, district: str, prediction_year: int
    ) -> pd.Series:
        """
        Build a feature record for a future prediction year by copying
        the latest available historical record for the given state and
        district, then updating its YEAR (and YEAR_INDEX, if present).

        The source DataFrame stored in this loader is never modified -
        only a detached copy of the latest record is updated and
        returned.

        Parameters
        ----------
        state : str
            State/UT name (case-insensitive, whitespace tolerant).
        district : str
            District name (case-insensitive, whitespace tolerant).
        prediction_year : int
            The future year to prepare a feature record for.

        Returns
        -------
        pd.Series
            A modified copy of the latest historical record, with YEAR
            set to `prediction_year` and YEAR_INDEX shifted by the same
            number of years (if the YEAR_INDEX column exists in the
            dataset).

        Raises
        ------
        DatasetLoaderError
            If inputs are invalid or no historical record exists to
            base the prediction record on.
        """
        try:
            prediction_year = int(prediction_year)
        except (TypeError, ValueError) as exc:
            raise DatasetLoaderError(
                f"'prediction_year' must be an integer, got "
                f"'{prediction_year}'."
            ) from exc

        try:
            latest_record = self.get_latest_record(state, district)
            prepared_record = latest_record.copy(deep=True)

            latest_year = latest_record[self._COL_YEAR]
            prepared_record[self._COL_YEAR] = prediction_year

            if self._COL_YEAR_INDEX in prepared_record.index:
                year_delta = prediction_year - latest_year
                prepared_record[self._COL_YEAR_INDEX] = (
                    latest_record[self._COL_YEAR_INDEX] + year_delta
                )
            else:
                logger.warning(
                    "'%s' column not found in dataset; YEAR_INDEX was "
                    "not updated for state='%s', district='%s'.",
                    self._COL_YEAR_INDEX, state, district,
                )

            logger.info(
                "Prediction record prepared for state='%s', "
                "district='%s', prediction_year=%s (based on YEAR=%s).",
                state, district, prediction_year, latest_year,
            )
            return prepared_record
        except DatasetLoaderError:
            raise
        except Exception as exc:
            logger.error(
                "Failed to prepare prediction record for state='%s', "
                "district='%s', prediction_year=%s: %s",
                state, district, prediction_year, exc,
            )
            raise DatasetLoaderError(
                "Failed to prepare prediction record for "
                f"state='{state}', district='{district}', "
                f"prediction_year={prediction_year}: {exc}"
            ) from exc

    # ----------------------------------------------------------------
    # Summary helpers
    # ----------------------------------------------------------------
    def dataset_summary(self) -> Dict[str, Any]:
        """
        Return a high-level summary of the loaded dataset.

        Returns
        -------
        Dict[str, Any]
            {
                "num_rows": int,
                "num_columns": int,
                "states": List[str],
                "districts": List[str],
                "year_range": {"min": int, "max": int}
            }

        Raises
        ------
        DatasetLoaderError
            If the dataset or required columns are unavailable.
        """
        try:
            self._require_columns(
                [self._COL_STATE, self._COL_DISTRICT, self._COL_YEAR]
            )
            dataframe = self._ensure_loaded()

            districts = (
                dataframe[self._COL_DISTRICT]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )

            summary: Dict[str, Any] = {
                "num_rows": int(dataframe.shape[0]),
                "num_columns": int(dataframe.shape[1]),
                "states": self.get_states(),
                "districts": sorted(districts),
                "year_range": {
                    "min": int(dataframe[self._COL_YEAR].min()),
                    "max": int(dataframe[self._COL_YEAR].max()),
                },
            }

            logger.info(
                "Dataset summary generated: %d rows, %d columns, "
                "%d states, %d districts, year_range=%s-%s.",
                summary["num_rows"], summary["num_columns"],
                len(summary["states"]), len(summary["districts"]),
                summary["year_range"]["min"], summary["year_range"]["max"],
            )
            return summary
        except DatasetLoaderError:
            raise
        except Exception as exc:
            logger.error("Failed to generate dataset summary: %s", exc)
            raise DatasetLoaderError(
                f"Failed to generate dataset summary: {exc}"
            ) from exc

    def get_statistics(self) -> Dict[str, int]:
        """
        Return aggregate dataset statistics in a single call.

        Centralizes stats logic so callers (e.g. api_dashboard) don't
        need to know internal dataset structure. Unlike dataset_summary,
        this never raises - it degrades to zeroed-out values on any
        failure, since it's intended for lightweight dashboard display.

        Returns
        -------
        Dict[str, int]
            {"records": int, "states": int, "districts": int, "years": int}
        """
        try:
            states_list = self.get_states() or []
            districts_set = set()
            for state in states_list:
                try:
                    districts_set.update(self.get_districts(state) or [])
                except DatasetLoaderError:
                    continue

            dataframe = self.dataframe
            records = int(len(dataframe)) if dataframe is not None else 0
            years_count = 0
            if dataframe is not None and self._COL_YEAR in dataframe.columns:
                years_count = int(dataframe[self._COL_YEAR].nunique())

            return {
                "records": records,
                "states": len(states_list),
                "districts": len(districts_set),
                "years": years_count,
            }
        except Exception as exc:
            logger.error("Failed to compute dataset statistics: %s", exc)
            return {
                "records": 0,
                "states": 0,
                "districts": 0,
                "years": 0,
            }

    def get_yearly_crime_trend(self) -> List[Dict[str, int]]:
        """
        Return aggregated year-wise total crime counts for trend visualization.

        Returns
        -------
        List[Dict[str, int]]
            A list of dictionaries sorted ascending by year, e.g.:
            [
                {"year": 2001, "crime_count": 1789423},
                {"year": 2002, "crime_count": 1823901},
                ...
            ]

        Raises
        ------
        DatasetLoaderError
            If dataset loading fails or the 'YEAR' column is missing.
        """
        try:
            self._require_columns([self._COL_YEAR])
            dataframe = self._ensure_loaded()

            # Identify numerical feature columns, explicitly excluding YEAR and YEAR_INDEX
            excluded_cols = {self._COL_YEAR, self._COL_YEAR_INDEX}
            numeric_cols = [
                col for col in dataframe.select_dtypes(include=["number"]).columns
                if col not in excluded_cols
            ]

            if not numeric_cols:
                raise DatasetLoaderError(
                    "No numerical crime feature columns found to compute yearly trend."
                )

            # Group by year, sum all numeric crime features across columns, then aggregate total
            trend_df = dataframe.groupby(self._COL_YEAR)[numeric_cols].sum()
            yearly_totals = trend_df.sum(axis=1).sort_index(ascending=True)

            trend_data = [
                {"year": int(year), "crime_count": int(count)}
                for year, count in yearly_totals.items()
            ]

            logger.info("Successfully generated yearly crime trend for %d years.", len(trend_data))
            return trend_data

        except DatasetLoaderError:
            raise
        except Exception as exc:
            logger.error("Failed to generate yearly crime trend: %s", exc)
            raise DatasetLoaderError(
                f"Failed to generate yearly crime trend: {exc}"
            ) from exc


# Singleton instance
dataset_loader = DatasetLoader()