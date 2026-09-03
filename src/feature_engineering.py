"""
feature_engineering.py

Feature Engineering module for the "Hybrid CNN-LSTM Framework for
Spatio-Temporal Crime Prediction" project.

This module consumes the cleaned dataset produced by preprocessing.py
(dataset/Crimes_in_india_2001-2013_cleaned.csv) and produces a fully
engineered dataset ready for spatio-temporal model training:

    dataset/Crimes_in_india_2001-2013_features.csv

Pipeline steps:
    1. Load the cleaned dataset.
    2. Encode categorical variables (STATE/UT, DISTRICT) via LabelEncoder,
       while retaining the original text columns.
    3. Generate spatial features (LATITUDE, LONGITUDE) from a predefined
       State/UT centroid coordinate mapping.
    4. Generate temporal features (YEAR_INDEX) from YEAR.
    5. Scale numerical predictor columns with MinMaxScaler, excluding the
       target variable and identifier/temporal-index columns.
    6. Persist the engineered dataset and a Markdown summary report.

Author: B.Tech Final Year Project
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime
import sklearn

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CrimeFeatureEngineer")


class CrimeFeatureEngineer:
    """
    Performs feature engineering on the cleaned Indian crime dataset to
    prepare it for the Hybrid CNN-LSTM spatio-temporal prediction model.

    The class encapsulates the full feature engineering pipeline:
    categorical encoding, spatial feature generation, temporal feature
    generation, numerical scaling, and reporting.

    Attributes:
        input_path (Path): Path to the cleaned input CSV dataset.
        output_path (Path): Path where the engineered dataset is saved.
        report_path (Path): Path where the Markdown report is saved.
        df (Optional[pd.DataFrame]): The working dataframe.
        label_encoders (Dict[str, LabelEncoder]): Fitted encoders per column.
        scaler (Optional[MinMaxScaler]): Fitted scaler for numeric columns.
    """

    # ----------------------------------------------------------------- #
    # Predefined approximate centroid coordinates (Latitude, Longitude)
    # for Indian States/UTs as recorded in NCRB crime records (2001-2013).
    # Keys are normalized to upper-case for robust matching.
    # ----------------------------------------------------------------- #
    STATE_COORDINATES: Dict[str, Tuple[float, float]] = {
        "ANDHRA PRADESH": (15.9129, 79.7400),
        "ARUNACHAL PRADESH": (28.2180, 94.7278),
        "ASSAM": (26.2006, 92.9376),
        "BIHAR": (25.0961, 85.3131),
        "CHHATTISGARH": (21.2787, 81.8661),
        "GOA": (15.2993, 74.1240),
        "GUJARAT": (22.2587, 71.1924),
        "HARYANA": (29.0588, 76.0856),
        "HIMACHAL PRADESH": (31.1048, 77.1734),
        "JAMMU & KASHMIR": (33.7782, 76.5762),
        "JHARKHAND": (23.6102, 85.2799),
        "KARNATAKA": (15.3173, 75.7139),
        "KERALA": (10.8505, 76.2711),
        "MADHYA PRADESH": (22.9734, 78.6569),
        "MAHARASHTRA": (19.7515, 75.7139),
        "MANIPUR": (24.6637, 93.9063),
        "MEGHALAYA": (25.4670, 91.3662),
        "MIZORAM": (23.1645, 92.9376),
        "NAGALAND": (26.1584, 94.5624),
        "ODISHA": (20.9517, 85.0985),
        "ORISSA": (20.9517, 85.0985),
        "PUNJAB": (31.1471, 75.3412),
        "RAJASTHAN": (27.0238, 74.2179),
        "SIKKIM": (27.5330, 88.5122),
        "TAMIL NADU": (11.1271, 78.6569),
        "TRIPURA": (23.9408, 91.9882),
        "UTTAR PRADESH": (26.8467, 80.9462),
        "UTTARAKHAND": (30.0668, 79.0193),
        "UTTARANCHAL": (30.0668, 79.0193),
        "WEST BENGAL": (22.9868, 87.8550),
        "TELANGANA": (18.1124, 79.0193),
        "A & N ISLANDS": (11.7401, 92.6586),
        "A&N ISLANDS": (11.7401, 92.6586),
        "ANDAMAN & NICOBAR ISLANDS": (11.7401, 92.6586),
        "CHANDIGARH": (30.7333, 76.7794),
        "D & N HAVELI": (20.1809, 73.0169),
        "D&N HAVELI": (20.1809, 73.0169),
        "DADRA & NAGAR HAVELI": (20.1809, 73.0169),
        "DAMAN & DIU": (20.4283, 72.8397),
        "DELHI": (28.7041, 77.1025),
        "DELHI UT": (28.7041, 77.1025),
        "LAKSHADWEEP": (10.5667, 72.6417),
        "PUDUCHERRY": (11.9416, 79.8083),
        "PONDICHERRY": (11.9416, 79.8083),
    }

    def __init__(
        self,
        input_path: str = "dataset/Crimes_in_india_2001-2013_cleaned.csv",
        output_path: str = "dataset/Crimes_in_india_2001-2013_features.csv",
        report_path: str = "results/feature_engineering_report.md",
    ) -> None:
        """
        Initialize the CrimeFeatureEngineer with I/O paths and internal state.

        Args:
            input_path: Relative path to the cleaned input dataset.
            output_path: Relative path where the engineered dataset is saved.
            report_path: Relative path where the Markdown report is saved.
        """
        self.input_path: Path = Path(input_path)
        self.output_path: Path = Path(output_path)
        self.report_path: Path = Path(report_path)

        self.df: Optional[pd.DataFrame] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[MinMaxScaler] = None

        # Bookkeeping used later for the report.
        self.encoded_columns: List[str] = []
        self.spatial_columns: List[str] = []
        self.temporal_columns: List[str] = []
        self.scaled_columns: List[str] = []
        self.initial_shape: Optional[Tuple[int, int]] = None

    # ----------------------------------------------------------------- #
    # Step 1: Load dataset
    # ----------------------------------------------------------------- #
    def load_dataset(self) -> None:
        """Load the cleaned dataset from disk into memory."""
        try:
            if not self.input_path.exists():
                raise FileNotFoundError(
                    f"Cleaned dataset not found at: {self.input_path}"
                )
            self.df = pd.read_csv(self.input_path)
            self.initial_shape = self.df.shape
            logger.info("Dataset loaded successfully...")
            logger.info(f"Initial dataset shape: {self.initial_shape}")
        except Exception as exc:
            logger.error(f"Failed to load dataset: {exc}")
            raise

    # ----------------------------------------------------------------- #
    # Step 2: Encode categorical features
    # ----------------------------------------------------------------- #
    def encode_categorical_features(self) -> None:
        """
        Encode STATE/UT and DISTRICT into numeric labels using LabelEncoder.

        New columns STATE_ENCODED and DISTRICT_ENCODED are appended; the
        original text columns are preserved for readability/EDA purposes.
        """
        logger.info("Encoding categorical variables...")
        self._ensure_loaded()

        try:
            column_mapping = [
                ("STATE/UT", "STATE_ENCODED"),
                ("DISTRICT", "DISTRICT_ENCODED"),
            ]

            for source_col, new_col in column_mapping:
                if source_col not in self.df.columns:
                    logger.warning(
                        f"Column '{source_col}' not found in dataset. "
                        f"Skipping encoding for it."
                    )
                    continue

                encoder = LabelEncoder()
                self.df[new_col] = encoder.fit_transform(
                    self.df[source_col].astype(str)
                )
                self.label_encoders[source_col] = encoder
                self.encoded_columns.append(new_col)

            logger.info(f"Categorical encoding complete. New columns: {self.encoded_columns}")
        except Exception as exc:
            logger.error(f"Error encoding categorical features: {exc}")
            raise

    # ----------------------------------------------------------------- #
    # Step 3: Generate spatial features
    # ----------------------------------------------------------------- #
    def generate_spatial_features(self) -> None:
        """
        Generate LATITUDE and LONGITUDE columns using a predefined
        State/UT centroid coordinate mapping.

        States/UTs absent from the mapping receive NaN coordinates and a
        warning is logged listing the unmapped names.
        """
        logger.info("Generating spatial features...")
        self._ensure_loaded()

        if "STATE/UT" not in self.df.columns:
            logger.warning(
                "Column 'STATE/UT' not found. Spatial features cannot be generated."
            )
            return

        try:
            latitudes: List[float] = []
            longitudes: List[float] = []
            unmapped_states = set()

            for state_name in self.df["STATE/UT"]:
                normalized_name = str(state_name).strip().upper()
                coordinates = self.STATE_COORDINATES.get(normalized_name)

                if coordinates is None:
                    unmapped_states.add(state_name)
                    latitudes.append(np.nan)
                    longitudes.append(np.nan)
                else:
                    latitudes.append(coordinates[0])
                    longitudes.append(coordinates[1])

            self.df["LATITUDE"] = latitudes
            self.df["LONGITUDE"] = longitudes
            self.spatial_columns = ["LATITUDE", "LONGITUDE"]

            if unmapped_states:
                logger.warning(
                    f"No coordinate mapping found for {len(unmapped_states)} "
                    f"State/UT value(s): {sorted(unmapped_states)}. "
                    f"LATITUDE/LONGITUDE set to NaN for affected rows."
                )

            logger.info("Spatial features generated: LATITUDE, LONGITUDE")
        except Exception as exc:
            logger.error(f"Error generating spatial features: {exc}")
            raise

    # ----------------------------------------------------------------- #
    # Step 4: Generate temporal features
    # ----------------------------------------------------------------- #
    def generate_temporal_features(self) -> None:
        """
        Generate a zero-based sequential YEAR_INDEX feature from YEAR.

        Example: 2001 -> 0, 2002 -> 1, ..., 2013 -> 12.
        The original YEAR column is retained.
        """
        logger.info("Generating temporal features...")
        self._ensure_loaded()

        try:
            if "YEAR" not in self.df.columns:
                raise KeyError("Required column 'YEAR' not found in dataset.")

            base_year = int(self.df["YEAR"].min())
            self.df["YEAR_INDEX"] = self.df["YEAR"].astype(int) - base_year
            self.temporal_columns = ["YEAR", "YEAR_INDEX"]

            logger.info(
                f"Temporal feature YEAR_INDEX generated using base year {base_year}."
            )
        except Exception as exc:
            logger.error(f"Error generating temporal features: {exc}")
            raise

    # ----------------------------------------------------------------- #
    # Step 5: Scale numerical features
    # ----------------------------------------------------------------- #
    def scale_numerical_features(self) -> None:
        """
        Scale numerical predictor columns to [0, 1] using MinMaxScaler.

        The target variable (TOTAL IPC CRIMES) and identifier/temporal
        columns (Id, YEAR, YEAR_INDEX) are explicitly excluded from scaling.
        """
        logger.info("Scaling numerical columns...")
        self._ensure_loaded()

        try:
            excluded_columns = {"TOTAL IPC CRIMES", "YEAR", "YEAR_INDEX", "Id"}
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            columns_to_scale = [col for col in numeric_columns if col not in excluded_columns]

            if not columns_to_scale:
                logger.warning("No eligible numerical columns found for scaling.")
                return

            self.scaler = MinMaxScaler()
            self.df[columns_to_scale] = self.scaler.fit_transform(self.df[columns_to_scale])
            self.scaled_columns = columns_to_scale

            logger.info(
                f"Scaled {len(columns_to_scale)} numerical column(s) using MinMaxScaler."
            )
        except Exception as exc:
            logger.error(f"Error scaling numerical features: {exc}")
            raise

    # ----------------------------------------------------------------- #
    # Step 6: Save engineered dataset
    # ----------------------------------------------------------------- #
    def save_feature_dataset(self) -> None:
        """Persist the engineered dataset to the configured output path."""
        logger.info("Saving engineered dataset...")
        self._ensure_loaded()

        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.df.to_csv(self.output_path, index=False)
            logger.info(f"Engineered dataset saved to: {self.output_path}")
        except Exception as exc:
            logger.error(f"Error saving feature dataset: {exc}")
            raise

    # ----------------------------------------------------------------- #
    # Step 6.5: Save deployment artifacts
    # ----------------------------------------------------------------- #
    def save_artifacts(self, artifacts_dir: str = "artifacts") -> None:
        """
        Persist fitted encoders, scaler, feature columns, and run metadata
        to the artifacts/ directory so the Django backend can load them at
        inference time without refitting on new data.
        """
        logger.info("Saving deployment artifacts...")
        self._ensure_loaded()

        artifacts_path = Path(artifacts_dir)

        try:
            artifacts_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Artifacts directory ready at: {artifacts_path}")
        except Exception as exc:
            logger.error(f"Error creating artifacts directory: {exc}")
            raise

        # ---- Save each fitted LabelEncoder separately ---- #
        encoder_filenames = {
            "STATE/UT": "state_ut_encoder.pkl",
            "DISTRICT": "district_encoder.pkl",
        }
        for source_col, encoder in self.label_encoders.items():
            try:
                filename = encoder_filenames.get(
                    source_col,
                    f"{source_col.lower().replace('/', '_')}_encoder.pkl",
                )
                encoder_path = artifacts_path / filename
                joblib.dump(encoder, encoder_path)
                logger.info(f"Saved LabelEncoder for '{source_col}' to: {encoder_path}")
            except Exception as exc:
                logger.error(f"Error saving LabelEncoder for '{source_col}': {exc}")

        # ---- Save fitted MinMaxScaler ---- #
        try:
            if self.scaler is not None:
                scaler_path = artifacts_path / "scaler.pkl"
                joblib.dump(self.scaler, scaler_path)
                logger.info(f"Saved MinMaxScaler to: {scaler_path}")
            else:
                logger.warning("No fitted scaler found. Skipping scaler.pkl save.")
        except Exception as exc:
            logger.error(f"Error saving MinMaxScaler: {exc}")

        # ---- Save ordered feature column names ---- #
        try:
            feature_columns_path = artifacts_path / "feature_columns.json"
            feature_columns = self.df.columns.tolist()
            with open(feature_columns_path, "w", encoding="utf-8") as f:
                json.dump(feature_columns, f, indent=4)
            logger.info(f"Saved feature column names to: {feature_columns_path}")
        except Exception as exc:
            logger.error(f"Error saving feature columns: {exc}")

        # ---- Save run metadata ---- #
        try:
            try:
                sklearn_version = sklearn.__version__
            except AttributeError:
                sklearn_version = "unknown"

            # Input feature count excludes the target variable and
            # non-feature columns (identifiers, raw text categoricals that
            # have already been label-encoded into STATE_ENCODED /
            # DISTRICT_ENCODED). This reflects the actual model input width,
            # not the total column count of the saved CSV.
            non_feature_columns = {"TOTAL IPC CRIMES", "Id", "STATE/UT", "DISTRICT"}
            input_feature_columns = [
                col for col in self.df.columns if col not in non_feature_columns
            ]

            metadata = {
                "project_title": "Hybrid CNN-LSTM Framework for Spatio-Temporal Crime Prediction",
                "dataset_name": self.input_path.name,
                "target_column": "TOTAL IPC CRIMES",
                "number_of_features": len(input_feature_columns),
                "generated_at": datetime.now().isoformat(),
                "sklearn_version": sklearn_version,
            }

            metadata_path = artifacts_path / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Saved metadata to: {metadata_path}")
        except Exception as exc:
            logger.error(f"Error saving metadata: {exc}")

        logger.info("Deployment artifacts saved successfully.")

    # ----------------------------------------------------------------- #
    # Step 7: Generate report
    # ----------------------------------------------------------------- #
    def generate_report(self) -> None:
        """Generate a Markdown summary report of the feature engineering run."""
        logger.info("Generating feature engineering report...")
        self._ensure_loaded()

        try:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)

            report_lines = [
                "# Feature Engineering Report",
                "",
                "## Dataset Shape",
                f"- Initial shape: {self.initial_shape}",
                f"- Final shape: {self.df.shape}",
                "",
                "## Encoded Columns",
                f"- {self.encoded_columns if self.encoded_columns else 'None'}",
                "- Method: sklearn LabelEncoder",
                "- Original categorical columns (STATE/UT, DISTRICT) retained.",
                "",
                "## Spatial Features Generated",
                f"- {self.spatial_columns if self.spatial_columns else 'None'}",
                "- Source: predefined State/UT centroid coordinate mapping.",
                "- Unmapped State/UT values were assigned NaN (see console log).",
                "",
                "## Temporal Features Generated",
                f"- {self.temporal_columns if self.temporal_columns else 'None'}",
                "- YEAR_INDEX is a zero-based sequential index derived from YEAR "
                "(e.g., 2001 -> 0, 2002 -> 1, ..., 2013 -> 12).",
                "",
                "## Scaling Method",
                "- MinMaxScaler (feature range 0-1)",
                f"- Number of columns scaled: {len(self.scaled_columns)}",
                "- Excluded from scaling: TOTAL IPC CRIMES, YEAR, YEAR_INDEX, Id",
                "",
                "## Final Feature Count",
                f"- Total columns in final dataset: {self.df.shape[1]}",
                f"- Total rows in final dataset: {self.df.shape[0]}",
                "",
                "## Output Dataset Path",
                f"- `{self.output_path}`",
                "",
                "## Readiness for Model Training",
                "- All categorical variables are numerically encoded.",
                "- Spatial (LATITUDE, LONGITUDE) and temporal (YEAR_INDEX) features "
                "are available for spatio-temporal sequence construction.",
                "- Numerical predictors are scaled to [0, 1]; the target variable "
                "(TOTAL IPC CRIMES) remains in its original scale.",
                "- Dataset is ready for consumption by cnn_model.py, lstm_model.py, "
                "and hybrid_model.py.",
                "",
            ]

            self.report_path.write_text("\n".join(report_lines), encoding="utf-8")
            logger.info(f"Feature engineering report saved to: {self.report_path}")
        except Exception as exc:
            logger.error(f"Error generating report: {exc}")
            raise

    # ----------------------------------------------------------------- #
    # Orchestration
    # ----------------------------------------------------------------- #
    def run_feature_engineering(self) -> None:
        """Execute the full feature engineering pipeline end to end."""
        try:
            self.load_dataset()
            self.encode_categorical_features()
            self.generate_spatial_features()
            self.generate_temporal_features()
            self.scale_numerical_features()
            self.save_feature_dataset()
            self.save_artifacts()
            self.generate_report()
            logger.info("Feature engineering completed successfully.")
        except Exception as exc:
            logger.error(f"Feature engineering pipeline failed: {exc}")
            raise

    # ----------------------------------------------------------------- #
    # Internal helper
    # ----------------------------------------------------------------- #
    def _ensure_loaded(self) -> None:
        """Raise a clear error if the pipeline runs out of order."""
        if self.df is None:
            raise ValueError(
                "Dataset has not been loaded yet. Call load_dataset() first."
            )


if __name__ == "__main__":
    engineer = CrimeFeatureEngineer()
    engineer.run_feature_engineering()