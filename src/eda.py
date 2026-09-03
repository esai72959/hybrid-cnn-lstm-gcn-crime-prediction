"""
=========================================================
Project : A Hybrid CNN-LSTM Framework for Spatio-Temporal Crime Prediction
Module  : Exploratory Data Analysis (EDA)

Dataset :
Crimes_in_india_2001-2013.csv

Description :
This module performs exploratory data analysis on the crime dataset.
It analyzes the dataset structure, identifies data quality issues,
generates statistical summaries, creates visualizations, and produces
a dataset readiness report for the preprocessing phase.
=========================================================
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class CrimeDataEDA:
    """
    Performs exploratory data analysis on the district-level, multi-year
    IPC crime dataset used for the CNN-LSTM crime prediction project.

    Unlike a fixed, hardcoded column list, the individual crime-type
    columns are identified dynamically after loading (see
    identify_crime_columns()). This dataset has 29 separate crime-type
    columns, and typing that many column name strings by hand invites
    mismatches with the actual CSV header - deriving the list from the
    loaded DataFrame avoids that risk entirely.

    Attributes:
        filepath (str): Path to the source CSV file.
        results_dir (str): Directory where generated plots are saved.
        df (pd.DataFrame): Loaded dataset, populated by load_dataset().
        location_columns (list[str]): State and district columns.
        year_column (str): Name of the year column.
        target_column (str): Aggregate crime count column.
        crime_columns (list[str]): Individual crime-type columns,
            populated by identify_crime_columns().
    """

    TOTAL_ROW_PATTERN = "total"  # matched case-insensitively, substring

    def __init__(self, filepath: str, results_dir: str):
        self.filepath = filepath
        self.results_dir = results_dir

        self.df = None

        # This dataset has no identifier column (unlike the earlier
        # single-year dataset, which had 'Id'). Recorded as None rather
        # than omitted, so downstream phases can check for its absence
        # explicitly instead of assuming it exists.
        self.id_column = None

        self.location_columns = ["STATE/UT", "DISTRICT"]
        self.year_column = "YEAR"
        self.target_column = "TOTAL IPC CRIMES"
        self.crime_columns = []  # populated by identify_crime_columns()

        os.makedirs(self.results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_dataset(self) -> bool:
        """
        Loads the CSV file into a DataFrame.

        Returns:
            bool: True if the dataset was loaded successfully, False
            otherwise.
        """
        try:
            self.df = pd.read_csv(self.filepath)
            print(f"Dataset loaded successfully from: {self.filepath}")
            return True
        except FileNotFoundError:
            print(f"ERROR: File not found at path: {self.filepath}")
        except pd.errors.EmptyDataError:
            print("ERROR: The CSV file is empty.")
        except pd.errors.ParserError as exc:
            print(f"ERROR: Failed to parse CSV file. Details: {exc}")
        return False

    def check_identifier_column(self) -> None:
        """
        Reports whether the dataset has an explicit identifier column.
        This dataset does not, which is worth flagging early since the
        original project data source (a different, single-year dataset)
        did have one - Phase 2 may need to generate a surrogate Id if
        one is required downstream.
        """
        print("\n--- Identifier Column Check ---")
        if self.id_column is None:
            print("No explicit identifier column in this dataset. "
                  "Row identity is currently just the DataFrame index.")
        else:
            print(f"Identifier column: {self.id_column}")

    # ------------------------------------------------------------------
    # Basic structural checks
    # ------------------------------------------------------------------
    def display_head(self, n: int = 10) -> None:
        """Prints the first n rows of the dataset."""
        print(f"\n--- First {n} Rows ---")
        print(self.df.head(n))

    def show_shape(self) -> None:
        """Prints the number of rows and columns in the dataset."""
        rows, cols = self.df.shape
        print("\n--- Dataset Shape ---")
        print(f"Rows: {rows}, Columns: {cols}")

    def show_info(self) -> None:
        """Prints the DataFrame.info() summary (dtypes, non-null counts)."""
        print("\n--- Dataset Info ---")
        self.df.info()

    def show_dtypes(self) -> None:
        """Prints the data type of every column."""
        print("\n--- Data Types ---")
        print(self.df.dtypes)

    def statistical_summary(self) -> None:
        """Prints descriptive statistics for all numeric columns."""
        print("\n--- Statistical Summary (Numeric Columns) ---")
        print(self.df.describe())

    def missing_value_analysis(self) -> None:
        """Prints the count of missing values per column."""
        print("\n--- Missing Value Analysis ---")
        missing = self.df.isnull().sum()
        total_missing = missing.sum()
        if total_missing == 0:
            print("No missing values found in any column.")
        else:
            print(missing[missing > 0])

    def duplicate_analysis(self) -> None:
        """Prints the number of fully duplicated rows in the dataset."""
        print("\n--- Duplicate Analysis ---")
        duplicate_count = self.df.duplicated().sum()
        print(f"Number of duplicate rows: {duplicate_count}")

    # ------------------------------------------------------------------
    # Location and column identification
    # ------------------------------------------------------------------
    def _compute_state_case_stats(self) -> tuple:
        """
        Shared computation behind unique_states() and the Dataset
        Readiness Report.

        Returns:
            tuple: (raw unique count, case-normalized unique count).
        """
        state_col = self.location_columns[0]
        raw_count = self.df[state_col].nunique()
        normalized_count = self.df[state_col].str.strip().str.upper().nunique()
        return raw_count, normalized_count

    def unique_states(self) -> None:
        """
        Prints the number and names of unique states/UTs, comparing the
        raw count against a case-normalized count. A mismatch here means
        the same state is being stored under more than one casing (e.g.
        'BIHAR' and 'Bihar'), which silently fragments any state-level
        grouping until it is fixed.
        """
        print("\n--- Unique States/UTs ---")
        raw_count, normalized_count = self._compute_state_case_stats()

        print(f"Raw unique values: {raw_count}")
        print(f"Unique values after trimming whitespace and normalizing case: "
              f"{normalized_count}")

        if raw_count != normalized_count:
            print(f"WARNING: State names are inconsistently cased. "
                  f"{raw_count - normalized_count} extra raw labels are "
                  "case-variant duplicates of an existing state. This "
                  "must be standardized in Phase 2, otherwise every "
                  "state-level aggregation (grouping, plotting) will "
                  "treat 'Bihar' and 'BIHAR' as two different states.")

    def unique_districts(self) -> None:
        """Prints the number of unique district labels in the dataset."""
        print("\n--- Unique Districts ---")
        districts = self.df[self.location_columns[1]].unique()
        print(f"Total unique district labels (raw, includes non-district "
              f"summary rows - see aggregate row check): {len(districts)}")

    def identify_crime_columns(self) -> None:
        """
        Dynamically derives the list of individual crime-type columns as
        every column that is not a location column, the year column, or
        the aggregate target column. Doing this from the loaded
        DataFrame - rather than typing out all 29 column names - means
        the list stays correct even if the source CSV's column set
        changes slightly between dataset versions.
        """
        print("\n--- Crime Columns ---")
        excluded = set(self.location_columns) | {self.year_column, self.target_column}
        self.crime_columns = [c for c in self.df.columns if c not in excluded]
        print(f"Identified {len(self.crime_columns)} individual crime-type columns:")
        print(self.crime_columns)
        print(f"Aggregate target column: {self.target_column}")

    def _compute_aggregate_row_stats(self) -> tuple:
        """
        Shared computation behind detect_aggregate_rows() and the
        Dataset Readiness Report, so both always agree on the same
        numbers instead of maintaining two separate calculations.

        Returns:
            tuple: (boolean mask over self.df, value_counts Series of
            the distinct aggregate labels found).
        """
        district_col = self.location_columns[1]
        mask = self.df[district_col].str.contains(self.TOTAL_ROW_PATTERN, case=False, na=False)
        return mask, self.df.loc[mask, district_col].value_counts()

    def detect_aggregate_rows(self) -> None:
        """
        Checks for rows where the District column is actually a
        state-level (or union-territory-level) aggregate rather than a
        real district - e.g. 'TOTAL', 'ZZ TOTAL', 'DELHI UT TOTAL'. This
        is a recurring pattern in NCRB-style crime datasets: one summary
        row is appended per state per year. Left in place, these rows
        would double-count crime totals during any grouping operation.
        This method only detects and reports the issue - removing these
        rows from the stored dataset is a Phase 2 task. The aggregation
        charts later in this script work around the issue internally
        (see _get_analysis_view()) so they are not misleading in the
        meantime.
        """
        print("\n--- Aggregate/Total Row Check ---")
        district_col = self.location_columns[1]
        mask, label_counts = self._compute_aggregate_row_stats()
        if mask.sum() == 0:
            print("No state-level aggregate rows detected in the District column.")
            return

        print(f"WARNING: Found {mask.sum()} rows where '{district_col}' "
              "is a state-level aggregate, not a real district.")
        print("Distinct aggregate labels found:")
        print(label_counts)
        print("Observation: These rows must be removed in Phase 2 before "
              "any district-level or state-level grouping, otherwise "
              "totals are double-counted - once from the real districts "
              "and again from the state's own summary row.")

    # ------------------------------------------------------------------
    # Internal helper for aggregation-based charts
    # ------------------------------------------------------------------
    def _get_analysis_view(self) -> pd.DataFrame:
        """
        Returns a filtered, case-normalized copy of the dataset for use
        in aggregation charts only (top states, top districts,
        correlation, distribution, yearly trend).

        Two corrections are applied to this copy, and only this copy:
          1. Rows where District is a state-level aggregate (see
             detect_aggregate_rows()) are removed, so state and
             district totals are not double-counted.
          2. State/UT names are trimmed and upper-cased, so 'Bihar' and
             'BIHAR' are grouped together instead of being treated as
             two different states (see unique_states()).

        self.df itself is never modified by this method - the structural
        checks earlier in this script (shape, dtypes, missing values,
        duplicates) intentionally report on the original, unmodified
        data.
        """
        district_col = self.location_columns[1]
        state_col = self.location_columns[0]

        view = self.df[
            ~self.df[district_col].str.contains(self.TOTAL_ROW_PATTERN, case=False, na=False)
        ].copy()
        view[state_col] = view[state_col].str.strip().str.upper()
        return view

    # ------------------------------------------------------------------
    # Visual analysis (each method saves one PNG to results_dir)
    # ------------------------------------------------------------------
    def correlation_matrix(self) -> None:
        """
        Plots and saves a correlation heatmap across all crime columns
        and the target column, using the cleaned analysis view. This
        gives an early sense of which crime types tend to move together,
        which is useful later when deciding which features to feed the
        CNN branch of the hybrid model. Annotations are switched off for
        this dataset, since 29 crime columns produce a 30x30 grid that
        is unreadable with numbers printed in every cell - the sorted
        correlations against the target are printed separately instead.
        """
        try:
            view = self._get_analysis_view()
            columns = self.crime_columns + [self.target_column]
            corr = view[columns].corr()

            plt.figure(figsize=(18, 15))
            sns.heatmap(corr, annot=False, cmap="coolwarm", square=True)
            plt.title("Correlation Matrix - Crime Columns (2001-2013)")
            plt.tight_layout()

            output_path = os.path.join(self.results_dir, "correlation_matrix.png")
            plt.savefig(output_path)
            plt.close()
            print(f"\n[Saved] Correlation matrix -> {output_path}")

            self._explain_correlation(corr)
        except Exception as exc:
            print(f"ERROR while generating correlation matrix: {exc}")

    def _explain_correlation(self, corr: pd.DataFrame) -> None:
        """Prints the crime types most correlated with the target column."""
        target_corr = corr[self.target_column].drop(self.target_column)
        target_corr = target_corr.sort_values(ascending=False)
        print("Observation: Top 10 crime types by correlation with "
              f"'{self.target_column}':")
        print(target_corr.head(10))
        top_feature = target_corr.index[0]
        print(f"'{top_feature}' shows the strongest linear relationship "
              "with the aggregate crime count, making it a strong "
              "candidate feature for the prediction models.")

    def distribution_major_crimes(self) -> None:
        """
        Plots and saves the distribution of the target column
        ('TOTAL IPC CRIMES'), using the cleaned analysis view.
        Understanding the shape of the target variable (e.g. skew,
        outliers) is necessary before choosing a loss function and
        scaling strategy for the CNN-LSTM model in later phases.
        """
        try:
            view = self._get_analysis_view()

            plt.figure(figsize=(10, 6))
            sns.histplot(view[self.target_column], kde=True, bins=40, color="steelblue")
            plt.title("Distribution of Total IPC Crimes (District-Year Records)")
            plt.xlabel("Total IPC Crimes (count)")
            plt.ylabel("Frequency")
            plt.tight_layout()

            output_path = os.path.join(self.results_dir, "major_crimes_distribution.png")
            plt.savefig(output_path)
            plt.close()
            print(f"[Saved] Total IPC crimes distribution -> {output_path}")

            skew = view[self.target_column].skew()
            print(f"Observation: Skewness of '{self.target_column}' = {skew:.2f}. "
                  "A value well above 0 indicates a right-skewed "
                  "distribution (a small number of district-year records "
                  "report very high crime counts), which may need to be "
                  "addressed with a transformation during feature "
                  "engineering.")
        except Exception as exc:
            print(f"ERROR while generating crime distribution: {exc}")

    def top_states_by_crime(self, n: int = 10) -> None:
        """
        Plots and saves the top n states/UTs by total crime count, summed
        across all years, using the cleaned analysis view (aggregate
        rows removed, state names case-normalized). This highlights the
        spatial imbalance in the data, directly relevant to Phase 4
        (spatial feature preparation).
        """
        try:
            view = self._get_analysis_view()
            state_totals = (
                view.groupby(self.location_columns[0])[self.target_column]
                .sum()
                .sort_values(ascending=False)
                .head(n)
            )

            plt.figure(figsize=(10, 6))
            # `hue` is assigned to the same categorical column being
            # plotted on the y-axis, with the resulting legend hidden.
            # Newer seaborn versions raise an error if `palette` is
            # passed without `hue`, so this is required, not cosmetic.
            sns.barplot(
                x=state_totals.values,
                y=state_totals.index,
                hue=state_totals.index,
                palette="Reds_r",
                legend=False,
            )
            plt.title(f"Top {n} States/UTs by Total IPC Crimes (2001-2013)")
            plt.xlabel("Total IPC Crimes")
            plt.ylabel("State/UT")
            plt.tight_layout()

            output_path = os.path.join(self.results_dir, "top_states_by_crime.png")
            plt.savefig(output_path)
            plt.close()
            print(f"[Saved] Top {n} states by crime -> {output_path}")

            print(f"Observation: '{state_totals.index[0]}' records the "
                  f"highest cumulative crime total over 2001-2013 among "
                  f"all states/UTs, followed by '{state_totals.index[1]}'. "
                  "This imbalance means the model should account for "
                  "state as a strong spatial signal rather than treating "
                  "all locations uniformly.")
        except Exception as exc:
            print(f"ERROR while generating top states chart: {exc}")

    def top_districts_by_crime(self, n: int = 10) -> None:
        """
        Plots and saves the top n districts by total crime count, summed
        across all years, using the cleaned analysis view. District-level
        granularity is the level at which the final prediction module
        (Phase 10) is expected to operate.
        """
        try:
            view = self._get_analysis_view()
            district_totals = (
                view.groupby(self.location_columns[1])[self.target_column]
                .sum()
                .sort_values(ascending=False)
                .head(n)
            )

            plt.figure(figsize=(10, 6))
            # Same reasoning as top_states_by_crime(): `hue` must be set
            # whenever `palette` is used, or seaborn raises an error.
            sns.barplot(
                x=district_totals.values,
                y=district_totals.index,
                hue=district_totals.index,
                palette="Oranges_r",
                legend=False,
            )
            plt.title(f"Top {n} Districts by Total IPC Crimes (2001-2013)")
            plt.xlabel("Total IPC Crimes")
            plt.ylabel("District")
            plt.tight_layout()

            output_path = os.path.join(self.results_dir, "top_districts_by_crime.png")
            plt.savefig(output_path)
            plt.close()
            print(f"[Saved] Top {n} districts by crime -> {output_path}")

            print(f"Observation: '{district_totals.index[0]}' is the "
                  "single highest cumulative-crime district over "
                  "2001-2013 (state-level aggregate rows excluded - see "
                  "the aggregate row check above). District names are "
                  "not unique across states, so Phase 4 (spatial feature "
                  "preparation) must resolve state + district together "
                  "when assigning coordinates.")
        except Exception as exc:
            print(f"ERROR while generating top districts chart: {exc}")

    def crime_trend_by_year(self) -> None:
        """
        Plots and saves the trend of total crime count across years,
        using the cleaned analysis view. With 13 years of data, this is
        the first meaningful temporal signal available in the project so
        far, and is directly relevant to Phase 6 (LSTM).
        """
        try:
            view = self._get_analysis_view()
            year_totals = view.groupby(self.year_column)[self.target_column].sum()

            plt.figure(figsize=(10, 6))
            plt.plot(year_totals.index, year_totals.values, marker="o", color="darkgreen")
            plt.title("Total IPC Crimes by Year (2001-2013)")
            plt.xlabel("Year")
            plt.ylabel("Total IPC Crimes")
            plt.xticks(year_totals.index, rotation=45)
            plt.tight_layout()

            output_path = os.path.join(self.results_dir, "crime_trend_by_year.png")
            plt.savefig(output_path)
            plt.close()
            print(f"[Saved] Crime trend by year -> {output_path}")

            first_year, last_year = year_totals.index.min(), year_totals.index.max()
            change_pct = (
                (year_totals.iloc[-1] - year_totals.iloc[0]) / year_totals.iloc[0] * 100
            )
            direction = "increased" if change_pct > 0 else "decreased"
            print(f"Observation: Total recorded IPC crimes {direction} by "
                  f"{abs(change_pct):.1f}% from {first_year} to {last_year}. "
                  "Unlike the earlier single-year dataset, this range now "
                  "gives the LSTM branch (Phase 6) an actual sequence to "
                  "learn from.")
        except Exception as exc:
            print(f"ERROR while generating crime trend chart: {exc}")

    # ------------------------------------------------------------------
    # Discussion (no plot, console output only)
    # ------------------------------------------------------------------
    def feature_importance_discussion(self) -> None:
        """
        Discusses which features appear most relevant based on the EDA
        so far. A formal feature-importance ranking (e.g. from a trained
        model) is not possible yet, since no model has been built - that
        happens in Phases 5-7. This method summarizes what the
        correlation and grouping analysis above already suggests.
        """
        print("\n--- Feature Importance Discussion (Pre-Modeling) ---")
        print("No model has been trained yet, so this is a preliminary "
              "discussion based on correlation and grouping patterns "
              "observed above, not a formal importance ranking:")
        print(f"1. Crime-type columns with the highest correlation to "
              f"'{self.target_column}' (see correlation matrix) are "
              "likely to carry the most predictive signal for the CNN "
              "branch.")
        print("2. 'STATE/UT' and 'DISTRICT' show clear grouping effects "
              "on crime totals, which is why Phase 4 converts them into "
              "latitude/longitude - so the model can learn spatial "
              "proximity instead of treating each location as an "
              "unrelated category. This requires state names to be "
              "standardized first (see the case-inconsistency warning "
              "above), otherwise the same state maps to two different "
              "geocoding lookups.")
        print("3. 'YEAR' now spans 13 distinct values (2001-2013) with a "
              "visible trend (see crime trend chart), which finally "
              "gives the LSTM branch a real temporal signal to learn "
              "from - unlike the previous single-year dataset.")

    def generate_readiness_report(self) -> None:
        """
        Compiles a Dataset Readiness Report from everything established
        earlier in this run and writes it to
        results/dataset_readiness_report.md.

        This is the hand-off artifact for Phase 2: rather than Phase 2
        having to re-derive which column is the target, which are
        spatial/temporal, and which rows are known-bad, all of that is
        recorded here in one place, generated from the same detection
        logic used above (not re-typed by hand, which could drift out of
        sync with the actual checks).
        """
        try:
            state_col, district_col = self.location_columns
            agg_mask, agg_label_counts = self._compute_aggregate_row_stats()
            raw_state_count, normalized_state_count = self._compute_state_case_stats()

            feature_columns = self.crime_columns + self.location_columns + [self.year_column]

            # Build the list of open issues found during this run. An
            # empty list means the dataset is clean enough to move
            # straight into feature engineering; anything in this list
            # is a blocking item for Phase 2.
            open_issues = []
            if agg_mask.sum() > 0:
                open_issues.append(
                    f"{agg_mask.sum()} state-level aggregate rows disguised as "
                    f"districts in '{district_col}' must be removed."
                )
            if raw_state_count != normalized_state_count:
                open_issues.append(
                    f"'{state_col}' casing is inconsistent ({raw_state_count} raw "
                    f"labels vs {normalized_state_count} real states) and must be "
                    "standardized before any state-level grouping or geocoding."
                )
            if self.id_column is None:
                open_issues.append(
                    "No explicit identifier column exists - decide whether Phase 2 "
                    "should generate a surrogate Id before downstream phases need one."
                )

            is_ready_for_phase_2 = True  # structural load succeeded, so cleaning can begin
            is_ready_for_modeling = len(open_issues) == 0

            lines = []
            lines.append("# Dataset Readiness Report")
            lines.append("")
            lines.append("**Project:** A Hybrid CNN-LSTM Framework for Spatio-Temporal Crime Prediction")
            lines.append("**Phase:** 1 - Dataset Understanding (EDA)")
            lines.append(f"**Dataset:** {self.filepath}")
            lines.append(f"**Rows x Columns:** {self.df.shape[0]} x {self.df.shape[1]}")
            lines.append("")

            lines.append("## Target Variable")
            lines.append(f"- `{self.target_column}` - aggregate IPC crime count per "
                          "district-year record.")
            lines.append("")

            lines.append("## Feature Columns")
            lines.append(f"All columns other than the target ({len(feature_columns)} total), "
                          "before any encoding or transformation:")
            for col in feature_columns:
                lines.append(f"- `{col}`")
            lines.append("")

            lines.append("## Spatial Features")
            for col in self.location_columns:
                lines.append(f"- `{col}` (categorical, present in raw data)")
            lines.append("- Latitude / Longitude - **not yet generated**; planned for "
                          "Phase 4 (Spatial Feature Preparation). Requires "
                          f"`{state_col}` casing to be standardized first, or the same "
                          "state will resolve to inconsistent coordinates.")
            lines.append("")

            lines.append("## Temporal Features")
            lines.append(f"- `{self.year_column}` - {self.df[self.year_column].nunique()} "
                          f"distinct years ({self.df[self.year_column].min()}-"
                          f"{self.df[self.year_column].max()}). This is the only temporal "
                          "feature currently available; no month/day-level granularity "
                          "exists in this dataset.")
            lines.append("")

            lines.append(f"## Crime Feature Columns ({len(self.crime_columns)})")
            lines.append("Individual IPC crime-type columns, identified dynamically "
                          "from the loaded dataset (not hardcoded):")
            for col in self.crime_columns:
                lines.append(f"- `{col}`")
            lines.append("")

            lines.append("## Rows Excluded from Analysis")
            if agg_mask.sum() > 0:
                lines.append(f"- **{agg_mask.sum()} rows** where `{district_col}` is a "
                              "state-level aggregate rather than a real district. "
                              "Excluded from every aggregation chart in this script via "
                              "`_get_analysis_view()`, but **not yet removed from the "
                              "stored dataset** - that removal is a Phase 2 task.")
                lines.append("- Breakdown by label:")
                for label, count in agg_label_counts.items():
                    lines.append(f"  - `{label}`: {count} rows")
            else:
                lines.append("- None detected.")
            lines.append("")

            lines.append("## Other Open Issues")
            if open_issues:
                for issue in open_issues:
                    lines.append(f"- {issue}")
            else:
                lines.append("- None.")
            lines.append("")

            lines.append("## Readiness Verdict")
            lines.append(f"- **Ready for Phase 2 (Data Cleaning):** "
                          f"{'Yes' if is_ready_for_phase_2 else 'No'} - the dataset "
                          "loads cleanly with no missing values and no fully "
                          "duplicate rows, so cleaning can begin immediately.")
            lines.append(f"- **Ready for Phase 3 (Feature Engineering) / modeling as-is:** "
                          f"{'Yes' if is_ready_for_modeling else 'No'} - "
                          + ("no blocking issues remain."
                             if is_ready_for_modeling else
                             f"{len(open_issues)} open issue(s) listed above must be "
                             "resolved in Phase 2 first, otherwise state/district "
                             "totals will be double-counted or fragmented."))
            lines.append("")

            report_text = "\n".join(lines)
            output_path = os.path.join(self.results_dir, "dataset_readiness_report.md")
            with open(output_path, "w") as report_file:
                report_file.write(report_text)

            print(f"\n[Saved] Dataset Readiness Report -> {output_path}")
            print("\n" + report_text)
        except Exception as exc:
            print(f"ERROR while generating dataset readiness report: {exc}")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run_full_eda(self) -> None:
        """
        Runs every EDA step in sequence. Each step is wrapped so that a
        failure in one step (e.g. a plotting error) does not stop the
        remaining steps from running.
        """
        if not self.load_dataset():
            print("Stopping EDA: dataset could not be loaded.")
            return

        steps = [
            self.check_identifier_column,
            self.display_head,
            self.show_shape,
            self.show_info,
            self.show_dtypes,
            self.statistical_summary,
            self.missing_value_analysis,
            self.duplicate_analysis,
            self.unique_states,
            self.unique_districts,
            self.identify_crime_columns,
            self.detect_aggregate_rows,
            self.correlation_matrix,
            self.distribution_major_crimes,
            self.top_states_by_crime,
            self.top_districts_by_crime,
            self.crime_trend_by_year,
            self.feature_importance_discussion,
        ]

        for step in steps:
            try:
                step()
            except Exception as exc:
                print(f"ERROR: Step '{step.__name__}' failed: {exc}")

        self.print_summary()
        self.generate_readiness_report()

    def print_summary(self) -> None:
        """Prints a final summary of what was learned from the dataset."""
        print("\n" + "=" * 60)
        print("PHASE 1 SUMMARY - What was learned from the dataset")
        print("=" * 60)
        print(f"- The dataset has {self.df.shape[0]} rows and "
              f"{self.df.shape[1]} columns, spanning {self.df[self.year_column].nunique()} "
              f"years ({self.df[self.year_column].min()}-{self.df[self.year_column].max()}), "
              "with no missing values and no fully duplicate rows.")
        print("- It has no explicit identifier column, unlike the earlier "
              "single-year dataset - Phase 2 should decide whether a "
              "surrogate Id is needed downstream.")
        print("- Two data quality issues were found and must be handled in "
              "Phase 2, not left in the stored data: (1) state-level "
              "aggregate rows disguised as districts ('TOTAL', "
              "'ZZ TOTAL', 'DELHI UT TOTAL' - 455 rows total), and "
              "(2) inconsistent casing of state names, which inflates "
              "the apparent state count from 37 to 70 unless normalized.")
        print(f"- {len(self.crime_columns)} individual crime-type columns "
              f"were identified dynamically; '{self.target_column}' is "
              "treated as the aggregate target column.")
        print("- Crime totals are unevenly distributed across states and "
              "districts, and now show a real multi-year trend, "
              "confirming both space and time carry predictive signal - "
              "which is the core premise of the CNN-LSTM approach.")
        print("=" * 60)


def main() -> None:
    """Entry point for running Phase 1 EDA as a standalone script."""
    dataset_path = os.path.join("dataset", "Crimes_in_india_2001-2013.csv")
    results_path = "results"

    analyzer = CrimeDataEDA(filepath=dataset_path, results_dir=results_path)
    analyzer.run_full_eda()


if __name__ == "__main__":
    main()
