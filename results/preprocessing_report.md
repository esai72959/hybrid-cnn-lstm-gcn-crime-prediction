# Preprocessing Report

**Project:** A Hybrid CNN-LSTM Framework for Spatio-Temporal Crime Prediction
**Phase:** 2 - Data Preprocessing
**Source Dataset:** dataset\Crimes_in_india_2001-2013.csv
**Cleaned Dataset:** dataset\Crimes_in_india_2001-2013_cleaned.csv
**Final Shape:** 9385 rows x 34 columns

## Actions Taken
- Removed 455 state-level aggregate rows disguised as districts in 'DISTRICT'.
- Standardized 'STATE/UT' casing and whitespace, reducing unique labels from 70 to 37.
- Generated surrogate identifier column 'Id', since the raw dataset had none.

## Identifier Column
- `Id` (surrogate, generated during this phase).

## Readiness Verdict
- **Ready for Phase 3 (Feature Engineering):** Yes - all open issues from the Phase 1 Dataset Readiness Report have been resolved and the cleaned dataset has been persisted to disk.
