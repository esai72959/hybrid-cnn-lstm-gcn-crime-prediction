# Dataset Readiness Report

**Project:** A Hybrid CNN-LSTM Framework for Spatio-Temporal Crime Prediction
**Phase:** 1 - Dataset Understanding (EDA)
**Dataset:** dataset\Crimes_in_india_2001-2013.csv
**Rows x Columns:** 9840 x 33

## Target Variable
- `TOTAL IPC CRIMES` - aggregate IPC crime count per district-year record.

## Feature Columns
All columns other than the target (32 total), before any encoding or transformation:
- `MURDER`
- `ATTEMPT TO MURDER`
- `CULPABLE HOMICIDE NOT AMOUNTING TO MURDER`
- `RAPE`
- `CUSTODIAL RAPE`
- `OTHER RAPE`
- `KIDNAPPING & ABDUCTION`
- `KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS`
- `KIDNAPPING AND ABDUCTION OF OTHERS`
- `DACOITY`
- `PREPARATION AND ASSEMBLY FOR DACOITY`
- `ROBBERY`
- `BURGLARY`
- `THEFT`
- `AUTO THEFT`
- `OTHER THEFT`
- `RIOTS`
- `CRIMINAL BREACH OF TRUST`
- `CHEATING`
- `COUNTERFIETING`
- `ARSON`
- `HURT/GREVIOUS HURT`
- `DOWRY DEATHS`
- `ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY`
- `INSULT TO MODESTY OF WOMEN`
- `CRUELTY BY HUSBAND OR HIS RELATIVES`
- `IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES`
- `CAUSING DEATH BY NEGLIGENCE`
- `OTHER IPC CRIMES`
- `STATE/UT`
- `DISTRICT`
- `YEAR`

## Spatial Features
- `STATE/UT` (categorical, present in raw data)
- `DISTRICT` (categorical, present in raw data)
- Latitude / Longitude - **not yet generated**; planned for Phase 4 (Spatial Feature Preparation). Requires `STATE/UT` casing to be standardized first, or the same state will resolve to inconsistent coordinates.

## Temporal Features
- `YEAR` - 13 distinct years (2001-2013). This is the only temporal feature currently available; no month/day-level granularity exists in this dataset.

## Crime Feature Columns (29)
Individual IPC crime-type columns, identified dynamically from the loaded dataset (not hardcoded):
- `MURDER`
- `ATTEMPT TO MURDER`
- `CULPABLE HOMICIDE NOT AMOUNTING TO MURDER`
- `RAPE`
- `CUSTODIAL RAPE`
- `OTHER RAPE`
- `KIDNAPPING & ABDUCTION`
- `KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS`
- `KIDNAPPING AND ABDUCTION OF OTHERS`
- `DACOITY`
- `PREPARATION AND ASSEMBLY FOR DACOITY`
- `ROBBERY`
- `BURGLARY`
- `THEFT`
- `AUTO THEFT`
- `OTHER THEFT`
- `RIOTS`
- `CRIMINAL BREACH OF TRUST`
- `CHEATING`
- `COUNTERFIETING`
- `ARSON`
- `HURT/GREVIOUS HURT`
- `DOWRY DEATHS`
- `ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY`
- `INSULT TO MODESTY OF WOMEN`
- `CRUELTY BY HUSBAND OR HIS RELATIVES`
- `IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES`
- `CAUSING DEATH BY NEGLIGENCE`
- `OTHER IPC CRIMES`

## Rows Excluded from Analysis
- **455 rows** where `DISTRICT` is a state-level aggregate rather than a real district. Excluded from every aggregation chart in this script via `_get_analysis_view()`, but **not yet removed from the stored dataset** - that removal is a Phase 2 task.
- Breakdown by label:
  - `TOTAL`: 408 rows
  - `ZZ TOTAL`: 35 rows
  - `DELHI UT TOTAL`: 12 rows

## Other Open Issues
- 455 state-level aggregate rows disguised as districts in 'DISTRICT' must be removed.
- 'STATE/UT' casing is inconsistent (70 raw labels vs 37 real states) and must be standardized before any state-level grouping or geocoding.
- No explicit identifier column exists - decide whether Phase 2 should generate a surrogate Id before downstream phases need one.

## Readiness Verdict
- **Ready for Phase 2 (Data Cleaning):** Yes - the dataset loads cleanly with no missing values and no fully duplicate rows, so cleaning can begin immediately.
- **Ready for Phase 3 (Feature Engineering) / modeling as-is:** No - 3 open issue(s) listed above must be resolved in Phase 2 first, otherwise state/district totals will be double-counted or fragmented.
