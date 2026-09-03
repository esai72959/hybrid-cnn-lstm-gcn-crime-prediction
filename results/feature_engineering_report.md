# Feature Engineering Report

## Dataset Shape
- Initial shape: (9385, 34)
- Final shape: (9385, 39)

## Encoded Columns
- ['STATE_ENCODED', 'DISTRICT_ENCODED']
- Method: sklearn LabelEncoder
- Original categorical columns (STATE/UT, DISTRICT) retained.

## Spatial Features Generated
- ['LATITUDE', 'LONGITUDE']
- Source: predefined State/UT centroid coordinate mapping.
- Unmapped State/UT values were assigned NaN (see console log).

## Temporal Features Generated
- ['YEAR', 'YEAR_INDEX']
- YEAR_INDEX is a zero-based sequential index derived from YEAR (e.g., 2001 -> 0, 2002 -> 1, ..., 2013 -> 12).

## Scaling Method
- MinMaxScaler (feature range 0-1)
- Number of columns scaled: 33
- Excluded from scaling: TOTAL IPC CRIMES, YEAR, YEAR_INDEX, Id

## Final Feature Count
- Total columns in final dataset: 39
- Total rows in final dataset: 9385

## Output Dataset Path
- `dataset\Crimes_in_india_2001-2013_features.csv`

## Readiness for Model Training
- All categorical variables are numerically encoded.
- Spatial (LATITUDE, LONGITUDE) and temporal (YEAR_INDEX) features are available for spatio-temporal sequence construction.
- Numerical predictors are scaled to [0, 1]; the target variable (TOTAL IPC CRIMES) remains in its original scale.
- Dataset is ready for consumption by cnn_model.py, lstm_model.py, and hybrid_model.py.
