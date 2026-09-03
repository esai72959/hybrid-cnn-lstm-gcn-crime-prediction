# CrimeCNN Training Report

## Dataset
- Source file: `dataset\Crimes_in_india_2001-2013_features.csv`
- Dataset shape: 9385 rows x 39 columns
- Input feature count: 33
- Target column: `TOTAL IPC CRIMES`
- Train / Val / Test samples: 6569 / 1877 / 939

## Feature Columns Used
```
MURDER, ATTEMPT TO MURDER, CULPABLE HOMICIDE NOT AMOUNTING TO MURDER, RAPE, CUSTODIAL RAPE, OTHER RAPE, KIDNAPPING & ABDUCTION, KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS, KIDNAPPING AND ABDUCTION OF OTHERS, DACOITY, PREPARATION AND ASSEMBLY FOR DACOITY, ROBBERY, BURGLARY, THEFT, AUTO THEFT, OTHER THEFT, RIOTS, CRIMINAL BREACH OF TRUST, CHEATING, COUNTERFIETING, ARSON, HURT/GREVIOUS HURT, DOWRY DEATHS, ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY, INSULT TO MODESTY OF WOMEN, CRUELTY BY HUSBAND OR HIS RELATIVES, IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES, CAUSING DEATH BY NEGLIGENCE, OTHER IPC CRIMES, STATE_ENCODED, DISTRICT_ENCODED, LATITUDE, LONGITUDE
```

## Architecture
```
Input(n_features, 1)
  -> Conv1D(64, kernel=3) -> BatchNorm -> ReLU -> MaxPooling1D(2)
  -> Conv1D(128, kernel=3) -> BatchNorm -> ReLU
  -> GlobalAveragePooling1D
  -> Dense(128, relu) -> Dropout(0.3)
  -> Dense(64, relu)   [Spatial Embedding Output]
  -> Dense(1, linear)  [Auxiliary head, training only]
```

## Training Configuration
- Optimizer: Adam (learning_rate=0.001)
- Loss: MeanSquaredError
- Metric: MAE
- Epochs (max): 30
- Epochs run (early stopping): 30
- Batch size: 32

## Results
- Final training loss (MSE): 2797223.5000
- Final validation loss (MSE): 632638.7500
- Best validation loss (MSE): 632638.7500
- Test loss (MSE): 795133.7500
- Test MAE: 515.1146

## Embedding
- Embedding dimension: 64

## Artifacts
- Trained model: `models\cnn_feature_extractor.keras`
- Loss curve: `results\cnn_loss_curve.png`
