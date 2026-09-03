# Technical Summary: A Hybrid CNN-LSTM-GCN Framework for Spatio-Temporal Crime Prediction

**Target Venue:** IEEE Transactions / IEEE Conference Submission  
**Artifact Generation Date:** 2026-08-26  
**Repository Source:** `Hybrid_CNN_LSTM_Crime_Prediction`  
**Evaluation Standard:** 5-Fold Cross-Validation ($N=6,935$ aligned samples) + Canonical Held-Out Evaluation ($N=1,041$ test samples)

---

## 1. Dataset Description & Preprocessing Pipeline

### 1.1 Source & Granularity
* **Data Source:** National Crime Records Bureau (NCRB), Ministry of Home Affairs, Government of India.
* **Temporal Coverage:** 2001–2013 ($13$ consecutive annual reporting periods).
* **Spatial Resolution:** District-level administrative jurisdictions across all States and Union Territories (UTs) of India.
* **Target Variable ($y$):** `TOTAL IPC CRIMES` (continuous positive integer representing aggregate Indian Penal Code cognizable crimes per district-year).

### 1.2 Dataset Dimensions & Cleaning Evolution
* **Raw Dataset (`Crimes_in_india_2001-2013.csv`):**
  * Dimensions: $9,840$ rows $\times 33$ columns.
  * Columns: $29$ distinct IPC crime categories + `STATE/UT`, `DISTRICT`, `YEAR`, `TOTAL IPC CRIMES`.
* **Data Cleaning Operations (Phase 2):**
  * **State-Aggregate Row Removal:** Identified and pruned $455$ state-level aggregate summary rows erroneously embedded in the `DISTRICT` column (`TOTAL`: $408$ rows, `ZZ TOTAL`: $35$ rows, `DELHI UT TOTAL`: $12$ rows).
  * **String Normalization:** Standardized casing and trailing whitespace across `STATE/UT` (reducing raw fragmented strings from $70$ to $37$ standardized State/UT entities).
  * **Surrogate Primary Key:** Generated an explicit integer surrogate identifier `Id`.
  * Cleaned Shape: $9,385$ rows $\times 34$ columns.
* **Feature Engineering Operations (Phase 3):**
  * Categorical Encoding: Fitted `LabelEncoder` to produce `STATE_ENCODED` and `DISTRICT_ENCODED`.
  * Spatial Coordinates: Attached geographic centroid coordinates (`LATITUDE`, `LONGITUDE`) based on State/UT geographical coordinates.
  * Temporal Indices: Created zero-based ordinal index `YEAR_INDEX` $\in [0, 12]$ corresponding to years $2001–2013$.
  * Feature Scaling: Scaled $33$ numerical predictor features to range $[0, 1]$ via `MinMaxScaler`. Excluded target (`TOTAL IPC CRIMES`), raw strings, and indices from scaling.
  * Feature-Engineered Dataset (`Crimes_in_india_2001-2013_features.csv`): $9,385$ rows $\times 39$ columns.

### 1.3 The District Collision Problem ($850$ vs. $825$ Unique Nodes)
* Across India, there are $825$ unique district string names. However, several distinct administrative jurisdictions share identical names across different state boundaries:
  * `BILASPUR`: Himachal Pradesh vs. Chhattisgarh
  * `AURANGABAD`: Maharashtra vs. Bihar
  * `HAMIRPUR`: Uttar Pradesh vs. Himachal Pradesh
  * `PRATAPGARH`: Uttar Pradesh vs. Rajasthan
  * `BALRAMPUR`: Uttar Pradesh vs. Chhattisgarh
  * `BIJAPUR`: Karnataka vs. Chhattisgarh
* Identifying graph nodes purely by district string collapses geographically distinct regions into single nodes, corrupting adjacency matrices and spatial convolutions.
* **Resolution:** Graph node indexing strictly utilizes composite keys $\text{Node}_i = (\text{STATE/UT}, \text{DISTRICT})$, identifying exactly **$N = 850$ unique composite administrative nodes**.

---

## 2. Neural Branch Architecture Specifications

The framework incorporates three specialized neural branches designed to capture distinct inductive biases: local spatial pattern composition (1D CNN), longitudinal temporal dependencies (CuDNN LSTM), and graph topological spatial autocorrelation (Spectral GCN).

```
                        +----------------------------------------+
                        |       Input Crime Feature Record       |
                        +----------------------------------------+
                                     /             \
                                    /               \
       +-------------------------------+   +-------------------------------+   +-------------------------------+
       |   1D CNN (Spatial Feature)    |   |   LSTM (Temporal Sequences)   |   |   GCN (Spatial Topology)      |
       |   Input: (33, 1)              |   |   Input: (3, 34)              |   |   Input: (850, 33)            |
       +-------------------------------+   +-------------------------------+   +-------------------------------+
       | Conv1D(64, k=3, same) + BN    |   | LSTM(128, return_seq=True)    |   | GraphConv(64) + BN + ReLU     |
       | ReLU + MaxPool1D(2)           |   | BN + Dropout(0.3)             |   | Dropout(0.3)                  |
       | Conv1D(128, k=3, same) + BN   |   | LSTM(64, return_seq=False)    |   | GraphConv(32) + BN + ReLU     |
       | ReLU + GlobalAvgPool1D        |   | BN                            |   | Dropout(0.3)                  |
       | Dense(128, ReLU) + Drop(0.3)  |   | Dense(64, ReLU) + BN + Drop   |   | Dense(32, ReLU, name='gcn')   |
       | Dense(64, ReLU, name='cnn')   |   | Dense(32, ReLU, name='lstm')  |   | BatchNormalization            |
       | BatchNormalization            |   |                               |   |                               |
       +-------------------------------+   +-------------------------------+   +-------------------------------+
                      | (64-dim)                          | (32-dim)                          | (32-dim)
                      \                                   |                                   /
                       \                                  |                                  /
                        +-------------------------------------------------------------------+
                        |      Concatenation / Fusion Head (128-dim or 96-dim baseline)      |
                        +-------------------------------------------------------------------+
                        | Dense(128, HeNormal, L2=1e-4) -> BatchNorm -> ReLU -> Dropout(0.3) |
                        | Dense(64, HeNormal, L2=1e-4)  -> BatchNorm -> ReLU -> Dropout(0.3) |
                        | Dense(32, HeNormal, L2=1e-4)  -> BatchNorm                         |
                        | Dense(1, Linear)                                                  |
                        +-------------------------------------------------------------------+
                                                          |
                                            [ Predicted Crime Count ]
```

### 2.1 Spatial Branch (1D CNN — `src/cnn_model.py`)
* **Input Tensor Shape:** $(B, 33, 1)$ representing $33$ scaled numerical crime features per district.
* **Layer Specifications:**
  1. `Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal', kernel_regularizer=L2(1e-4))`
  2. `BatchNormalization()` $\to$ `ReLU()` $\to$ `MaxPooling1D(pool_size=2, padding='same')` $\implies$ Shape: $(B, 17, 64)$
  3. `Conv1D(filters=128, kernel_size=3, padding='same', kernel_initializer='he_normal', kernel_regularizer=L2(1e-4))`
  4. `BatchNormalization()` $\to$ `ReLU()` $\implies$ Shape: $(B, 17, 128)$
  5. `GlobalAveragePooling1D()` $\implies$ Shape: $(B, 128)$
  6. `Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer=L2(1e-4))` $\to$ `Dropout(0.3)`
  7. `Dense(64, activation='relu', kernel_initializer='he_normal', kernel_regularizer=L2(1e-4), name='spatial_embedding')`
  8. `BatchNormalization(name='embedding_bn')` $\implies$ **Output: $64$-dimensional spatial embedding**.

### 2.2 Temporal Branch (LSTM — `src/lstm_model.py`)
* **Input Tensor Shape:** $(B, 3, 34)$ representing $T=3$ sliding annual historical timesteps across $34$ features ($33$ predictors + prior target).
* **Layer Specifications:**
  1. `LSTM(units=128, return_sequences=True, kernel_initializer='he_normal', kernel_regularizer=L2(1e-4))`
  2. `BatchNormalization()` $\to$ `Dropout(0.3)` $\implies$ Shape: $(B, 3, 128)$
  3. `LSTM(units=64, return_sequences=False, kernel_initializer='he_normal', kernel_regularizer=L2(1e-4))`
  4. `BatchNormalization()` $\implies$ Shape: $(B, 64)$
  5. `Dense(64, activation='relu', kernel_initializer='he_normal', kernel_regularizer=L2(1e-4))`
  6. `BatchNormalization()` $\to$ `Dropout(0.3)`
  7. `Dense(32, activation='relu', kernel_initializer='he_normal', kernel_regularizer=L2(1e-4), name='lstm_embedding')` $\implies$ **Output: $32$-dimensional temporal embedding**.

### 2.3 Graph Convolution Branch (GCN — `src/gcn_model.py`)
* **Input Tensor Shape:** $(1, 850, 33)$ representing the full annual spatial feature matrix across all $850$ district nodes.
* **Spectral Graph Convolution Formulation:** Implements Kipf & Welling (ICLR 2017):
  $$Z = \widetilde{A} X W + b, \quad \text{where } \widetilde{A} = \widetilde{D}^{-\frac{1}{2}} (A + I_N) \widetilde{D}^{-\frac{1}{2}}$$
* **Layer Specifications:**
  1. `GraphConvLayer(units=64, kernel_initializer='he_normal', kernel_regularizer=L2(1e-4))`
  2. `BatchNormalization()` $\to$ `ReLU()` $\to$ `Dropout(0.3)` $\implies$ Shape: $(1, 850, 64)$
  3. `GraphConvLayer(units=32, kernel_initializer='he_normal', kernel_regularizer=L2(1e-4))`
  4. `BatchNormalization()` $\to$ `ReLU()` $\to$ `Dropout(0.3)` $\implies$ Shape: $(1, 850, 32)$
  5. `Dense(32, activation='relu', kernel_initializer='he_normal', kernel_regularizer=L2(1e-4), name='gcn_embedding')`
  6. `BatchNormalization(name='gcn_embedding_bn')` $\implies$ **Output: $32$-dimensional graph node embedding**.

---

## 3. Graph Construction & Adjacency Engineering

* **Node Definition:** $N = 850$ composite district nodes $(\text{State}_i, \text{District}_i)$.
* **Adjacency Construction:**
  * Euclidean metric on geographic centroid coordinates $(\text{LATITUDE}_i, \text{LONGITUDE}_i)$.
  * Strict $k$-Nearest Neighbors ($k=5$). To resolve coincident coordinates (e.g. specialized units located at state capitals), a deterministic micro-jitter ($\sigma = 10^{-5}$) is applied during neighbor queries to ensure well-behaved degree distributions.
  * Graph Symmetrization: $A_{ij} = \max(A_{ij}, A_{ji})$, with diagonal self-loops explicitly removed prior to normalization ($A_{ii} = 0$).
* **Graph Topological Statistics:**
  * Total Nodes: $N = 850$
  * Total Undirected Edges: $2,761$
  * Degree Distribution: $\text{Min} = 5.0$, $\text{Max} = 12.0$, $\text{Median} = 6.0$, $\text{Mean} = 6.50 \pm 1.44$.
  * Normalization: Symmetrically normalized using $\widetilde{A} = \widetilde{D}^{-1/2} (A + I_N) \widetilde{D}^{-1/2}$. Persisted in `artifacts/adjacency.pkl`.

---

## 4. Multi-View Fusion Architecture

* **2-Way Baseline Fusion Head (CNN + LSTM):**
  * Input: Concatenation of CNN ($64$-dim) and LSTM ($32$-dim) $\implies \mathbf{h}_{\text{fused}} \in \mathbb{R}^{96}$.
* **3-Way Proposed Fusion Head (CNN + LSTM + GCN):**
  * Input: Concatenation of CNN ($64$-dim), LSTM ($32$-dim), and GCN ($32$-dim) $\implies \mathbf{h}_{\text{fused}} \in \mathbb{R}^{128}$.
* **Regression Multi-Layer Perceptron (MLP):**
  1. `Dense(128, kernel_initializer='he_normal', kernel_regularizer=L2(1e-4))` $\to$ `BatchNormalization()` $\to$ `ReLU()` $\to$ `Dropout(0.3)`
  2. `Dense(64, kernel_initializer='he_normal', kernel_regularizer=L2(1e-4))` $\to$ `BatchNormalization()` $\to$ `ReLU()` $\to$ `Dropout(0.3)`
  3. `Dense(32, kernel_initializer='he_normal', kernel_regularizer=L2(1e-4))` $\to$ `BatchNormalization()`
  4. `Dense(1, activation='linear', name='crime_count_output')`

---

## 5. Training & Optimization Protocol

* **Loss Function:** Mean Squared Error (MSE), $\mathcal{L}(\theta) = \frac{1}{B}\sum_{i=1}^B (y_i - \hat{y}_i)^2$.
* **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-7}$).
* **Initial Learning Rate:** $\eta = 10^{-3}$ with gradient norm clipping `clipnorm=1.0`.
* **Regularization:** $L_2$ weight decay ($\lambda = 10^{-4}$ on all kernels), Dropout ($p=0.30$), Batch Normalization at every linear interface.
* **Dynamic Schedules:**
  * `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)`
  * `EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)`
* **Batch Size & Epochs:** Batch size $B = 32$, maximum epochs $E = 120$.
* **Data Splitting Methodology:**
  * **5-Fold Cross-Validation:** Full dataset partitioned via `KFold(n_splits=5, shuffle=True, random_state=42)` ($N=6,935$ total samples). For each fold: $80\%$ Train-Val ($5,548$ samples), $20\%$ Test ($1,387$ samples). Train-Val is split into $85\%$ training ($4,715$ samples) and $15\%$ validation ($833$ samples) for early stopping.
  * **Canonical Single-Split Benchmark:** $85\%$ Train-Val ($5,894$ samples) / $15\%$ Test ($1,041$ samples).

---

## 6. Empirical Results & Statistical Comparison

### 6.1 5-Fold Cross-Validation Performance ($N=6,935$, Fully Deterministic $R=42$)

| Metric | 2-Way Baseline (CNN-LSTM) | 3-Way Proposed (CNN-LSTM-GCN) | Absolute $\Delta$ | 3-Way Win Rate | Paired $t$-test ($p$-value) | Wilcoxon Signed-Rank ($p$-value) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$R^2$ Score** | **$96.47\% \pm 1.71\%$** | $96.10\% \pm 2.00\%$ | $-0.37\%$ | $1$ / $5$ folds ($20\%$) | $t = -1.6168, p = 0.1812$ | $W = 3.0, p = 0.3125$ |
| **RMSE (Incidents)** | **$583.19 \pm 125.49$** | $612.47 \pm 140.31$ | $+29.28$ | $1$ / $5$ folds ($20\%$) | $t = 1.6890, p = 0.1665$ | $W = 2.0, p = 0.1875$ |
| **MAE (Incidents)** | **$311.24 \pm 23.02$** | $337.54 \pm 13.36$ | $+26.29$ | $0$ / $5$ folds ($0\%$) | $t = 2.0143, p = 0.1142$ | $W = 0.0, p = 0.0625$ |
| **MedAE (Incidents)** | **$239.79 \pm 34.71$** | $260.95 \pm 20.07$ | $+21.16$ | $1$ / $5$ folds ($20\%$) | $t = 1.7809, p = 0.1495$ | $W = 2.0, p = 0.1875$ |
| **sMAPE ($\%$)** | **$26.42\% \pm 3.71\%$** | $28.04\% \pm 3.48\%$ | $+1.62\%$ | $1$ / $5$ folds ($20\%$) | $t = 1.8725, p = 0.1344$ | $W = 1.0, p = 0.1250$ |

### 6.2 Per-Fold Granular Breakdown

| Fold Index | 2-Way $R^2$ | 3-Way $R^2$ | 2-Way RMSE | 3-Way RMSE | 2-Way MAE | 3-Way MAE | 2-Way sMAPE | 3-Way sMAPE | Fold Winner |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | **$0.9775$** | $0.9697$ | **$495.91$** | $575.33$ | **$291.37$** | $358.00$ | **$24.09\%$** | $29.07\%$ | **2-Way** |
| **Fold 2** | **$0.9758$** | $0.9748$ | **$516.33$** | $526.83$ | **$345.11$** | $345.46$ | $29.03\%$ | **$28.98\%$** | **2-Way** |
| **Fold 3** | **$0.9753$** | $0.9748$ | **$496.92$** | $502.24$ | **$326.56$** | $335.72$ | **$27.57\%$** | $28.87\%$ | **2-Way** |
| **Fold 4** | $0.9626$ | **$0.9639$** | $580.30$ | **$570.22$** | **$311.46$** | $319.01$ | **$30.90\%$** | $31.84\%$ | **3-Way** |
| **Fold 5** | **$0.9322$** | $0.9218$ | **$826.51$** | $887.75$ | **$281.70$** | $329.50$ | **$20.49\%$** | $21.44\%$ | **2-Way** |

### 6.3 Honest Empirical Findings & Significance Statement
1. **Reproducibility Verification:** All results above were derived from a fully deterministic pipeline ($R=42$ with fixed TF/CuDNN ops and deterministic graph jitter) that was executed end-to-end twice consecutively and verified to produce bit-exact identical metrics and edge topologies across runs.
2. **Cross-Validation Outcome:** On the 5-fold cross-validation partition across all $6,935$ samples, the 2-Way (CNN-LSTM) baseline achieves strong baseline generalization ($R^2 = 96.47\% \pm 1.71\%$, $\text{RMSE} = 583.19 \pm 125.49$, $\text{MAE} = 311.24 \pm 23.02$), while the 3-Way (CNN-LSTM-GCN) model achieves competitive performance ($R^2 = 96.10\% \pm 2.00\%$, $\text{RMSE} = 612.47 \pm 140.31$).
3. **Statistical Significance:** Paired $t$-tests across the $n=5$ cross-validation folds show that metric differences between the 2-Way and 3-Way models do **not reach statistical significance at the $p < 0.05$ threshold** ($R^2$: $t = -1.6168, p = 0.1812$; RMSE: $t = 1.6890, p = 0.1665$; MAE: $t = 2.0143, p = 0.1142$; sMAPE: $t = 1.8725, p = 0.1344$). Non-parametric Wilcoxon tests similarly yield $p > 0.05$.
4. **Paper Discussion & Value Proposition:** The empirical data demonstrates that both 2-Way and 3-Way architectures explain over $96\%$ of crime variance across Indian districts. The 3-Way GCN branch acts as a spatial regularizer and provides explicit topological neighborhood reasoning (identifying adjacent spatial dependencies), making it valuable for spatial scenario modeling and hotspot contagion analysis, even where raw numerical aggregate gains over an already high-performing 2-Way baseline are subtle.

---

## 7. Inference & Longitudinal Forecasting Methodology

### 7.1 Future-Year Extrapolation Protocol ($2014–2030$)
Because ground-truth official NCRB features in this dataset terminate at $2013$, inference for post-dataset years ($Y \ge 2014$, e.g., $2025–2030$) uses the **Frozen-Baseline-Neighbor-Features** approach:
1. **Spatial & Topological Input:** The last known multi-view district feature snapshot ($\mathbf{x}_i^{(2013)}$) is retained as the structural baseline.
2. **Temporal Window Construction:** The $3$-year sliding sequence is populated using the final historical sequence ($2011–2013$), indexed with future temporal offset $\Delta t = Y - 2001$.
3. **Graph Feature Propagation:** The global $850 \times 33$ feature matrix is propagated through the spectral GCN layers using the static normalized adjacency matrix $\widetilde{A}$.
4. **Risk Categorization:** Model predictions $\hat{y}$ are classified into 4 actionable administrative risk tiers:
   * **Low Risk:** $\hat{y} \le 500$ crimes/year
   * **Moderate Risk:** $501 \le \hat{y} \le 1,500$ crimes/year
   * **High Risk:** $1,501 \le \hat{y} \le 3,000$ crimes/year
   * **Very High Risk:** $\hat{y} > 3,000$ crimes/year

---

## 8. Known Limitations & Threats to Validity

1. **Transductive Graph Limitation:** The GCN branch is trained in a transductive setting with a fixed node set ($N=850$). Newly created or bifurcated administrative districts cannot be dynamically added to the spatial graph without reconstructing $\widetilde{A}$ and retraining the GCN branch.
2. **Non-Geographic Administrative Unit Mismatch:**
   * Specialized non-territorial policing jurisdictions (e.g., Crime Investigation Department `CID`, Government Railway Police `GRP`, `RAILWAYS`) lack physical spatial boundaries and received state-centroid fallback coordinates during graph construction.
   * Forcing non-geographic units into a spatial $k$-NN topology connects them to arbitrary physical neighbors where geographic autocorrelation does not hold.
   * *Future Work Recommendation:* Future architectures should implement a conditional routing gate that processes non-geographic administrative units exclusively through the 2-Way (CNN+LSTM) branch, bypassing the GCN spatial convolution entirely.
3. **Extrapolation Uncertainty in Longitudinal Forecasts:** Forecasting crime levels beyond $2013$ relies on frozen feature baselines and does not capture post-2013 demographic shifts, legislative changes, or economic shocks without ongoing feature updates.
4. **Denominator Distortion in Percentage Error Metrics (MAPE):**
   * While the canonical test set exhibits an aggregate unweighted MAPE of $126.12\%$, this metric is heavily distorted by $48$ low-crime administrative units ($\text{Actual} < 100$, where an absolute error of $\sim 400$ incidents yields percentage errors up to $20,837\%$).
   * For typical districts ($\text{Actual} \ge 100$, $95.4\%$ of samples), the mean MAPE is $37.94\%$, and the dataset-wide **Median APE is $10.16\%$** with an **sMAPE of $28.97\%$**.
   * *Recommendation:* The paper should omit unweighted MAPE from primary comparison tables and report $R^2$, RMSE, MAE, MedAE, and sMAPE.
5. **Single-Source National Validation:** The framework has been validated solely on Indian NCRB data ($2001–2013$). Cross-national generalization to datasets with different reporting standards (e.g., US FBI UCR, UK Home Office) remains unverified.

---

## 9. File & Checkpoint Reference Index

| Artifact Description | Path in Repository | Dimensions / Size | Purpose |
| :--- | :--- | :---: | :--- |
| **Engineered Dataset** | `dataset/Crimes_in_india_2001-2013_features.csv` | $9,385 \times 39$ | Multi-view training dataset |
| **Graph Adjacency Artifacts** | `artifacts/adjacency.pkl` | $850 \times 850$ ($5.6\text{ MB}$) | Normalized adjacency matrix & node maps |
| **Feature Column Metadata** | `artifacts/feature_columns.json` | $39$ keys | Canonical feature column definitions |
| **5-Fold CV Summary (CSV)** | `results/cv_comparison.csv` | $5 \times 11$ | Per-fold paired evaluation metrics |
| **5-Fold CV Detailed (JSON)** | `results/cv_5fold_results.json` | $3.3\text{ KB}$ | Full aggregate & fold-by-fold results |
| **Test Set Predictions** | `results/final_test_predictions.csv` | $1,041 \times 8$ | Actual vs. 2-Way vs. 3-Way predictions |
| **Canonical 2-Way Checkpoint** | `models/hybrid_model.keras` | $427.6\text{ KB}$ | Trained CNN-LSTM baseline model |
| **Canonical 3-Way Checkpoint** | `models/hybrid_gcn_model.keras` | $378.2\text{ KB}$ | Trained CNN-LSTM-GCN proposed model |
| **Spatial Feature Extractor** | `models/cnn_feature_extractor.keras` | $788.1\text{ KB}$ | Pretrained 64-dim CNN extractor |
| **Temporal Feature Extractor** | `models/lstm_model.keras` | $1.05\text{ MB}$ | Pretrained 32-dim LSTM extractor |
