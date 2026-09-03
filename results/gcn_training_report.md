# CrimeGCN Branch Training Report

## Architecture
- Model Type: Native Spectral Graph Convolutional Network (Keras Layer)
- Total Graph Nodes: 850 composite (STATE/UT, DISTRICT) nodes
- Input Features: 33 engineered predictors per node
- Output Embedding Dimension: 32 (named `gcn_embedding`)

## Layer Pipeline
```
Input(850, 33)
  -> GraphConvLayer(64, HeNormal, L2(1e-4)) -> BatchNorm -> ReLU -> Dropout(0.3)
  -> GraphConvLayer(32, HeNormal, L2(1e-4)) -> BatchNorm -> ReLU -> Dropout(0.3)
  -> Dense(32, ReLU) -> BatchNorm [gcn_embedding output: (850, 32)]
  -> Dense(1, Linear) [Auxiliary head for pretraining]
```

## Auxiliary Pretraining Note
- Note: The standalone GCN auxiliary head is trained on 13 annual full-graph snapshots (2001-2013).
- Its purpose is auxiliary feature pretraining to condition the 32-dim spatial-graph embedding.
- Final predictive accuracy is evaluated on the 3-way fused model in `hybrid_model_v2.py`.

## Auxiliary Training Metrics
- Final Training Loss (MSE): 13531018.0000
- Final Validation Loss (MSE): 20631378.0000
- Best Validation Loss (MSE): 20573178.0000

## Artifacts
- GCN Model: `models\gcn_model.keras`
- GCN Feature Extractor: `models\gcn_feature_extractor.keras`
- Loss Curve: `results\gcn_loss_curve.png`
