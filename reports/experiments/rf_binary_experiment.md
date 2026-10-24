# Stage 5: Random Forest Binary Detection Experiment

- **Task**: Binary Botnet Detection (`attack`: 0=Normal, 1=Attack)
- **Input Features**: 22 Canonical Features from `config/settings.py`
- **Training Dataset**: `botnet_train_v2_clean.parquet` (2,567,965 samples)
- **Validation Dataset**: `botnet_val_v2_clean.parquet` (550,278 samples)
- **Training Time**: 52.96s (0.88 mins)
- **Model Saved Path**: `models\rf_binary_pipeline.joblib` (0.59 MB)

## Validation Performance Summary

| Metric | Value |
|---|---|
| **Overall Accuracy** | 100.0000% |
| **ROC-AUC** | 1.000000 |
| **Attack PR-AUC** | 1.000000 |
| **Normal PR-AUC** | 1.000000 |
| **False Positive Rate** | 0.0000% |

### Per-Class Metrics

| Class | Support | Precision | Recall | F1-Score | PR-AUC |
|---|---|---|---|---|---|
| **Normal (0)** | 72 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| **Attack (1)** | 550206 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

### Confusion Matrix

| True \ Pred | Predicted Normal | Predicted Attack |
|---|---|---|
| **Actual Normal** | 72 (TN) | 0 (FP) |
| **Actual Attack** | 0 (FN) | 550206 (TP) |

