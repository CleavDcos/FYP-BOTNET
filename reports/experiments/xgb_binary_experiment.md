# Stage 5: XGBoost Binary Detection Experiment

- **Task**: Binary Botnet Detection (`attack`: 0=Normal, 1=Attack)
- **Input Features**: 22 Canonical Features from `config/settings.py`
- **Training Dataset**: `botnet_train_v2_clean.parquet` (2,567,965 samples)
- **Validation Dataset**: `botnet_val_v2_clean.parquet` (550,278 samples)
- **Training Class Counts**: Normal (0): 334, Attack (1): 2,567,631
- **Validation Class Counts**: Normal (0): 72, Attack (1): 550,206
- **Training Time**: 13.54s (0.23 mins)
- **Evaluation Time**: 1.98s
- **Model Saved Path**: `models\xgb_binary_pipeline.joblib` (0.13 MB)

## Validation Performance Summary

| Metric | Value |
|---|---|
| **Overall Accuracy** | 99.9989% |
| **Macro-F1** | 0.979164 |
| **Weighted-F1** | 0.999989 |
| **ROC-AUC** | 1.000000 |
| **Attack PR-AUC** | 1.000000 |
| **Normal PR-AUC** | 0.997951 |
| **False Positive Rate** | 4.1667% |

### Per-Class Metrics

| Class | Support | Precision | Recall | F1-Score | PR-AUC |
|---|---|---|---|---|---|
| **Normal (0)** | 72 | 0.958333 | 0.958333 | 0.958333 | 0.997951 |
| **Attack (1)** | 550206 | 0.999995 | 0.999995 | 0.999995 | 1.000000 |

### Confusion Matrix

| True \ Pred | Predicted Normal | Predicted Attack |
|---|---|---|
| **Actual Normal** | 69 (TN) | 3 (FP) |
| **Actual Attack** | 3 (FN) | 550203 (TP) |

### Top Feature Importances

| Feature | Importance |
|---|---|
| `TnP_PDstIP` | 0.802912 |
| `TnBPSrcIP` | 0.065295 |
| `TnP_PerProto` | 0.026792 |
| `state_CON` | 0.019355 |
| `N_IN_Conn_P_DstIP` | 0.011005 |
| `proto_udp` | 0.008194 |
| `stddev` | 0.007434 |
| `proto_arp` | 0.007375 |
| `TnBPDstIP` | 0.006886 |
| `proto_ipv6-icmp` | 0.006467 |

