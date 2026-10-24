# Stage 5B: Random Forest Attack-Category Classification Experiment

- **Task**: Multiclass Attack-Category Classification (Malicious traffic only)
- **Target Categories**: DDoS, DoS, Reconnaissance, Theft
- **Input Features**: 22 Canonical Features from `config/settings.py`
- **Training Dataset**: `botnet_train_v2_clean.parquet` (2,567,631 malicious samples)
- **Validation Dataset**: `botnet_val_v2_clean.parquet` (550,206 malicious samples)
- **Training Time**: 103.63s (1.73 mins)
- **Evaluation Time**: 12.88s
- **Model Saved Path**: `models\rf_category_pipeline.joblib` (10.42 MB)

## Data Sample Distribution

| Category | Training Count | Validation Count | Representation Status |
|---|---|---|---|
| **DDoS** | 1,348,005 | 289,275 | Standard |
| **DoS** | 1,155,814 | 247,257 | Standard |
| **Reconnaissance** | 63,757 | 13,662 | Standard |
| **Theft** | 55 | 12 | ⚠️ Rare Class (< 50 samples) |

## Validation Performance Summary

| Metric | Value |
|---|---|
| **Overall Accuracy** | 99.9942% |
| **Macro Precision** | 0.999744 |
| **Macro Recall** | 0.999971 |
| **Macro F1-Score** | 0.999857 |
| **Weighted Precision** | 0.999942 |
| **Weighted Recall** | 0.999942 |
| **Weighted F1-Score** | 0.999942 |

### Per-Class Metrics

| Category | Support | Precision | Recall | F1-Score | Reliable Evaluation |
|---|---|---|---|---|---|
| **DDoS** | 289,275 | 0.999983 | 0.999927 | 0.999955 | Yes |
| **DoS** | 247,257 | 0.999943 | 0.999956 | 0.999949 | Yes |
| **Reconnaissance** | 13,662 | 0.999049 | 1.000000 | 0.999524 | Yes |
| **Theft** | 12 | 1.000000 | 1.000000 | 1.000000 | ⚠️ No (High variance due to low support) |

### Data Imbalance & Evaluation Caveats

- **Theft**: Support of 12 samples is below the reliability threshold (50). Metrics should be interpreted with caution due to high variance.

### Confusion Matrix

Predicted Categories across columns: `DDoS, DoS, Reconnaissance, Theft`

| Actual \ Predicted | DDoS | DoS | Reconnaissance | Theft |
|---|---|---|---|---|
| **DDoS** | 289,254 | 14 | 7 | 0 |
| **DoS** | 5 | 247,246 | 6 | 0 |
| **Reconnaissance** | 0 | 0 | 13,662 | 0 |
| **Theft** | 0 | 0 | 0 | 12 |

### Top 15 Feature Importances

| Rank | Feature | Importance |
|---|---|---|
| 1 | `dur` | 0.198337 |
| 2 | `N_IN_Conn_P_DstIP` | 0.109957 |
| 3 | `TnP_Per_Dport` | 0.101322 |
| 4 | `rate` | 0.084197 |
| 5 | `TnP_PerProto` | 0.065656 |
| 6 | `stddev` | 0.053210 |
| 7 | `TnBPDstIP` | 0.050858 |
| 8 | `dport` | 0.043682 |
| 9 | `TnP_PDstIP` | 0.040049 |
| 10 | `N_IN_Conn_P_SrcIP` | 0.038477 |
| 11 | `mean` | 0.026708 |
| 12 | `pkts` | 0.022277 |
| 13 | `TnBPSrcIP` | 0.021168 |
| 14 | `dpkts` | 0.020156 |
| 15 | `spkts` | 0.019975 |

