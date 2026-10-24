# Stage 5: Binary XGBoost Imbalance Weighting Experiment

Controlled evaluation of class-weighting and sample-weighting strategies for binary botnet detection (`attack`: 0=Normal, 1=Attack).

- **Training Dataset**: `botnet_train_v2_clean.parquet` (2,567,965 samples: Normal=334, Attack=2,567,631)
- **Validation Dataset**: `botnet_val_v2_clean.parquet` (550,278 samples: Normal=72, Attack=550,206)
- **Test Dataset**: Completely untouched and never loaded.

## Overall Performance Comparison

| Model / Configuration | Accuracy | Normal Prec / Rec / F1 | Attack Prec / Rec / F1 | Macro-F1 | Weighted-F1 | FPR | ROC-AUC | Train Time |
|---|---|---|---|---|---|---|---|---|
| **RF (Balanced Baseline)** | 100.0000% | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.000000 | 1.000000 | 0.0000% | 1.000000 | 52.96s |
| **XGB (Unweighted Baseline)** | 99.9989% | 0.9583 / 0.9583 / 0.9583 | 1.0000 / 1.0000 / 1.0000 | 0.979164 | 0.999989 | 4.1667% | 1.000000 | 13.54s |
| **XGB (sample_weight_balanced)** | 99.9996% | 0.9730 / 1.0000 / 0.9863 | 1.0000 / 1.0000 / 1.0000 | 0.993150 | 0.999996 | 0.0000% | 1.000000 | 11.12s |
| **XGB (scale_pos_exact_ratio)** | 99.8762% | 0.0956 / 1.0000 / 0.1745 | 1.0000 / 0.9988 / 0.9994 | 0.586963 | 0.999273 | 0.0000% | 0.999987 | 8.48s |
| **XGB (scale_pos_mild_dampened)** | 99.9907% | 0.5854 / 1.0000 / 0.7385 | 1.0000 / 0.9999 / 1.0000 | 0.869208 | 0.999919 | 0.0000% | 0.999990 | 8.64s |
| **XGB (scale_pos_moderate_dampened)** | 99.9965% | 0.7912 / 1.0000 / 0.8834 | 1.0000 / 1.0000 / 1.0000 | 0.941709 | 0.999967 | 0.0000% | 0.999997 | 8.92s |

## Confusion Matrices

Order: `[[TN (Normal), FP], [FN, TP (Attack)]]`

### Baseline: Random Forest (Balanced)
- TN: 72, FP: 0, FN: 0, TP: 550206

### Baseline: XGBoost (Unweighted)
- TN: 69, FP: 3, FN: 3, TP: 550203

### Candidate: XGBoost (sample_weight_balanced)
- Description: compute_sample_weight('balanced', y_train) on training samples
- TN: 72, FP: 0, FN: 2, TP: 550204

### Candidate: XGBoost (scale_pos_exact_ratio)
- Description: scale_pos_weight=0.00013008 (N_neg / N_pos = 334 / 2567631)
- TN: 72, FP: 0, FN: 681, TP: 549525

### Candidate: XGBoost (scale_pos_mild_dampened)
- Description: scale_pos_weight=0.001 (mild dampening of attack class weight)
- TN: 72, FP: 0, FN: 51, TP: 550155

### Candidate: XGBoost (scale_pos_moderate_dampened)
- Description: scale_pos_weight=0.01 (moderate dampening of attack class weight)
- TN: 72, FP: 0, FN: 19, TP: 550187

