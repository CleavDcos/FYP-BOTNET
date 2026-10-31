# MLPClassifier Category Experiment

- **Target:** `category`
- **Features:** 22 canonical features
- **Training rows used:** 30,055 of 2,567,631
- **Per-class cap:** 10,000
- **Training class counts:** `{'Reconnaissance': 10000, 'DoS': 10000, 'DDoS': 10000, 'Theft': 55}`
- **Validation rows:** 550,206 (full validation split)
- **Training time:** 4.290s
- **Model artifact:** `models\mlp_category_pipeline.joblib`

## Validation metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.997119 |
| Macro precision | 0.868280 |
| Macro recall | 0.998191 |
| Macro F1 | 0.910396 |
| Weighted F1 | 0.997124 |

Confusion matrix label order: `['DDoS', 'DoS', 'Reconnaissance', 'Theft']`.

## Evaluation notes

The validation set was not sampled or resampled. Training used a fixed-seed per-class cap to make these algorithms practical on the multi-million-row dataset. Compare per-class results and support, especially for Normal and Theft, rather than relying on accuracy alone. The test split remains reserved for final evaluation after model selection.
