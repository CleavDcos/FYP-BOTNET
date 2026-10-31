# LinearSVC Binary Experiment

- **Target:** `attack`
- **Features:** 22 canonical features
- **Training rows used:** 10,334 of 2,567,965
- **Per-class cap:** 10,000
- **Training class counts:** `{'1': 10000, '0': 334}`
- **Validation rows:** 550,278 (full validation split)
- **Training time:** 0.128s
- **Model artifact:** `models\svm_binary_pipeline.joblib`

## Validation metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.999860 |
| Macro precision | 0.741611 |
| Macro recall | 0.999930 |
| Macro F1 | 0.825757 |
| Weighted F1 | 0.999884 |
| False positive rate | 0.000000 |

Confusion matrix label order: `[0, 1]`.

## Evaluation notes

The validation set was not sampled or resampled. Training used a fixed-seed per-class cap to make these algorithms practical on the multi-million-row dataset. Compare per-class results and support, especially for Normal and Theft, rather than relying on accuracy alone. The test split remains reserved for final evaluation after model selection.
