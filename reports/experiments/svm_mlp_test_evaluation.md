# Final Test Evaluation: SVM and MLP

These results use the held-out test split exactly once after candidate selection on validation. No model was fitted or tuned on test data.

## Selected models and results

| Task | Selected model | Test rows | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Binary | LinearSVC | 550,279 | 0.999862 | 0.741497 | 0.999931 | 0.825654 | 0.999886 |
| Category | MLPClassifier | 550,208 | 0.997101 | 0.848251 | 0.998172 | 0.891066 | 0.997109 |

## Class supports and selection basis

### Binary

- **Test class counts:** `{'1': 550208, '0': 71}`
- **Selection basis:** Highest validation macro-F1 among the SVM and MLP binary candidates (0.825757).
- **Prediction time:** 0.614s
- **Confusion matrix label order:** `[0, 1]`

- **False positive rate:** 0.000000 (0 false positives among 71 Normal records)

### Category

- **Test class counts:** `{'DDoS': 289344, 'DoS': 247189, 'Reconnaissance': 13663, 'Theft': 12}`
- **Selection basis:** Highest validation macro-F1 among the SVM and MLP category candidates (0.910396).
- **Prediction time:** 0.851s
- **Confusion matrix label order:** `['DDoS', 'DoS', 'Reconnaissance', 'Theft']`

## Interpretation notes

The test split contains very few Normal and Theft records. Their precision, recall, and F1 estimates can change sharply with just one or two errors. Read the per-class precision/recall/F1 and support values in the JSON alongside aggregate metrics. These test results estimate performance for the chosen validation-selected models; they do not replace evaluation on other networks or deployments.
