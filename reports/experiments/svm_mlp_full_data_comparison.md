# Full-Data SVM and MLP Validation Comparison

Each model is trained independently. All eligible rows from the clean training split are used; there is no sampling or resampling. Results are measured on the complete validation split.

| Task | Model | Training rows | Validation macro-F1 | Accuracy | Training time |
|---|---|---:|---:|---:|---:|
| Binary | LinearSVC | 2,567,965 | 0.938263 | 0.999964 | 475.07s |
| Binary | MLPClassifier | 2,567,965 | 0.944436 | 0.999967 | 69.16s |
| Category | LinearSVC | 2,567,631 | 0.752831 | 0.912229 | 327.27s |
| Category | MLPClassifier | 2,567,631 | 0.898340 | 0.997388 | 424.46s |

## Method notes

- SVM and MLP are separate candidates; neither feeds into or combines with the other.
- Class weights address imbalance without changing the training rows.
- MLP retains its best validation macro-F1 checkpoint. LinearSVC uses L2 regularization; its convergence status is reported in its task reports.
- The held-out test split was not used for these full-data runs. Earlier test results in the repository refer to the older capped-sample model artifacts and do not describe these retrained models.
- Normal has 72 validation rows in binary detection; Theft has 12 in category classification. Interpret those class metrics cautiously.
