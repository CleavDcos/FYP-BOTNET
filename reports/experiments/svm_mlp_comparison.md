# SVM and MLP Validation Comparison

Both algorithms use the same canonical 22 features, a fixed-seed maximum of 10,000 training examples per class, and the complete untouched validation split. Validation was used for model comparison; the test split remains reserved for final evaluation.

## Results

| Model | Task | Train rows | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Train time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LinearSVC | Binary | 10,334 | 0.999860 | 0.741611 | 0.999930 | 0.825757 | 0.999884 | 0.128s |
| MLP | Binary | 10,334 | 0.999820 | 0.709579 | 0.986023 | 0.792842 | 0.999856 | 0.267s |
| LinearSVC | Category | 30,055 | 0.915523 | 0.756939 | 0.955845 | 0.797148 | 0.915505 | 0.730s |
| MLP | Category | 30,055 | 0.997119 | 0.868280 | 0.998191 | 0.910396 | 0.997124 | 4.290s |

## Class balance and interpretation

- Binary training used 10,000 Attack examples and all 334 Normal examples; validation has 550,206 Attack examples and only 72 Normal examples. The SVM false-positive rate was 0%; the MLP false-positive rate was 2.78% (2 of 72 Normal records).
- Category training used 10,000 each for DDoS, DoS, and Reconnaissance, plus all 55 Theft examples. Validation has 12 Theft examples, so Theft metrics have high variance.
- These capped-sample results are exploratory and are not final test results. After selecting a model and configuration using validation, evaluate it once on the untouched test split and report per-class metrics and supports.
- Feature transformations were fitted inside each model pipeline on its sampled training data. Sampling was deterministic with seed 42; the validation split was neither sampled nor resampled.

See the individual JSON and Markdown reports for confusion matrices, class supports, hyperparameters, and model artifact paths.
