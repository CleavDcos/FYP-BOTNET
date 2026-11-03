# Normal-Only Autoencoder: Binary Botnet Anomaly Detection

## Training and validation protocol

- **Training rows used:** 334 Normal records (all Normal rows in the training split)
- **Attacks used to fit the autoencoder:** No
- **Validation rows:** 550,278 (full validation split)
- **Validation class counts:** `{'1': 550206, '0': 72}`
- **Features:** 22 canonical input features; 31 after encoding
- **Architecture:** input -> 16 -> 8 bottleneck -> 16 -> reconstruction output
- **Regularization:** L2 alpha=0.05 plus denoising input noise std=0.1; best checkpoint by Normal validation reconstruction MSE
- **Best epoch:** 300 of 300 completed
- **Maximum epoch limit reached:** True
- **Validation-selected anomaly threshold:** 0.058639504
- **Training time:** 0.48s
- **Model artifact:** `models\autoencoder_binary_full_data.joblib`
- **Test split used:** No

## Validation metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.999947 |
| Balanced accuracy | 0.805555 |
| Macro F1 | 0.876055 |
| Normal precision | 0.977778 |
| Normal recall | 0.611111 |
| Attack precision | 0.999949 |
| Attack recall | 0.999998 |
| False positive rate | 0.388889 |
| ROC-AUC from reconstruction error | 0.995836 |

Confusion matrix `[[TN, FP], [FN, TP]]`: `[[44, 28], [1, 550205]]`.
Counts: TN=44, FP=28, FN=1, TP=550205.

## Interpretation and limitations

The autoencoder is trained only on Normal traffic. A record is flagged as an attack when its reconstruction error is at least the selected threshold. The primary threshold was selected on validation by macro-F1; an alternative threshold maximizing balanced accuracy is recorded in the JSON. The same validation split is reported, so these scores are for model comparison rather than an unbiased final estimate. The test split was not used.

There are only 334 Normal training records and 72 Normal validation records. That is very little data for learning benign behavior; per-class metrics and the chosen threshold can be unstable. This autoencoder is a binary detector only and does not predict DDoS/DoS/Reconnaissance/Theft categories.
