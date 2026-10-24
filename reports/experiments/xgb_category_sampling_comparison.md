# Stage 5B: XGBoost Attack-Category Multiclass Experiments

Evaluation of XGBoost multiclass classifiers across three training data strategies:
1. **Clean Weighted**: Clean training data with balanced sample weighting.
2. **ROS**: Random Over-Sampling category training set (Normal excluded).
3. **SMOTENC**: SMOTENC synthetic oversampling category training set (Normal excluded).

- **Validation Set**: Held-out clean validation set (550,206 samples, strictly malicious flows)
- **Validation Distribution**: {'DDoS': 289275, 'DoS': 247257, 'Reconnaissance': 13662, 'Theft': 12}

## Overall Validation Performance Comparison

| Metric | Clean Weighted | ROS | SMOTENC |
|---|---|---|---|
| **Overall Accuracy** | 99.9696% | 99.9787% | 99.9766% |
| **Macro Precision** | 0.999782 | 0.980638 | 0.999856 |
| **Macro Recall** | 0.999853 | 0.999895 | 0.999885 |
| **Macro F1-Score** | 0.999817 | 0.989882 | 0.999870 |
| **Weighted F1-Score** | 0.999696 | 0.999787 | 0.999766 |
| **Training Time** | 42.16s | 38.62s | 39.07s |
| **Evaluation Time** | 0.37s | 0.37s | 0.35s |

## Per-Class F1-Score and Recall

| Category | Support | Clean Weighted (F1 / Rec) | ROS (F1 / Rec) | SMOTENC (F1 / Rec) |
|---|---|---|---|---|
| **DDoS** | 289,275 | 0.999715 / 0.999492 | 0.999799 / 0.999679 | 0.999777 / 0.999630 |
| **DoS** | 247,257 | 0.999664 / 0.999919 | 0.999765 / 0.999903 | 0.999741 / 0.999911 |
| **Reconnaissance** | 13,662 | 0.999890 / 1.000000 | 0.999963 / 1.000000 | 0.999963 / 1.000000 |
| **Theft** *(Underrepresented)* | 12 | 1.000000 / 1.000000 | 0.960000 / 1.000000 | 1.000000 / 1.000000 |

## Confusion Matrices

### Clean Weighted Confusion Matrix (Order: DDoS, DoS, Reconnaissance, Theft)

| True \ Pred | Pred DDoS | Pred DoS | Pred Reconnaissance | Pred Theft |
|---|---|---|---|---|
| **DDoS** | 289,128 | 146 | 1 | 0 |
| **DoS** | 18 | 247,237 | 2 | 0 |
| **Reconnaissance** | 0 | 0 | 13,662 | 0 |
| **Theft** | 0 | 0 | 0 | 12 |

### ROS Confusion Matrix (Order: DDoS, DoS, Reconnaissance, Theft)

| True \ Pred | Pred DDoS | Pred DoS | Pred Reconnaissance | Pred Theft |
|---|---|---|---|---|
| **DDoS** | 289,182 | 92 | 1 | 0 |
| **DoS** | 23 | 247,233 | 0 | 1 |
| **Reconnaissance** | 0 | 0 | 13,662 | 0 |
| **Theft** | 0 | 0 | 0 | 12 |

### SMOTENC Confusion Matrix (Order: DDoS, DoS, Reconnaissance, Theft)

| True \ Pred | Pred DDoS | Pred DoS | Pred Reconnaissance | Pred Theft |
|---|---|---|---|---|
| **DDoS** | 289,168 | 106 | 1 | 0 |
| **DoS** | 22 | 247,235 | 0 | 0 |
| **Reconnaissance** | 0 | 0 | 13,662 | 0 |
| **Theft** | 0 | 0 | 0 | 12 |

## Top Feature Importances (Clean Weighted Model)

| Feature | Importance |
|---|---|
| `N_IN_Conn_P_DstIP` | 0.181948 |
| `TnP_Per_Dport` | 0.126526 |
| `dur` | 0.123870 |
| `N_IN_Conn_P_SrcIP` | 0.074917 |
| `dpkts` | 0.067394 |
| `stddev` | 0.062887 |
| `state_INT` | 0.062676 |
| `state_REQ` | 0.041565 |
| `TnP_PerProto` | 0.038006 |
| `rate` | 0.036872 |

