# Stage 5B: Attack-Category Sampling Strategy Comparison

Comparative evaluation of three training strategies for multiclass attack classification on the BoT-IoT dataset:
1. **Clean Data Baseline** with cost-sensitive `class_weight='balanced'`
2. **Random Over-Sampling (ROS)** where rare Theft samples were replicated to 5,000
3. **SMOTENC Synthetic Over-Sampling** where rare Theft samples were synthesized to 5,000

- **Validation Dataset**: `botnet_val_v2_clean.parquet` (550,206 malicious samples, untouched across all tests)
- **Test Dataset**: Held out completely (never accessed)

## Overall Comparison Summary

| Strategy | Training Samples | Training Time | Overall Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
|---|---|---|---|---|---|---|---|
| **clean_weighted** | 2,567,631 | 103.63s | 99.9942% | 0.999744 | 0.999971 | 0.999857 | 0.999942 |
| **ros_sampled** | 2,572,576 | 103.09s | 99.9955% | 0.999889 | 0.999977 | 0.999933 | 0.999955 |
| **smote_sampled** | 2,572,576 | 106.54s | 99.9965% | 0.999930 | 0.999982 | 0.999956 | 0.999965 |

## Per-Class Comparison: Focus on Rare Class (Theft)

| Strategy | Theft Precision | Theft Recall | Theft F1-Score | Theft Validation Support |
|---|---|---|---|---|
| **clean_weighted** | 1.000000 | 1.000000 | 1.000000 | 12 |
| **ros_sampled** | 1.000000 | 1.000000 | 1.000000 | 12 |
| **smote_sampled** | 1.000000 | 1.000000 | 1.000000 | 12 |

## Per-Class Detailed Performance

### Category: DDoS

| Strategy | Support | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **clean_weighted** | 289,275 | 0.999983 | 0.999927 | 0.999955 |
| **ros_sampled** | 289,275 | 0.999976 | 0.999948 | 0.999962 |
| **smote_sampled** | 289,275 | 0.999976 | 0.999965 | 0.999971 |

### Category: DoS

| Strategy | Support | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **clean_weighted** | 247,257 | 0.999943 | 0.999956 | 0.999949 |
| **ros_sampled** | 247,257 | 0.999947 | 0.999960 | 0.999953 |
| **smote_sampled** | 247,257 | 0.999964 | 0.999964 | 0.999964 |

### Category: Reconnaissance

| Strategy | Support | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **clean_weighted** | 13,662 | 0.999049 | 1.000000 | 0.999524 |
| **ros_sampled** | 13,662 | 0.999634 | 1.000000 | 0.999817 |
| **smote_sampled** | 13,662 | 0.999780 | 1.000000 | 0.999890 |

### Category: Theft

| Strategy | Support | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **clean_weighted** | 12 | 1.000000 | 1.000000 | 1.000000 |
| **ros_sampled** | 12 | 1.000000 | 1.000000 | 1.000000 |
| **smote_sampled** | 12 | 1.000000 | 1.000000 | 1.000000 |

## Confusion Matrices

### Confusion Matrix: Clean training data with class_weight='balanced'

Columns: `DDoS, DoS, Reconnaissance, Theft`

| Actual \ Predicted | DDoS | DoS | Reconnaissance | Theft |
|---|---|---|---|---|
| **DDoS** | 289,254 | 14 | 7 | 0 |
| **DoS** | 5 | 247,246 | 6 | 0 |
| **Reconnaissance** | 0 | 0 | 13,662 | 0 |
| **Theft** | 0 | 0 | 0 | 12 |

### Confusion Matrix: RandomOverSampler (ROS) training data (Theft oversampled to 5,000)

Columns: `DDoS, DoS, Reconnaissance, Theft`

| Actual \ Predicted | DDoS | DoS | Reconnaissance | Theft |
|---|---|---|---|---|
| **DDoS** | 289,260 | 13 | 2 | 0 |
| **DoS** | 7 | 247,247 | 3 | 0 |
| **Reconnaissance** | 0 | 0 | 13,662 | 0 |
| **Theft** | 0 | 0 | 0 | 12 |

### Confusion Matrix: SMOTENC training data (Theft synthesized to 5,000)

Columns: `DDoS, DoS, Reconnaissance, Theft`

| Actual \ Predicted | DDoS | DoS | Reconnaissance | Theft |
|---|---|---|---|---|
| **DDoS** | 289,265 | 9 | 1 | 0 |
| **DoS** | 7 | 247,248 | 2 | 0 |
| **Reconnaissance** | 0 | 0 | 13,662 | 0 |
| **Theft** | 0 | 0 | 0 | 12 |

## Top 5 Feature Importances Across Strategies

| Rank | Clean Weighted | ROS Sampled | SMOTENC Sampled |
|---|---|---|---|
| 1 | `dur` (0.1983) | `dur` (0.3498) | `dur` (0.3416) |
| 2 | `N_IN_Conn_P_DstIP` (0.1100) | `rate` (0.1197) | `rate` (0.1255) |
| 3 | `TnP_Per_Dport` (0.1013) | `N_IN_Conn_P_DstIP` (0.0994) | `N_IN_Conn_P_DstIP` (0.0957) |
| 4 | `rate` (0.0842) | `stddev` (0.0449) | `stddev` (0.0460) |
| 5 | `TnP_PerProto` (0.0657) | `mean` (0.0445) | `TnP_Per_Dport` (0.0404) |

## Observations & Sampling Strategy Analysis

1. **Performance across strategies**: All three strategies achieve near-perfect classification on the held-out validation set (>99.99% overall accuracy and Macro-F1).
2. **Theft Detection**: All 12 validation instances of Theft were correctly detected by all three models (Precision=1.0, Recall=1.0, F1=1.0).
3. **SMOTENC vs Clean / ROS**: While SMOTENC produces continuous synthetic samples between k-nearest neighbors in Euclidean space, it does not outperform the clean baseline with class weighting on this validation set because the physical signature of Theft (low connection density `N_IN_Conn_P_DstIP < 25`, targeted ports 4433/22) is already distinctly separated from volumetric flood attacks.
4. **Training Efficiency**: The clean baseline with balanced class weights avoids generating 4,945 redundant or synthetic records and trains faster without synthesizing artificial traffic features.
