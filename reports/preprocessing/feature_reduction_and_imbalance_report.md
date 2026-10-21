# UNSW BoT-IoT Feature Reduction & Class Imbalance Handling Report

**Stage:** Feature Reduction & Train-Only Class Imbalance Mitigation  
**Execution Date:** October 20, 2026  
**Seed:** 42 (Fully Reproducible)  
**Integrity Status:** **Validation and Test splits held out 100% untouched; all original Parquet files unmodified.**

---

## 1. Executive Summary & Core Accomplishments

1. **Documented Feature Reduction (34 $ightarrow$ 22 Input Features):**
   - In accordance with the Stage 3 feature analysis, **12 redundant and collinear features** were excluded to eliminate extreme multicollinearity ($r > 0.85$ up to $r = 0.9983$) and reduce model dimensionality.
   - **22 high-utility, leakage-free features** were retained (19 numerical, 3 categorical) alongside the 3 target labels.
2. **Train-Only Class Imbalance Handling (3 Experimental Regimes):**
   - **Approach A (Cost-Sensitive Class Weighting):** Precomputed balanced class weights for binary, category, and subcategory tasks. Zero synthetic records are generated; the exact original 2,567,965 training samples are preserved.
   - **Approach B (Cautious Random Oversampling - ROS):** Boosted `Normal` (from 334 to 10,000) and `Theft` (from 55 to 5,000) using true historical samples, raising minority representation by ~30x while preserving real feature distributions.
   - **Approach C (Synthetic Minority Over-sampling - SMOTENC):** Generated continuous synthetic records with categorical constraint enforcement (handling `proto`, `flgs`, `state`) for `Normal` (10,000) and `Theft` (5,000).
3. **Strict Enforcement of Low-Sample Constraints:**
   - **`Data_Exfiltration` Exception:** With only 4 samples in training, SMOTE was **explicitly prohibited** for this subcategory. We do not claim synthetic records can model this 4-record behavior.
   - **`Theft` as a Category:** `Theft` is preserved and treated as a distinct 4th attack class in multi-class setups.
4. **Data Isolation & Storage Accounting:**
   - `botnet_val.parquet` and `botnet_test.parquet` remain strictly unmodified.
   - All experimental training variations were saved as independent Parquet files with zero in-place overwriting.

---

## 2. Feature Reduction Accounting & Justification

The input feature set was pruned from 34 down to 22 non-redundant features:

### 2.1 Excluded Features (12 Columns) & Technical Rationale

| Excluded Feature | Primary Collinear Partner | Correlation ($r$) | Technical Rationale for Exclusion |
| :--- | :--- | :---: | :--- |
| `dbytes` | `dpkts` | **0.9983** | Directional byte count is almost perfectly collinear with packet count. Dropping `dbytes` removes redundancy while preserving packet dynamics. |
| `bytes` | `pkts` | **0.9803** | Total transaction byte volume is redundant with total packets. Retaining `pkts` avoids dual-count distortion in tree and linear models. |
| `sbytes` | `spkts` | **0.9612** | Source-sent byte volume is redundant with source packets. |
| `sum` | `dur` | **0.8821** | Aggregate packet duration sum mirrors flow duration (`dur`). |
| `srate` | `rate` | **0.8803** | In unilateral flood attacks, source rate is virtually identical to overall rate. |
| `drate` | `rate` | High sparsity | Destination packet rate is zero in >95% of records because victims cannot respond during DDoS/DoS floods. |
| `AR_P_Proto_P_Dport` | `rate` | **0.9244** | Average rate per protocol per destination port strongly duplicates overall flow rate. |
| `max` | `mean` | **0.9087** | In high-rate attacks with fixed packet intervals, `max` packet duration is redundant with `mean`. Retained `mean`. |
| `Pkts_P_State_P_DestIP` | `Pkts_P_State_P_SrcIP` | **0.9018** | Symmetric cross-window state density counter. Retaining `Pkts_P_State_P_Protocol_P_SrcIP` provides host-level attack fanout. |
| `AR_P_Proto_P_Sport` | `AR_P_Proto_P_SrcIP` | High collinearity | Per-port protocol rate duplicates host-level rate ratios. |
| `AR_P_Proto_P_SrcIP` | `N_IN_Conn_P_SrcIP` | High collinearity | Redundant with inbound host connection density. |
| `AR_P_Proto_P_DstIP` | `N_IN_Conn_P_DstIP` | High collinearity | Redundant with inbound destination connection density. |

---

### 2.2 Retained Optimized Feature Inventory (22 Input Features)

| No. | Feature Name | Dtype | Category | Role in Botnet Detection |
| :---: | :--- | :---: | :--- | :--- |
| 1 | `proto` | `object` | Categorical | Protocol (`tcp`, `udp`, `icmp`, `arp`, `ipv6-icmp`) |
| 2 | `flgs` | `object` | Categorical | Flow flags string |
| 3 | `state` | `object` | Categorical | Connection state (`INT`, `CON`, `REQ`, `RST`, etc.) |
| 4 | `sport` | `int64` | Discrete Numerical | Sanitized source port number |
| 5 | `dport` | `int64` | Discrete Numerical | Sanitized destination port number |
| 6 | `pkts` | `int64` | Continuous Numerical | Total packet volume |
| 7 | `spkts` | `int64` | Continuous Numerical | Source-to-destination packet volume |
| 8 | `dpkts` | `int64` | Continuous Numerical | Destination-to-source packet volume |
| 9 | `dur` | `float64` | Continuous Numerical | Flow duration (top category MI discriminator: 0.722) |
| 10 | `mean` | `float64` | Continuous Numerical | Mean packet duration |
| 11 | `stddev` | `float64` | Continuous Numerical | Packet duration variance |
| 12 | `min` | `float64` | Continuous Numerical | Minimum packet duration |
| 13 | `rate` | `float64` | Continuous Numerical | Total transmission velocity (packets/sec) |
| 14 | `TnBPSrcIP` | `int64` | Aggregated Behavioral | Total bytes sent by source IP window |
| 15 | `TnBPDstIP` | `int64` | Aggregated Behavioral | Total bytes received by destination IP window |
| 16 | `TnP_PSrcIP` | `int64` | Aggregated Behavioral | Total packets sent by source IP window |
| 17 | `TnP_PDstIP` | `int64` | Aggregated Behavioral | Total packets received by destination IP window |
| 18 | `TnP_PerProto` | `int64` | Aggregated Behavioral | Total packets per protocol window |
| 19 | `TnP_Per_Dport`| `int64` | Aggregated Behavioral | Total packets per destination port window |
| 20 | `N_IN_Conn_P_DstIP` | `int64` | Connection Density | Inbound connections per destination IP (top binary MI) |
| 21 | `N_IN_Conn_P_SrcIP` | `int64` | Connection Density | Inbound connections per source IP |
| 22 | `Pkts_P_State_P_SrcIP` | `int64` | Connection Density | Packets per state, protocol, and source IP |

---

## 3. Class Imbalance Handling Comparison

We deliberately avoided blindly oversampling all classes to equal sizes (e.g. 1.3 million samples each), which would explode training times, distort variance, and lead to massive memorization of minority noise. Instead, we established principled targets:
- **Normal Target:** 10,000 samples (sufficient statistical density for loss convergence, ~30x original sample size).
- **Theft Target:** 5,000 samples (sufficient statistical density for category discrimination, ~90x original sample size).
- **DoS, DDoS, Reconnaissance:** Preserved at true natural distribution.

### 3.1 Class Counts Across Experimental Training Sets

#### Category Level (`category`)

| Category | Original Train | Approach A (Class Weights) | Approach B (ROS Oversampling) | Approach C (SMOTENC Synthetic) |
| :--- | :---: | :---: | :---: | :---: |
| **DDoS** | 1,348,005 | 1,348,005 (w = 0.3810) | 1,348,005 | 1,348,005 |
| **DoS** | 1,155,814 | 1,155,814 (w = 0.4444) | 1,155,814 | 1,155,814 |
| **Reconnaissance**| 63,757 | 63,757 (w = 8.0555) | 63,757 | 63,757 |
| **Normal** | 334 | 334 (w = 1,537.70) | **10,000** | **10,000** |
| **Theft** | 55 | 55 (w = 9,338.05) | **5,000** | **5,000** |
| **Total Rows** | **2,567,965** | **2,567,965** | **2,582,576** | **2,582,576** |

---

#### Binary Attack Level (`attack`)

| Class | Original Train | Approach A (Class Weights) | Approach B (ROS Oversampling) | Approach C (SMOTENC Synthetic) |
| :---: | :---: | :---: | :---: | :---: |
| `1` (Attack) | 2,567,631 | 2,567,631 (w = 0.5001) | 2,572,576 | 2,572,576 |
| `0` (Normal) | 334 | 334 (w = 3,844.26) | **10,000** | **10,000** |
| **Total Rows** | **2,567,965** | **2,567,965** | **2,582,576** | **2,582,576** |

---

#### Subcategory Level (`subcategory`) & Data Exfiltration Integrity

| Subcategory | Original Train | Approach A (Class Weights) | Approach B (ROS Oversampling) | Approach C (SMOTENC Synthetic) | Reliability / Modeling Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **UDP** | 1,386,861 | 1,386,861 (w = 0.2315) | 1,386,861 | 1,386,861 | High volume baseline |
| **TCP** | 1,115,226 | 1,115,226 (w = 0.2878) | 1,115,226 | 1,115,226 | High volume baseline |
| **Service_Scan** | 51,217 | 51,217 (w = 6.2674) | 51,217 | 51,217 | High reliability |
| **OS_Fingerprint**| 12,540 | 12,540 (w = 25.598) | 12,540 | 12,540 | High reliability |
| **Normal** | 334 | 334 (w = 961.06) | 10,000 | 10,000 | Balanced baseline |
| **HTTP** | 1,732 | 1,732 (w = 185.33) | 1,732 | 1,732 | Moderate sample size |
| **Keylogging** | 51 | 51 (w = 6,294.03) | 4,615 | 4,996 | Oversampled within Theft |
| **Data_Exfiltration**| **4** | **4** (w = 80,248.9) | **385** | **4** | **SMOTE Prohibited; 4 Genuine Samples** |

> ### Critical Integrity Documentation: Data_Exfiltration
> - In `botnet_train_smote.parquet`, SMOTE was strictly constrained to synthesize records between `Normal` flows and between `Theft` flows.
> - Because `Data_Exfiltration` contains only 4 records in the entire training set, applying SMOTE to synthesize new exfiltration records would interpolate along arbitrary collinear lines between 4 points, hallucinating unrealistic network flows.
> - Thus, in the SMOTENC dataset, all 4 original `Data_Exfiltration` samples are preserved as genuine, and synthetic Theft records are categorized under `Keylogging`.
> - **Final Recommendation for Modeling:** Modelers must evaluate `Data_Exfiltration` as part of `Theft` at the category level, or explicitly report that subcategory exfiltration has too few original test samples ($N=1$) for statistically valid performance claims.

---

## 4. Strengths and Limitations of Each Approach

### Approach A: Cost-Sensitive Class Weighting (`class_weights.json`)
- **Strengths:** Zero data distortion; 0% chance of overfitting to duplicate records; zero computational overhead during data loading; retains 100% genuine testbed physics.
- **Limitations:** Extreme weights (e.g. weight of 3,844 on Normal) can destabilize gradient descent or cause high variance in tree splits if a few outlier normal records possess high loss.

### Approach B: Cautious Random Oversampling (`botnet_train_ros.parquet`)
- **Strengths:** Simple, highly stable; guarantees minority classes are seen frequently in mini-batches without synthesizing imaginary flow combinations.
- **Limitations:** Duplicate minority records can cause tree models to create tight, overfitted decision boundaries around repeated samples.

### Approach C: Categorical SMOTE (`botnet_train_smote.parquet`)
- **Strengths:** Smooths the decision boundary by generating convex combinations of minority continuous features; forces classifiers to generalize across the minority manifold.
- **Limitations:** Synthetic flows do not represent actual packet captures; risks interpolating across sparse regions in network space that cannot physically occur in real IoT protocols.

---

## 5. Dataset Artifact Summary

| Dataset File | Role | Rows | Features | Storage Size |
| :--- | :--- | :---: | :---: | :---: |
| [`botnet_train_reduced.parquet`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/botnet_train_reduced.parquet) | Base Training Set (Reduced Features) | 2,567,965 | 22 Features + 3 Labels | 134.12 MB |
| [`botnet_train_ros.parquet`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/botnet_train_ros.parquet) | Oversampled Training Set (ROS) | 2,582,576 | 22 Features + 3 Labels | 134.90 MB |
| [`botnet_train_smote.parquet`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/botnet_train_smote.parquet) | Synthetic Training Set (SMOTENC) | 2,582,576 | 22 Features + 3 Labels | 134.95 MB |
| [`botnet_val.parquet`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/botnet_val.parquet) | Validation Set (Untouched Original) | 550,278 | 34 Features + 3 Labels | 41.51 MB |
| [`botnet_test.parquet`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/botnet_test.parquet) | Test Set (Untouched Original) | 550,279 | 34 Features + 3 Labels | 41.47 MB |
| [`class_weights.json`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/class_weights.json) | Precomputed Loss Weights | N/A | Balanced Weights | 1.2 KB |
| [`run_feature_reduction_and_imbalance.py`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/run_feature_reduction_and_imbalance.py) | Pipeline Script | N/A | Reproducible Code | ~15 KB |
