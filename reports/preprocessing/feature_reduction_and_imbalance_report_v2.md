# UNSW BoT-IoT Feature Reduction & Imbalance Handling Audit Report (v2)

**Pipeline Stage:** Review, Remediation, and Versioned Dataset Standardization  
**Execution Date:** October 20, 2026  
**Seed:** 42 (Strictly Reproducible)  
**Status:** **Remediation Completed; Zero Target Leakage; All Original Splits Unmodified.**

---

## 1. Executive Summary & Remediation Actions

Following a detailed architectural review of the initial feature-reduction and imbalance pipeline, several critical design corrections were executed to guarantee mathematical rigor and prevent data leakage:

1. **Uniform 22-Feature Canonical Schema:**
   - The exact same 22 selected features, in the exact same canonical order, were applied to generate clean versioned validation ([`botnet_val_v2_clean.parquet`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/botnet_val_v2_clean.parquet)) and test ([`botnet_test_v2_clean.parquet`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/botnet_test_v2_clean.parquet)) files.
   - The original 34-feature splits ([`botnet_train.parquet`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/botnet_train.parquet), `_val`, `_test`) remain 100% byte-identical and unmodified to allow empirical benchmarking between the 34-feature baseline and the 22-feature reduced representation.
2. **Absolute Target Leakage Elimination:**
   - In earlier iterations, secondary target labels were inadvertently bundled alongside the feature matrix during oversampling.
   - In this remediated pipeline, `X_train` contains **strictly the 22 input features**. The target columns (`attack`, `category`, `subcategory`) are completely excluded from the feature space during resampling, transformations, and model feeding.
3. **Corrected RandomOverSampler (ROS):**
   - Resampling for the category task is fitted purely on `(X_train[22_features], y_category)`. Secondary labels are never passed.
4. **Corrected SMOTENC (Zero Subcategory Fabrication):**
   - Synthetic records generated for the minority class `Theft` (from 55 up to 5,000) are assigned **strictly the `category = 'Theft'` label**.
   - No synthetic subcategory labels (e.g., `Keylogging` or synthetic `Data_Exfiltration`) are fabricated. Synthetic records do not have granular subcategory claims attached.
5. **Subcategory Isolation & Data Exfiltration Micro-Class Status:**
   - `Data_Exfiltration` contains only 4 samples in Train, 1 in Validation, and 1 in Test.
   - It is explicitly segregated as a subcategory evaluation target only, and formally documented as too rare for reliable statistical metrics.

---

## 2. Canonical 22-Feature Set vs. Excluded 12 Features

### 2.1 The 22 Canonical Features (Identical in Train, Val, and Test)

| No. | Feature Name | Dtype | Category | Role in Network Flow Discrimination |
| :---: | :--- | :---: | :--- | :--- |
| 1 | `proto` | `object` | Categorical | Protocol (`tcp`, `udp`, `icmp`, `arp`, `ipv6-icmp`) |
| 2 | `flgs` | `object` | Categorical | TCP flow flags string |
| 3 | `state` | `object` | Categorical | Flow state (`INT`, `CON`, `REQ`, `RST`, etc.) |
| 4 | `sport` | `int64` | Discrete Numerical | Sanitized source port number |
| 5 | `dport` | `int64` | Discrete Numerical | Sanitized destination port number |
| 6 | `pkts` | `int64` | Continuous Numerical | Total packet volume in connection |
| 7 | `spkts` | `int64` | Continuous Numerical | Packets sent from source to destination |
| 8 | `dpkts` | `int64` | Continuous Numerical | Packets sent from destination to source |
| 9 | `dur` | `float64` | Continuous Numerical | Flow duration (top mutual information for category: 0.722) |
| 10 | `mean` | `float64` | Continuous Numerical | Mean packet duration |
| 11 | `stddev` | `float64` | Continuous Numerical | Packet duration standard deviation |
| 12 | `min` | `float64` | Continuous Numerical | Minimum packet duration |
| 13 | `rate` | `float64` | Continuous Numerical | Total flow velocity (packets/sec) |
| 14 | `TnBPSrcIP` | `int64` | Aggregated Behavioral | Total bytes sent by source IP window |
| 15 | `TnBPDstIP` | `int64` | Aggregated Behavioral | Total bytes received by destination IP window |
| 16 | `TnP_PSrcIP` | `int64` | Aggregated Behavioral | Total packets sent by source IP window |
| 17 | `TnP_PDstIP` | `int64` | Aggregated Behavioral | Total packets received by destination IP window |
| 18 | `TnP_PerProto` | `int64` | Aggregated Behavioral | Total packets per protocol window |
| 19 | `TnP_Per_Dport`| `int64` | Aggregated Behavioral | Total packets per destination port window |
| 20 | `N_IN_Conn_P_DstIP` | `int64` | Connection Density | Inbound connections per destination IP (top binary MI) |
| 21 | `N_IN_Conn_P_SrcIP` | `int64` | Connection Density | Inbound connections per source IP |
| 22 | `Pkts_P_State_P_Protocol_P_SrcIP` | `int64` | Density Metric | Packets per state, protocol, and source IP |

---

### 2.2 Excluded 12 Features & Rationale

| Excluded Feature | Collinear Partner | Correlation ($r$) | Exclusion Rationale |
| :--- | :--- | :---: | :--- |
| `dbytes` | `dpkts` | **0.9983** | Directional byte count duplicates packet count. |
| `bytes` | `pkts` | **0.9803** | Total byte count duplicates packet count. |
| `sbytes` | `spkts` | **0.9612** | Source-sent byte volume duplicates source packets. |
| `sum` | `dur` | **0.8821** | Aggregate packet duration sum mirrors flow duration. |
| `srate` | `rate` | **0.8803** | Source rate duplicates total transmission rate in unilateral floods. |
| `drate` | `rate` | High sparsity | Destination rate is zero in >95% of attack records. |
| `AR_P_Proto_P_Dport` | `rate` | **0.9244** | Destination port protocol rate duplicates overall rate. |
| `max` | `mean` | **0.9087** | Redundant with `mean` duration under fixed inter-arrival times. |
| `Pkts_P_State_P_Protocol_P_DestIP` | `Pkts_P_State_P_Protocol_P_SrcIP` | **0.9018** | Symmetric cross-window duplicate. Retained source-oriented metric. |
| `AR_P_Proto_P_Sport` | `AR_P_Proto_P_SrcIP` | High collinearity | Per-port protocol rate duplicates host rate ratios. |
| `AR_P_Proto_P_SrcIP` | `N_IN_Conn_P_SrcIP` | High collinearity | Duplicates inbound host connection density. |
| `AR_P_Proto_P_DstIP` | `N_IN_Conn_P_DstIP` | High collinearity | Duplicates inbound destination connection density. |

*Important Note:* The original 34-feature datasets remain intact so downstream model evaluation can explicitly compare model performance between 34 features vs. 22 features, verifying whether dimensionality reduction improves inference speed without compromising detection accuracy.

---

## 3. Dataset Inventory and Class Balance Summary

### 3.1 Category-Level Training Sets (`category`)

| Category | Base Train (`botnet_train_v2_clean`) | Loss Weights ($w$) (`class_weights_v2.json`) | ROS Train (`botnet_train_v2_ros_category`) | SMOTE Train (`botnet_train_v2_smote_category`) |
| :--- | :---: | :---: | :---: | :---: |
| **DDoS** | 1,348,005 | $0.3810$ | 1,348,005 | 1,348,005 |
| **DoS** | 1,155,814 | $0.4444$ | 1,155,814 | 1,155,814 |
| **Reconnaissance**| 63,757 | $8.0555$ | 63,757 | 63,757 |
| **Normal** | 334 | $1,537.70$ | **10,000** | **10,000** |
| **Theft** | 55 | $9,338.05$ | **5,000** | **5,000** |
| **Total Rows** | **2,567,965** | **2,567,965** | **2,582,576** | **2,582,576** |
| **Input Columns**| **22 Features + Targets** | **Loss Penalty** | **22 Features + category** | **22 Features + category** |

---

### 3.2 Binary Attack Training Sets (`attack`)

| Class | Base Train (`botnet_train_v2_clean`) | Loss Weights ($w$) (`class_weights_v2.json`) | ROS Train (`botnet_train_v2_ros_binary`) | SMOTE Train (`botnet_train_v2_smote_binary`) |
| :---: | :---: | :---: | :---: | :---: |
| `1` (Attack) | 2,567,631 | $0.5001$ | 2,567,631 | 2,567,631 |
| `0` (Normal) | 334 | $3,844.26$ | **10,000** | **10,000** |
| **Total Rows** | **2,567,965** | **2,567,965** | **2,577,631** | **2,577,631** |
| **Input Columns**| **22 Features + Targets** | **Loss Penalty** | **22 Features + attack** | **22 Features + attack** |

---

### 3.3 Subcategory Distribution & Data Exfiltration Status

| Subcategory | Train Split | Validation Split | Test Split | Reliability Status |
| :--- | :---: | :---: | :---: | :--- |
| **UDP** | 1,386,861 | 297,184 | 297,185 | High Reliability |
| **TCP** | 1,115,226 | 238,977 | 238,977 | High Reliability |
| **Service_Scan** | 51,217 | 10,975 | 10,976 | High Reliability |
| **OS_Fingerprint**| 12,540 | 2,687 | 2,687 | High Reliability |
| **HTTP** | 1,732 | 371 | 371 | Moderate Sample Size |
| **Normal** | 334 | 72 | 71 | Balanced via ROS/SMOTE/Weights |
| **Keylogging** | 51 | 11 | 11 | Rare Signature (Theft Sub-type) |
| **Data_Exfiltration**| **4** | **1** | **1** | **Severe Micro-Class: Statistically Un-evaluable** |

> ### Formal Protocol on `Data_Exfiltration`:
> 1. With only 1 test sample, a false negative causes test recall to register 0%, while a true positive registers 100%. Cross-validation is impossible.
> 2. `Data_Exfiltration` must be evaluated under the parent `Theft` category for benchmark reporting.
> 3. Subcategory models that include `Data_Exfiltration` must explicitly disclose that the metric is based on $N=1$ evaluation samples.

---

## 4. Verification Checkpoint Audit

Strict programmatic assertions verified all datasets upon completion:

| Verification Metric | Expected Benchmark | Pipeline Result | Status |
| :--- | :---: | :---: | :---: |
| Original Split Immutability | Byte size & mtime unchanged | 100% byte identical | **PASSED** |
| Feature Consistency | Exactly 22 canonical features in identical order | Exactly matched across Train/Val/Test | **PASSED** |
| Target Leakage Prevention | 0 target labels in feature matrices | 0 target labels in input features | **PASSED** |
| Missing / NaN Values | 0 nulls across all generated files | 0 nulls detected | **PASSED** |
| Synthetic Integrity | Synthetic Theft labeled strictly as 'Theft' | 0 fabricated subcategory labels | **PASSED** |

---

## 5. Clean Versioned Dataset Artifact Inventory

### Clean Baseline Datasets (22 Canonical Features + 3 Isolated Targets)
- **Train Split (22 Features):** [`botnet_train_v2_clean.parquet`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/botnet_train_v2_clean.parquet) (2,567,965 rows, 112.02 MB)
- **Validation Split (22 Features):** [`botnet_val_v2_clean.parquet`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/botnet_val_v2_clean.parquet) (550,278 rows, 24.01 MB)
- **Test Split (22 Features):** [`botnet_test_v2_clean.parquet`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/botnet_test_v2_clean.parquet) (550,279 rows, 24.01 MB)

### Category Task Resampled Training Sets (22 Features + `category`)
- **Category ROS Dataset:** [`botnet_train_v2_ros_category.parquet`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/botnet_train_v2_ros_category.parquet) (2,582,576 rows, 112.42 MB)
- **Category SMOTENC Dataset:** [`botnet_train_v2_smote_category.parquet`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/botnet_train_v2_smote_category.parquet) (2,582,576 rows, 112.79 MB)

### Binary Attack Task Resampled Training Sets (22 Features + `attack`)
- **Binary ROS Dataset:** [`botnet_train_v2_ros_binary.parquet`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/botnet_train_v2_ros_binary.parquet) (2,577,631 rows, 112.18 MB)
- **Binary SMOTENC Dataset:** [`botnet_train_v2_smote_binary.parquet`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/botnet_train_v2_smote_binary.parquet) (2,577,631 rows, 112.56 MB)

### Loss Weights & Scripts
- **Class Weights JSON:** [`class_weights_v2.json`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/class_weights_v2.json)
- **Remediated Pipeline Script:** [`run_stage4_remediation.py`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/run_stage4_remediation.py)
