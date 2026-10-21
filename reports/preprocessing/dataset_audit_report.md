# UNSW BoT-IoT 5% Dataset Comprehensive Audit Report

**Date of Audit:** October 20, 2026  
**Auditor:** ML Engineering Pair Assistant  
**Dataset Name:** UNSW BoT-IoT (5% Subset, All 46 Features)  
**Task Objective:** Rigorous data audit prior to data cleaning, feature selection, balancing, and model development. All original raw files have been maintained strictly unmodified.

---

## 1. Executive Summary

A comprehensive, low-memory chunked audit was conducted on the UNSW BoT-IoT 5% All Features dataset located in the project root. The dataset is partitioned across four CSV files (`UNSW_2018_IoT_Botnet_Full5pc_1.csv` through `UNSW_2018_IoT_Botnet_Full5pc_4.csv`) totaling **969.44 MB** and **3,668,522 connection flow records**.

### Key High-Level Findings:
1. **File Integrity & Schema Uniformity:** All four CSV files are present, physically intact, and share the exact same 46 columns in identical sequential order. No columns or records are missing.
2. **Missing Values & Duplicate Integrity:**
   - **Missing (Null / NaN) Values:** Exactly **0** missing values exist across all 3,668,522 rows and 46 features.
   - **Row Uniqueness:** All 3,668,522 rows possess globally unique `pkSeqID` identifiers (from 1 to 3,668,521 continuous sequence). Furthermore, zero duplicate flow records exist when excluding `pkSeqID`.
3. **Severe Class Imbalance:**
   - **Binary Level (`attack`):** Malicious flows account for **99.987%** (3,668,045 records), whereas Normal traffic accounts for only **0.013%** (**477 records**).
   - **Category Level (`category`):** The dataset is overwhelmingly dominated by DoS (44.98%) and DDoS (52.52%). Minority classes are sparse: Reconnaissance represents 2.48% (91,082 records), Normal represents 0.013% (477 records), and Theft represents just **0.00215%** (**79 records**).
   - **Subcategory Level (`subcategory`):** Subcategories range from UDP (54.01%) and TCP (43.43%) down to Keylogging (73 records) and Data Exfiltration (**6 records**).
4. **File Distribution Bias / Data Segregation:**
   - The dataset was sequentially split across the four files chronologically / by attack wave. Files 1, 2, and 3 contain **0 normal samples** and **0 theft samples**.
   - **All 477 Normal records, all 79 Theft records, and all 91,082 Reconnaissance records reside exclusively in File 4**. Random k-fold cross-validation or sequential train/test splitting will fail catastrophically if not handled with stratified multi-file awareness.
5. **Data Quality & Leakage Alerts:**
   - `sport` and `dport` contain occasional hexadecimal port strings (e.g., `'0x0303'`, `'0x5000'`) corresponding to ICMP type/code encodings rather than pure integers.
   - Exact redundant paired features exist: `proto` and `proto_number`, `flgs` and `flgs_number`, `state` and `state_number`.
   - Strong target leakage risk exists if source/destination IP addresses (`saddr`, `daddr`) or sequential row keys (`pkSeqID`, `seq`, `stime`, `ltime`) are directly fed into classifiers.

---

## 2. File Verification & Physical Dimensions

The four expected CSV files were verified on the local filesystem. File sizes and row counts were verified using low-memory chunked streaming without loading files simultaneously into memory.

| File Name | File Size (Bytes) | File Size (MB) | Row Count | Column Count |
| :--- | :---: | :---: | :---: | :---: |
| `UNSW_2018_IoT_Botnet_Full5pc_1.csv` | 272,335,105 | 259.72 MB | 1,000,000 | 46 |
| `UNSW_2018_IoT_Botnet_Full5pc_2.csv` | 273,082,424 | 260.43 MB | 1,000,000 | 46 |
| `UNSW_2018_IoT_Botnet_Full5pc_3.csv` | 282,131,555 | 269.06 MB | 1,000,000 | 46 |
| `UNSW_2018_IoT_Botnet_Full5pc_4.csv` | 189,577,740 | 180.79 MB | 668,522 | 46 |
| **Total Combined** | **1,017,126,824** | **969.44 MB** | **3,668,522** | **46** |

---

## 3. Dataset Schema & Data Types Inspection

All four CSV files strictly adhere to identical 46-column headers.

### Full Feature Inventory & Types

| Index | Column Name | Inferred Dtype | Description / Role | Quality / Processing Notes |
| :---: | :--- | :---: | :--- | :--- |
| 1 | `pkSeqID` | `int64` | Global Record Identifier | Sequential primary key (1 to 3,668,521). Exclude from ML. |
| 2 | `stime` | `float64` | Record start time (Unix epoch) | Timestamp feature. Risk of temporal leakage. |
| 3 | `flgs` | `object` | Flow flags string | 9 unique flag combinations (`e`, `e s`, `e d`, etc.). |
| 4 | `flgs_number` | `int64` | Encoded flow flags | Exact 1:1 integer encoding of `flgs` (values 1–9). |
| 5 | `proto` | `object` | Network protocol string | 5 protocols (`udp`, `tcp`, `icmp`, `ipv6-icmp`, `arp`). |
| 6 | `proto_number` | `int64` | Encoded protocol | Exact 1:1 integer encoding of `proto` (values 1–5). |
| 7 | `saddr` | `object` | Source IP Address | Categorical identifier. High risk of environment-specific overfitting. |
| 8 | `sport` | `object` / `int64` | Source Port | Contains mixed integer and hex strings (`0x0303`). Needs sanitization. |
| 9 | `daddr` | `object` | Destination IP Address | Categorical identifier. High risk of environment-specific overfitting. |
| 10 | `dport` | `object` / `int64` | Destination Port | Contains mixed integer and hex strings (`0x5000`). Needs sanitization. |
| 11 | `pkts` | `int64` | Total packet count in transaction | Flow volume metric. |
| 12 | `bytes` | `int64` | Total byte count in transaction | Flow volume metric. |
| 13 | `state` | `object` | Flow state string | 11 unique states (`INT`, `RST`, `CON`, `REQ`, etc.). |
| 14 | `state_number` | `int64` | Encoded flow state | Exact 1:1 integer encoding of `state` (values 1–11). |
| 15 | `ltime` | `float64` | Record last time (Unix epoch) | Timestamp feature (`stime` + `dur`). |
| 16 | `seq` | `int64` | Sequence number | Incremental transaction counter. High leakage risk. |
| 17 | `dur` | `float64` | Total flow duration | Temporal duration metric. |
| 18 | `mean` | `float64` | Average packet duration | Statistical duration feature. |
| 19 | `stddev` | `float64` | Standard deviation of packet duration | Statistical duration feature. |
| 20 | `sum` | `float64` | Aggregate duration sum | Statistical duration feature. |
| 21 | `min` | `float64` | Minimum packet duration | Statistical duration feature. |
| 22 | `max` | `float64` | Maximum packet duration | Statistical duration feature. |
| 23 | `spkts` | `int64` | Source to destination packet count | Directional packet counter. |
| 24 | `dpkts` | `int64` | Destination to source packet count | Directional packet counter. |
| 25 | `sbytes` | `int64` | Source to destination byte count | Directional byte counter. |
| 26 | `dbytes` | `int64` | Destination to source byte count | Directional byte counter. |
| 27 | `rate` | `float64` | Flow rate (pkts/sec) | Traffic velocity metric. |
| 28 | `srate` | `float64` | Source flow rate | Traffic velocity metric. |
| 29 | `drate` | `float64` / `int64` | Destination flow rate | Traffic velocity metric (inferred as int64 in files 2 & 4). |
| 30 | `TnBPSrcIP` | `int64` | Total bytes per source IP | Aggregated host-level behavior window feature. |
| 31 | `TnBPDstIP` | `int64` | Total bytes per destination IP | Aggregated host-level behavior window feature. |
| 32 | `TnP_PSrcIP` | `int64` | Total packets per source IP | Aggregated host-level behavior window feature. |
| 33 | `TnP_PDstIP` | `int64` | Total packets per destination IP | Aggregated host-level behavior window feature. |
| 34 | `TnP_PerProto` | `int64` | Total packets per protocol | Aggregated protocol behavior window feature. |
| 35 | `TnP_Per_Dport` | `int64` | Total packets per destination port | Aggregated service behavior window feature. |
| 36 | `AR_P_Proto_P_SrcIP`| `float64` | Average rate per protocol per source IP | Behavioral rate ratio feature. |
| 37 | `AR_P_Proto_P_DstIP`| `float64` | Average rate per protocol per dest IP | Behavioral rate ratio feature. |
| 38 | `N_IN_Conn_P_DstIP` | `int64` | Number of incoming connections per dest IP | Connection density feature. |
| 39 | `N_IN_Conn_P_SrcIP` | `int64` | Number of incoming connections per source IP | Connection density feature. |
| 40 | `AR_P_Proto_P_Sport`| `float64` | Average rate per protocol per sport | Behavioral rate ratio feature. |
| 41 | `AR_P_Proto_P_Dport`| `float64` | Average rate per protocol per dport | Behavioral rate ratio feature. |
| 42 | `Pkts_P_State_P_Protocol_P_DestIP` | `int64` | Packets per state, proto, dest IP | Multi-dimensional aggregated density metric. |
| 43 | `Pkts_P_State_P_Protocol_P_SrcIP` | `int64` | Packets per state, proto, src IP | Multi-dimensional aggregated density metric. |
| 44 | `attack` | `int64` | Binary Ground-Truth Label | **Target Label (Binary):** 0 = Normal, 1 = Attack. |
| 45 | `category` | `object` | Multi-Class Attack Group Label | **Target Label (Multi-class):** 5 categories. |
| 46 | `subcategory` | `object` | Granular Attack Signature Label | **Target Label (Subcategory):** 8 subcategories. |

---

## 4. Missing Value and Duplicate Row Analysis

1. **Missing (Null / NaN) Values:**
   - **Result:** **0 null values across all 46 columns in all 3,668,522 rows.**
   - Every single field has complete data representation. No initial imputation is required.
2. **Duplicate Row Analysis:**
   - **Primary Key Uniqueness:** The `pkSeqID` column is 100% unique across all 3,668,522 rows.
   - **Content Uniqueness (Excluding `pkSeqID`):** An exact hashing check across the remaining 45 feature dimensions revealed **0 duplicate records**. Every flow transaction has at least one unique attribute (timestamps, sequences, or counters).

---

## 5. Attack Label Identification & Frequency Distributions

Three explicit ground-truth label columns are present in the dataset schema:
1. `attack`: Binary classification indicator (0 or 1).
2. `category`: Coarse attack classification (DoS, DDoS, Reconnaissance, Normal, Theft).
3. `subcategory`: Granular attack signature (TCP, UDP, HTTP, Service_Scan, OS_Fingerprint, Keylogging, Data_Exfiltration, Normal).

### 5.1 Binary Detection Distribution (`attack`)
Exported to [`label_frequency_attack.csv`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/label_frequency_attack.csv).

| Class | Meaning | File 1 | File 2 | File 3 | File 4 | Overall Total | Percentage (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | Malicious Attack | 1,000,000 | 1,000,000 | 1,000,000 | 668,045 | **3,668,045** | **99.98700%** |
| `0` | Normal Traffic | 0 | 0 | 0 | 477 | **477** | **0.01300%** |
| **Total** | | 1,000,000 | 1,000,000 | 1,000,000 | 668,522 | **3,668,522** | **100.000%** |

> **Critical Observation:**
> The binary label exhibits an extreme imbalance ratio of **~7,690 to 1** (99.987% Attack vs 0.013% Normal). In addition, **100% of Normal traffic is confined within File 4**.

---

### 5.2 Attack Category Distribution (`category`)
Exported to [`label_frequency_category.csv`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/label_frequency_category.csv).

| Category | File 1 | File 2 | File 3 | File 4 | Overall Total | Percentage (%) | Class Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **DDoS** | 0 | 349,740 | 1,000,000 | 576,884 | **1,926,624** | **52.51772%** | Majority Class |
| **DoS** | 1,000,000 | 650,260 | 0 | 0 | **1,650,260** | **44.98433%** | Majority Class |
| **Reconnaissance**| 0 | 0 | 0 | 91,082 | **91,082** | **2.48280%** | Moderate Minority |
| **Normal** | 0 | 0 | 0 | 477 | **477** | **0.01300%** | Extreme Minority |
| **Theft** | 0 | 0 | 0 | 79 | **79** | **0.00215%** | Ultra Minority |
| **Total** | 1,000,000 | 1,000,000 | 1,000,000 | 668,522 | **3,668,522** | **100.000%** | |

---

### 5.3 Granular Subcategory Distribution (`subcategory`)
Exported to [`label_frequency_subcategory.csv`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/label_frequency_subcategory.csv).

| Subcategory | File 1 | File 2 | File 3 | File 4 | Overall Total | Percentage (%) | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **UDP** | 382,715 | 650,260 | 371,371 | 576,884 | **1,981,230** | **54.00622%** | High-volume flooding |
| **TCP** | 615,800 | 348,751 | 628,629 | 0 | **1,593,180** | **43.42839%** | High-volume flooding |
| **Service_Scan** | 0 | 0 | 0 | 73,168 | **73,168** | **1.99448%** | Port/Service probing |
| **OS_Fingerprint**| 0 | 0 | 0 | 17,914 | **17,914** | **0.48832%** | OS scanning |
| **HTTP** | 1,485 | 989 | 0 | 0 | **2,474** | **0.06744%** | Web DoS/DDoS floods |
| **Normal** | 0 | 0 | 0 | 477 | **477** | **0.01300%** | Benign IoT baseline |
| **Keylogging** | 0 | 0 | 0 | 73 | **73** | **0.00200%** | Information theft |
| **Data_Exfiltration**| 0 | 0 | 0 | 6 | **6** | **0.00016%** | Information theft |
| **Total** | 1,000,000 | 1,000,000 | 1,000,000 | 668,522 | **3,668,522** | **100.000%** | |

---

### 5.4 Cross-Tabulation Matrix (`attack` × `category` × `subcategory`)

Every category and subcategory maps unambiguously:

| `attack` | `category` | `subcategory` | Count | Percentage of Dataset |
| :---: | :--- | :--- | :---: | :---: |
| `1` | DoS | UDP | 1,032,975 | 28.15780% |
| `1` | DDoS | TCP | 977,380 | 26.64234% |
| `1` | DDoS | UDP | 948,255 | 25.84842% |
| `1` | DoS | TCP | 615,800 | 16.78605% |
| `1` | Reconnaissance | Service_Scan | 73,168 | 1.99448% |
| `1` | Reconnaissance | OS_Fingerprint | 17,914 | 0.48832% |
| `1` | DoS | HTTP | 1,485 | 0.04048% |
| `1` | DDoS | HTTP | 989 | 0.02696% |
| `0` | Normal | Normal | 477 | 0.01300% |
| `1` | Theft | Keylogging | 73 | 0.00199% |
| `1` | Theft | Data_Exfiltration | 6 | 0.00016% |

---

## 6. Schema Consistency and Cross-File Discrepancies

1. **Column Order & Names:** Perfectly identical across all 4 files.
2. **Data Type Discrepancies:**
   - **`drate` (Destination Packet Rate):** Inferred as `float64` in File 1 and File 3, but `int64` in File 2 and File 4 (due to all values being integer-valued in those chunks). When loading, explicit type casting to `float64` is required.
   - **`sport` and `dport` (Port Numbers):** While primarily integer values, both fields contain hexadecimal strings (`0x0303`, `0x5000`) representing ICMP packets:
     - File 1: 8 hex occurrences
     - File 2: 6 hex occurrences
     - File 3: 8 hex occurrences
     - File 4: 9,030 hex occurrences
     Pandas defaults to `object` dtype if read without conversion.
3. **Partition Segregation:**
   - File 1 contains purely DoS traffic (100%).
   - File 2 contains a mix of DoS (65%) and DDoS (35%).
   - File 3 contains purely DDoS traffic (100%).
   - File 4 contains DDoS (86.3%), Reconnaissance (13.6%), Normal (0.07%), and Theft (0.01%).
   - **Conclusion:** The files were generated chronologically during simulation runs. Any naive training setup using Files 1–3 for train and File 4 for test would result in **0 Normal and 0 Recon/Theft samples in training**, causing total model failure.

---

## 7. Feature Audit: Redundancies, Identifiers, & Leakage Analysis

### 7.1 Identifiers (Must be excluded from ML models)
- **`pkSeqID`**: Global row number (1 to 3,668,521). This is a purely synthetic primary key.
- **`seq`**: Packet sequence counter. Strongly correlated with attack duration and record creation order.

### 7.2 Target & Simulation Leakage Risks
- **`saddr` & `daddr` (Source & Destination IP Addresses):**
  - Botnet simulation traffic originated from a fixed set of attacker nodes (`192.168.100.147`, `192.168.100.148`, `192.168.100.149`, `192.168.100.150`).
  - Including IP addresses will result in a model that simply memorizes the 4 attacker IP addresses rather than learning genuine traffic behavioral characteristics, leading to 100% artificial accuracy on test data and complete failure on new networks.
- **`stime` & `ltime` (Timestamps):**
  - Absolute Unix timestamps reflect the exact simulation schedule of the testbed in June 2018. Classifiers will easily branch on timestamp ranges to separate attack phases.

### 7.3 Exact Duplicate / Redundant Feature Encodings
The dataset creators provided both raw string representations and integer encoded columns:
- `proto` $\longleftrightarrow$ `proto_number`: Exact 1:1 mapping (`tcp`=1, `arp`=2, `udp`=3, `icmp`=4, `ipv6-icmp`=5).
- `flgs` $\longleftrightarrow$ `flgs_number`: Exact 1:1 mapping (9 distinct flag values).
- `state` $\longleftrightarrow$ `state_number`: Exact 1:1 mapping (11 distinct states).
One of each pair should be retained during preprocessing to avoid redundant multicollinearity.

### 7.4 Constant & Near-Constant Features
- No numeric columns are strictly constant (0 variance) across the entire 3.66M rows.
- However, specific sub-metrics (such as `dpkts` and `dbytes` in unilateral DoS UDP floods) have massive zero concentrations (>95% zeros) because target victims cannot respond during overwhelming flood volume.

---

## 8. Key Issues Requiring Decisions Before Preprocessing

Before proceeding with cleaning, feature selection, or baseline training, the team must align on the following design decisions:

1. **Handling Severe Imbalance in Binary Classification:**
   - Normal traffic consists of only 477 records out of 3.66 million (0.013%).
   - *Option A:* Stratified downsampling of the attack traffic (e.g., preserving all 477 Normal samples and sampling 10,000 to 50,000 Attack records for rapid iteration).
   - *Option B:* Cost-sensitive learning (`class_weight='balanced'`) paired with SMOTE or ADASYN applied **strictly on training folds**.
2. **Multi-Class Attack Scope & The "Theft" Class:**
   - `Theft` has only 79 records (73 Keylogging, 6 Data Exfiltration).
   - In a train/test split, `Data_Exfiltration` has too few samples for standard k-fold cross-validation.
   - The team should decide whether to retain `Theft` as a 4th attack class or merge/evaluate it under an anomaly detection framing.
3. **Multi-File Aggregation and Stratification Protocol:**
   - Files cannot be trained sequentially. Data must be sampled or shuffled via a unified stratified splitting pipeline across all 4 files.
4. **Hexadecimal Port Sanitization:**
   - Decide whether to convert hex port strings (`'0x0303'` $\rightarrow$ `771`, `'0x5000'` $\rightarrow$ `20480`) to valid integers, or treat ICMP flows with a designated sentinel port (e.g., `-1`).

---

## 9. Generated Artifacts and Paths

- **Audit Metrics JSON:** [`audit_metrics.json`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/audit_metrics.json)
- **Binary Label Frequency Table:** [`label_frequency_attack.csv`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/label_frequency_attack.csv)
- **Category Label Frequency Table:** [`label_frequency_category.csv`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/label_frequency_category.csv)
- **Subcategory Label Frequency Table:** [`label_frequency_subcategory.csv`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/label_frequency_subcategory.csv)
- **Audit Script:** [`run_audit.py`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/run_audit.py)
