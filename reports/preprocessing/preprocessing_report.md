# UNSW BoT-IoT 5% Dataset Preprocessing Report

**Stage:** Data Cleaning, Standardization, and Feature Harmonization  
**Pipeline Execution Date:** October 20, 2026  
**Input Dataset:** UNSW BoT-IoT 5% All Features (4 CSV partitions)  
**Output Dataset:** [`botnet_preprocessed.parquet`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/botnet_preprocessed.parquet)  
**Raw Data Integrity Status:** **100% Preserved (All raw CSVs untouched)**

---

## 1. Overview and Preprocessing Objectives

Following the comprehensive dataset audit, this stage implemented an end-to-end reproducible preprocessing pipeline. The goals of this pipeline were:
1. **Unify the 4 CSV partitions** into a clean, standardized, and high-performance columnar format ([`botnet_preprocessed.parquet`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/botnet_preprocessed.parquet)) without modifying the raw files.
2. **Eliminate target and environmental leakage** by excluding primary keys, sequential counters, testbed timestamps, and static attacker/victim IP addresses.
3. **Resolve duplicate feature pairs** (`proto`/`proto_number`, `flgs`/`flgs_number`, `state`/`state_number`) by adopting an explicit, documented encoding strategy.
4. **Standardize data types** across partitions, explicitly correcting `drate` to `float64` and converting mixed hexadecimal ICMP port entries in `sport` and `dport` to decimal integers.
5. **Isolate ground-truth target labels** (`attack`, `category`, `subcategory`) so they are strictly reserved for evaluation and supervised targets.
6. **Execute rigorous validation assertions** to guarantee zero data loss, zero null introduction, and exact label preservation across all 3,668,522 records.

---

## 2. Feature Selection & Exclusion Accounting

The raw dataset contains **46 columns**. Through this pipeline:
- **6 columns** were excluded due to identifier, temporal, or testbed environment leakage.
- **3 columns** were excluded due to redundant duplicate encodings.
- **3 columns** were reserved strictly as target labels (`attack`, `category`, `subcategory`).
- **34 features** were retained as clean, leakage-free input features for machine learning.

### 2.1 Excluded Features (9 Columns Total)

| Excluded Column | Category | Rationale for Exclusion |
| :--- | :--- | :--- |
| `pkSeqID` | Identifier | Synthetic global primary key (1 to 3,668,521). Possesses no generalization power and would act as an artificial memorization shortcut. |
| `seq` | Sequence Counter | Transaction sequence number. Highly correlated with attack duration and simulation ordering. |
| `stime` | Temporal Leakage | Unix epoch start time (`1528084952` to `1529380791`). Reflects the exact calendar schedule of the UNSW testbed in June 2018. |
| `ltime` | Temporal Leakage | Unix epoch end time. Functionally identical to `stime` + `dur`, leaking simulation scheduling. |
| `saddr` | Environment Leakage | Source IP address. Over 99% of attack flows originate from just 4 fixed attacker hosts (`192.168.100.147-150`). Models trained on IP will memorize attacker addresses rather than learning attack behavior. |
| `daddr` | Environment Leakage | Destination IP address. Target victim IP addresses (`192.168.100.3-7`) represent the specific lab topology and will fail in external IoT networks. |
| `proto_number` | Redundant Duplicate | Exact 1:1 numerical duplicate of `proto`. Retaining `proto` preserves clear protocol semantics. |
| `flgs_number` | Redundant Duplicate | Exact 1:1 numerical duplicate of `flgs`. Retaining `flgs` preserves discrete network flag symbols. |
| `state_number` | Redundant Duplicate | Exact 1:1 numerical duplicate of `state`. Retaining `state` preserves network transaction states. |

### 2.2 Documentation of Duplicate Encoding Decision
The raw dataset contains redundant pairs:
- `proto` (string: `'tcp'`, `'udp'`, `'icmp'`, `'arp'`, `'ipv6-icmp'`) vs. `proto_number` (int: 1, 2, 3, 4, 5).
- `flgs` (string: `'e'`, `'e s'`, `'e d'`, etc.) vs. `flgs_number` (int: 1-9).
- `state` (string: `'INT'`, `'CON'`, `'REQ'`, `'RST'`, etc.) vs. `state_number` (int: 1-11).

**Design Decision:** We retained the semantic string columns (`proto`, `flgs`, `state`) and excluded the raw numeric integers (`proto_number`, `flgs_number`, `state_number`).
**Reasoning:** The integer representations provided in the dataset are arbitrary ordinal encodings (e.g., `'tcp'`=1, `'arp'`=2, `'udp'`=3). Linear and distance-based classifiers (e.g., Logistic Regression, SVM) would erroneously interpret these as mathematical magnitudes (implying UDP is "three times" TCP). By retaining categorical strings, the downstream modeling pipeline can apply proper One-Hot Encoding (OHE) or target-independent frequency encoding during the model training phase.

---

## 3. Data Type Standardization & Sanitization

### 3.1 Resolving Hexadecimal Port Values (`sport`, `dport`)
- **Finding:** While `sport` and `dport` are primarily decimal integer port numbers, ICMP records in the BoT-IoT dataset encode ICMP message type and code as hexadecimal strings (e.g., `'0x0303'` for Destination Unreachable, `'0x5000'`). In File 4 alone, 9,030 hexadecimal port entries were identified.
- **Implementation:** The pipeline parsed all port entries:
  - Standard decimal strings were safely converted to `int64`.
  - Hexadecimal strings (`0x...`) were converted to their base-16 numerical equivalent (`'0x0303'` $
ightarrow$ `771`, `'0x5000'` $
ightarrow$ `20480`).
  - Total hexadecimal conversions: **9,052 source ports** and **9,052 destination ports**.
  - Any missing/unparseable values default to `-1`.

### 3.2 Resolving Cross-Partition Dtype Discrepancies (`drate`)
- **Finding:** Files 1 and 3 contained floating-point values in `drate`, while Files 2 and 4 contained purely integer values, causing Pandas to infer inconsistent types.
- **Implementation:** Explicitly forced `drate` to `float64` during chunked ingestion and Parquet writing.

---

## 4. Retained Model Input Features (34 Features)

All 34 retained features are standardized and ready for feature scaling and model training:

| No. | Feature Name | Standardized Dtype | Role / Feature Type | Description |
| :---: | :--- | :---: | :--- | :--- |
| 1 | `flgs` | `object` / `string` | Categorical (Flow Flags) | 9 distinct TCP/flow flag states |
| 2 | `proto` | `object` / `string` | Categorical (Protocol) | Network protocol (`tcp`, `udp`, `icmp`, etc.) |
| 3 | `sport` | `int64` | Discrete Numerical | Sanitized source port number |
| 4 | `dport` | `int64` | Discrete Numerical | Sanitized destination port number |
| 5 | `pkts` | `int64` | Continuous Numerical | Total packet count in connection |
| 6 | `bytes` | `int64` | Continuous Numerical | Total byte volume in connection |
| 7 | `state` | `object` / `string` | Categorical (Flow State) | Transaction state (`INT`, `CON`, `REQ`, etc.) |
| 8 | `dur` | `float64` | Continuous Numerical | Connection duration (seconds) |
| 9 | `mean` | `float64` | Continuous Numerical | Mean packet duration |
| 10 | `stddev` | `float64` | Continuous Numerical | Standard deviation of packet duration |
| 11 | `sum` | `float64` | Continuous Numerical | Aggregate packet duration sum |
| 12 | `min` | `float64` | Continuous Numerical | Minimum packet duration |
| 13 | `max` | `float64` | Continuous Numerical | Maximum packet duration |
| 14 | `spkts` | `int64` | Continuous Numerical | Packets sent from source to destination |
| 15 | `dpkts` | `int64` | Continuous Numerical | Packets sent from destination to source |
| 16 | `sbytes` | `int64` | Continuous Numerical | Bytes sent from source to destination |
| 17 | `dbytes` | `int64` | Continuous Numerical | Bytes sent from destination to source |
| 18 | `rate` | `float64` | Continuous Numerical | Overall transmission rate (packets/sec) |
| 19 | `srate` | `float64` | Continuous Numerical | Source transmission rate |
| 20 | `drate` | `float64` | Continuous Numerical | Destination transmission rate |
| 21 | `TnBPSrcIP` | `int64` | Aggregated Behavioral | Total bytes per source IP window |
| 22 | `TnBPDstIP` | `int64` | Aggregated Behavioral | Total bytes per destination IP window |
| 23 | `TnP_PSrcIP` | `int64` | Aggregated Behavioral | Total packets per source IP window |
| 24 | `TnP_PDstIP` | `int64` | Aggregated Behavioral | Total packets per destination IP window |
| 25 | `TnP_PerProto` | `int64` | Aggregated Behavioral | Total packets per protocol window |
| 26 | `TnP_Per_Dport` | `int64` | Aggregated Behavioral | Total packets per destination port window |
| 27 | `AR_P_Proto_P_SrcIP` | `float64` | Aggregated Rate Ratio | Average rate per protocol per source IP |
| 28 | `AR_P_Proto_P_DstIP` | `float64` | Aggregated Rate Ratio | Average rate per protocol per destination IP |
| 29 | `N_IN_Conn_P_DstIP` | `int64` | Connection Density | Inbound connections per destination IP |
| 30 | `N_IN_Conn_P_SrcIP` | `int64` | Connection Density | Inbound connections per source IP |
| 31 | `AR_P_Proto_P_Sport` | `float64` | Aggregated Rate Ratio | Average rate per protocol per source port |
| 32 | `AR_P_Proto_P_Dport` | `float64` | Aggregated Rate Ratio | Average rate per protocol per destination port |
| 33 | `Pkts_P_State_P_Protocol_P_DestIP` | `int64` | Multi-dimensional Density | Packets grouped by state, protocol, destination IP |
| 34 | `Pkts_P_State_P_Protocol_P_SrcIP` | `int64` | Multi-dimensional Density | Packets grouped by state, protocol, source IP |

---

## 5. Preserved Ground-Truth Target Labels

All three label dimensions are preserved as dedicated target variables:

### 5.1 Binary Target (`attack`)
- **`1` (Attack):** 3,668,045 records (99.987%)
- **`0` (Normal):** 477 records (0.013%)

### 5.2 Multiclass Target (`category`)
- **`DDoS`:** 1,926,624 records (52.518%)
- **`DoS`:** 1,650,260 records (44.984%)
- **`Reconnaissance`:** 91,082 records (2.483%)
- **`Normal`:** 477 records (0.013%)
- **`Theft`:** 79 records (0.00215%)

### 5.3 Granular Subcategory Target (`subcategory`)
- **`UDP`:** 1,981,230
- **`TCP`:** 1,593,180
- **`Service_Scan`:** 73,168
- **`OS_Fingerprint`:** 17,914
- **`HTTP`:** 2,474
- **`Normal`:** 477
- **`Keylogging`:** 73
- **`Data_Exfiltration`:** 6

---

## 6. Pipeline Validation & Integrity Verification

To ensure zero silent data corruption, the pipeline executed strict assertion checks upon completion:

| Validation Check | Expected Result | Actual Result | Status |
| :--- | :---: | :---: | :---: |
| Raw CSV File Modification | 0 Bytes Changed, 0 mtime Changed | Identical byte size and timestamp | **PASSED** |
| Processed Row Count | Exactly 3,668,522 rows | 3,668,522 rows | **PASSED** |
| Processed Column Count | 34 Features + 3 Targets = 37 Columns | 37 Columns | **PASSED** |
| Missing / NaN Values | 0 across all columns | 0 across all columns | **PASSED** |
| Binary Attack Distribution | Exactly 3,668,045 Attack / 477 Normal | Exactly 3,668,045 Attack / 477 Normal | **PASSED** |
| Normal Traffic Count | Exactly 477 rows preserved | Exactly 477 rows preserved | **PASSED** |
| Theft Traffic Count | Exactly 79 rows preserved | Exactly 79 rows preserved | **PASSED** |

---

## 7. Storage Efficiency

- **Raw CSVs Total Size:** 1,017.13 MB
- **Standardized Parquet Size:** **74.98 MB**
- **Storage Footprint Reduction:** ~75% reduction via Snappy columnar compression, enabling sub-second memory-mapped scans for upcoming feature selection and model training stages.
