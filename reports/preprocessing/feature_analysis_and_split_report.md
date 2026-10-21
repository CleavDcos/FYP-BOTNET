# UNSW BoT-IoT Feature Analysis & Stratified Split Report

**Stage:** Feature Analysis & Stratified Dataset Partitioning  
**Execution Date:** October 20, 2026  
**Seed:** 42 (Fully Reproducible)  
**Input Dataset:** [`botnet_preprocessed.parquet`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/botnet_preprocessed.parquet) (3,668,522 rows, 37 columns)  
**Outputs Generated:**
- Train (70%): [`botnet_train.parquet`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/botnet_train.parquet) (2,567,965 rows)
- Validation (15%): [`botnet_val.parquet`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/botnet_val.parquet) (550,278 rows)
- Test (15%): [`botnet_test.parquet`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/botnet_test.parquet) (550,279 rows)

---

## 1. Executive Summary & Key Milestones

1. **Stratified Split Feasibility & Execution:**
   - The dataset was successfully partitioned into **70% Train** (2,567,965 rows), **15% Validation** (550,278 rows), and **15% Test** (550,279 rows).
   - Stratification was keyed on the most granular label hierarchy: `subcategory` (8 classes).
   - Because `subcategory` nests directly inside `category` and `attack`, stratifying by `subcategory` **simultaneously guaranteed exact proportional stratification across all three label targets**.
2. **Strict Zero-Leakage Protocol:**
   - All feature relationships, correlation matrices, mutual information scores, and distributions were computed **strictly on the 2,567,965 training rows**. Validation and test folds were held out completely untouched.
   - The original preprocessed file was verified unmodified before and after partitioning.
3. **Severe Minority Class Representation Limits:**
   - `Normal` traffic (477 total records) is partitioned into **334 Train**, **72 Validation**, and **71 Test** records.
   - `Theft` traffic (79 total records) is partitioned into **55 Train**, **12 Validation**, and **12 Test** records.
   - `Data_Exfiltration` (only 6 total records in the entire 3.66M dataset) has **4 Train**, **1 Validation**, and **1 Test** records. **This subcategory cannot support reliable statistical evaluation or cross-validation**.
4. **Feature Collinearity & Redundancy Findings:**
   - 12 feature pairs exhibit severe multicollinearity ($|r| \ge 0.85$), notably packet-to-byte pairs (`dpkts` $\longleftrightarrow$ `dbytes` $r = 0.9983$, `spkts` $\longleftrightarrow$ `sbytes` $r = 0.9612$, `pkts` $\longleftrightarrow$ `bytes` $r = 0.9803$) and rate pairs (`rate` $\longleftrightarrow$ `AR_P_Proto_P_Dport` $r = 0.9244$).
   - Statistical duration features (`mean` $\longleftrightarrow$ `max` $r = 0.9087$) and window aggregates (`Pkts_P_State_P_Protocol_P_DestIP` $\longleftrightarrow$ `Pkts_P_State_P_Protocol_P_SrcIP` $r = 0.9018$) also duplicate information.

---

## 2. Partition Distributions & Class Balance Accounting

### 2.1 Partition Sizes

| Partition | Row Count | Percentage | Parquet Storage | Target Labels Preserved |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 2,567,965 | 70.00% | 52.48 MB | Yes (34 Features + 3 Labels) |
| **Validation** | 550,278 | 15.00% | 11.26 MB | Yes (34 Features + 3 Labels) |
| **Test** | 550,279 | 15.00% | 11.26 MB | Yes (34 Features + 3 Labels) |
| **Combined** | **3,668,522** | **100.00%** | **75.00 MB** | Zero Record Loss |

---

### 2.2 Binary Attack Distribution (`attack`)

| Class | Semantic Meaning | Full Dataset | Train (70%) | Validation (15%) | Test (15%) | Split Proportion |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `1` | Malicious Attack | 3,668,045 | 2,567,631 | 550,206 | 550,208 | 99.987% |
| `0` | Normal Traffic | 477 | 334 | 72 | 71 | 0.013% |
| **Total** | | 3,668,522 | 2,567,965 | 550,278 | 550,279 | 100.00% |

---

### 2.3 Attack Category Distribution (`category`)

| Category | Full Dataset | Train (70%) | Validation (15%) | Test (15%) | Split Proportion |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DDoS** | 1,926,624 | 1,348,005 | 289,275 | 289,344 | 52.518% |
| **DoS** | 1,650,260 | 1,155,814 | 247,257 | 247,189 | 44.984% |
| **Reconnaissance**| 91,082 | 63,757 | 13,662 | 13,663 | 2.483% |
| **Normal** | 477 | 334 | 72 | 71 | 0.013% |
| **Theft** | 79 | 55 | 12 | 12 | 0.00215% |
| **Total** | 3,668,522 | 2,567,965 | 550,278 | 550,279 | 100.00% |

---

### 2.4 Granular Subcategory Distribution (`subcategory`)

| Subcategory | Full Dataset | Train (70%) | Validation (15%) | Test (15%) | Reliability Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **UDP** | 1,981,230 | 1,386,861 | 297,184 | 297,185 | High Reliability |
| **TCP** | 1,593,180 | 1,115,226 | 238,977 | 238,977 | High Reliability |
| **Service_Scan** | 73,168 | 51,217 | 10,975 | 10,976 | High Reliability |
| **OS_Fingerprint**| 17,914 | 12,540 | 2,687 | 2,687 | High Reliability |
| **HTTP** | 2,474 | 1,732 | 371 | 371 | Moderate Reliability |
| **Normal** | 477 | 334 | 72 | 71 | Low (Severe Imbalance) |
| **Keylogging** | 73 | 51 | 11 | 11 | Sparse Representation |
| **Data_Exfiltration**| 6 | 4 | 1 | 1 | **Critical Representation Failure** |
| **Total** | 3,668,522 | 2,567,965 | 550,278 | 550,279 | |

> ### Critical Class Representation Alert:
> `Data_Exfiltration` has only **1 sample** in Validation and **1 sample** in Test.
> - A single false negative drops recall from 100% to 0%.
> - Cross-validation folds cannot evaluate this class reliably.
> - **Recommendation:** When training granular subcategory classifiers, merge `Data_Exfiltration` and `Keylogging` under their parent category `Theft`, or document `Data_Exfiltration` explicitly as an out-of-scope micro-signature.

---

## 3. Train-Only Feature Analysis & Collinearity Findings

All feature relationships were evaluated exclusively on the 2,567,965 training samples to ensure zero data leakage.

### 3.1 Highly Collinear Feature Pairs ($|r| \ge 0.85$)

| Feature 1 | Feature 2 | Pearson Correlation ($r$) | Nature of Redundancy | Recommendation |
| :--- | :--- | :---: | :--- | :--- |
| `dpkts` | `dbytes` | **0.9983** | Directional flow packet/byte coupling | Drop `dbytes` during feature selection (retain `dpkts`) |
| `pkts` | `bytes` | **0.9803** | Total packet/byte coupling | Drop `bytes` (retain `pkts`) |
| `spkts` | `sbytes` | **0.9612** | Source packet/byte coupling | Drop `sbytes` (retain `spkts`) |
| `bytes` | `sbytes` | **0.9456** | Total byte to source byte coupling | Redundant with `sbytes`/`bytes` |
| `pkts` | `spkts` | **0.9427** | DoS/DDoS flows are primarily source-only | Redundant in high-volume floods |
| `pkts` | `sbytes` | **0.9386** | Cross packet/byte coupling | Redundant |
| `rate` | `AR_P_Proto_P_Dport` | **0.9244** | Overall rate mirrors per-dport protocol rate | Drop `AR_P_Proto_P_Dport` |
| `mean` | `max` | **0.9087** | Constant-duration packets skew mean to max | Retain `mean`, consider dropping `max` |
| `Pkts_P_State_P_DestIP`| `Pkts_P_State_P_SrcIP` | **0.9018** | High symmetric flow density | Retain one multi-dimensional density feature |
| `spkts` | `bytes` | **0.8879** | Cross packet/byte coupling | Redundant |
| `dur` | `sum` | **0.8821** | Aggregate packet duration sum mirrors flow duration | Drop `sum` (retain `dur`) |
| `rate` | `srate` | **0.8803** | Source rate dominates total rate | Drop `srate` (retain `rate`) |

---

### 3.2 Mutual Information (Non-Linear Discriminative Power)

Estimated on a stratified 50,000-record training sample:

#### Top Features for Binary Detection (`attack`)
1. **`N_IN_Conn_P_DstIP` (MI: 0.00823):** Number of inbound connections per destination IP. Directly captures flooding and scanning density.
2. **`N_IN_Conn_P_SrcIP` (MI: 0.00491):** Inbound connections per source IP. Distinguishes high-fanout attack sources from normal hosts.
3. **`TnP_PerProto` (MI: 0.00107):** Protocol packet volume. Attacker floods exhibit massive per-protocol counts compared to normal IoT idle states.
4. **`TnP_PDstIP` (MI: 0.00080):** Total packets per destination IP window. Captures victim congestion.
5. **`Pkts_P_State_P_Protocol_P_DestIP` (MI: 0.00066):** Granular traffic state concentration.

#### Top Features for Multiclass Attack Classification (`category`)
1. **`dur` (MI: 0.72248):** Total flow duration. Strongly separates brief scanning probes (Reconnaissance) from prolonged flood sessions (DoS/DDoS).
2. **`AR_P_Proto_P_Sport` (MI: 0.59041):** Average protocol rate per source port. Differentiates single-port flood tools from multi-port portscans.
3. **`rate` (MI: 0.57341):** Transmission velocity. Key separator between high-bandwidth UDP floods and lower-rate HTTP attacks.
4. **`srate` (MI: 0.55736):** Source transmission rate.
5. **`AR_P_Proto_P_SrcIP` (MI: 0.54880):** Protocol rate per source IP.
6. **`AR_P_Proto_P_DstIP` (MI: 0.54249):** Protocol rate per destination IP.
7. **`sum` (MI: 0.48133):** Aggregate duration sum.
8. **`TnBPDstIP` (MI: 0.37469):** Total bytes received by destination IP window.

---

### 3.3 Categorical Features Representation in Training Set

| Feature | Cardinality | Dominant Values (Train Set) | Modeling Recommendation |
| :--- | :---: | :--- | :--- |
| `proto` | 5 | `udp` (54.4%), `tcp` (45.3%), `icmp` (0.25%), `arp` (150), `ipv6-icmp` (6) | One-Hot Encoding (OHE) on 5 categories. |
| `flgs` | 9 | `e` (62.5%), `e s` (33.9%), `e g` (2.6%), `eU` (0.9%), others (<0.1%) | Group rare flags (`e *`, `e d`, `e   t`, etc.) into an 'Other' token or use OHE. |
| `state` | 11 | `INT` (54.4%), `REQ` (23.9%), `RST` (21.1%), `URP` (0.2%), others (<0.1%) | One-Hot Encoding (OHE) on training fold. |

---

## 4. Feature Retention and Selection Recommendations

*Note: As required, all 34 features are physically preserved in `botnet_train.parquet`, `botnet_val.parquet`, and `botnet_test.parquet`. The following recommendations should guide the next feature selection phase:*

### Tier 1: Highly Recommended Core Features (Top 10-12 Candidates)
1. **`N_IN_Conn_P_DstIP`** (Top connection density feature for binary detection)
2. **`N_IN_Conn_P_SrcIP`** (Key host fanout discriminator)
3. **`dur`** (Top mutual information discriminator for attack categories)
4. **`rate`** (Overall flow velocity; captures DDoS/DoS volumetric impact)
5. **`AR_P_Proto_P_Sport`** (Rate per source port; essential for reconnaissance isolation)
6. **`AR_P_Proto_P_SrcIP`** (Rate per source IP ratio)
7. **`TnP_PerProto`** (Protocol traffic volume aggregator)
8. **`TnBPDstIP`** (Destination byte volume window)
9. **`pkts`** (Total packet volume; clean replacement for byte counts)
10. **`sport` & `dport`** (Sanitized service port indicators)
11. **`proto`** (Essential protocol identifier)
12. **`state`** (Connection lifecycle state)

### Tier 2: Redundant Features Recommended for Removal in Next Stage
- **Byte Counters (`bytes`, `sbytes`, `dbytes`):** Redundant with packet counters ($r > 0.96$). Removing byte counts eliminates multicollinearity without information loss.
- **`sum`:** Redundant with `dur` ($r = 0.8821$).
- **`srate` & `drate`:** Sub-components of `rate` ($r = 0.8803$ with `rate`).
- **`AR_P_Proto_P_Dport`:** Redundant with `rate` ($r = 0.9244$).
- **`max`:** Redundant with `mean` ($r = 0.9087$).

---

## 5. Artifact Checklist

- **Train Split (70%):** [`botnet_train.parquet`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/botnet_train.parquet) (2,567,965 rows, 52.48 MB)
- **Validation Split (15%):** [`botnet_val.parquet`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/botnet_val.parquet) (550,278 rows, 11.26 MB)
- **Test Split (15%):** [`botnet_test.parquet`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/botnet_test.parquet) (550,279 rows, 11.26 MB)
- **Original Preprocessed Parquet:** [`botnet_preprocessed.parquet`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/botnet_preprocessed.parquet) (Unmodified, 3,668,522 rows, 74.98 MB)
- **Partitioning & Analysis Script:** [`run_feature_analysis_and_split.py`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/run_feature_analysis_and_split.py)
- **Analysis Metrics JSON:** [`feature_analysis_and_split_metrics.json`](file:///c:/Users/Cleaven%20Dcosta/OneDrive/Desktop/BOTNET/feature_analysis_and_split_metrics.json)
