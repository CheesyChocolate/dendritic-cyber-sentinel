# Dendritic Cell Algorithm (DCA) for Intrusion Detection

## Based on CIC-IDS-2017 Dataset

---

## Objective

Implement and evaluate the **Dendritic Cell Algorithm (DCA)** for binary
anomaly detection (BENIGN vs ATTACK) on the **CIC-IDS-2017 dataset** using
biologically-inspired signal processing.

---

## Dataset: CIC-IDS-2017

- Realistic 5-day capture of network traffic
- Includes:
  - BENIGN traffic
  - DDoS, DoS, PortScan
  - Web attacks, Infiltration, Botnets
- 80+ flow features from CICFlowMeter

---

## Data Preprocessing

1. **Load & Merge:** All CSV files into one DataFrame
2. **Clean:**
   - Remove null, duplicate, single-value, and infinite rows/columns
3. **Feature Reduction:**
   - Drop columns with >50% zeros or >30% missing values
4. **Scale Features:** StandardScaler used on all numeric columns

---

## Label Encoding

- Cleaned and encoded `Label` column
- Converted to **binary labels** for DCA:
  - `BENIGN` = 0 (normal)
  - All others = 1 (anomalous)

### Label Distribution

- BENIGN: 2,268,719
- ATTACK:   556,376
- **Total:** 2,825,095

---

## DCA Overview

- Unsupervised anomaly detection algorithm
- Inspired by dendritic cells in the immune system
- Detects and responds to pathogens (anomalies)
- Uses **signal processing** to identify malicious traffic
- Simulates a network of dendritic cells (DCs) to analyze traffic
- Each DC collects signals over its lifetime and computes a **k-value** to
determine if the traffic is anomalous
- Matures and labels data based on k-value:
  - `k > 0` → Anomalous
  - `k <= 0` → Normal
- Hyperparameters:
  - `k`-threshold: Determines the sensitivity of the algorithm
  - Lifespan: Duration for which a DC collects signals
  - Signal weights: Importance of each signal type in the k-value calculation
  - Signal types: Different types of signals (e.g., PAMP, Danger, Safe) that a DC can collect


---

## Signal Mapping
Mapped features into 3 signal types based on behavior:

```python
signal_mapping = {
    "Fwd Packet Length Max": "PAMP",
    "Bwd Packet Length Max": "PAMP",
    "Packet Length Variance": "PAMP",

    "Flow IAT Mean": "Danger",
    "Total Fwd Packets": "Danger",
    "Flow Duration": "Danger",

    "Fwd IAT Min": "Safe",
    "Min Packet Length": "Safe",
    "Fwd Packet Length Min": "Safe"
}
```

---

## Dendritic Cell Simulation
- 100 simulated DCs with random lifespan (80-120 samples)
- Each DC collects signals (`PAMP`, `Danger`, `Safe`) over its lifetime
- Computes **k-value**: high = anomaly, low = normal
- Matures and labels data based on k-value:
  - `k > 0` → Anomalous
  - `k <= 0` → Normal

---

## Prediction & Voting
- Samples assigned multiple contexts (from DCs)
- Final label = **majority vote** from DCs

---

## Binary Classification Results
```
              precision    recall  f1-score   support

      BENIGN       0.91      0.88      0.89   2,268,719
      ATTACK       0.57      0.63      0.60     556,376

    accuracy                           0.83   2,825,095
   macro avg       0.74      0.76      0.75
weighted avg       0.84      0.83      0.84
```

---

## Observations
- **Strong BENIGN detection** (91% precision)
- **Moderate attack detection** (63% recall)
- High overall **unsupervised accuracy** (83%)

---

## Planned Visualizations
- Track and visualize **k-values** by class
- Histogram of average `k` values:
  - Compare BENIGN vs ATTACK distribution
  - Fine-tune threshold (`k > 2`, `k > 5`, etc.)

---

## Conclusions
- DCA is effective and unsupervised
- Feature-to-signal mapping is critical
- Binary conversion aligns well with DCA's behavior

---

## Future Work
- **Enhance DCA performance**:
  <!--
   Tune the `k`-threshold
   Weight signals differently (e.g., `2*PAMP`, `-3*Safe`)
   -->
  1. Using other optimization algorithms for hyperparameter tuning(e.g., PSO, GA)
  2. Use ensemble DCA runs for stability
  3. Explore **multiclass DCA**
  4. Benchmark against **supervised ML models**
  5. Using other outlier detection algorithms for comparison
  6. Explore **transfer learning** for DCA on different datasets
  7. Investigate **real-time DCA** for online anomaly detection

---

## Thank You!
Questions?
