# SentinelAI — CICIDS2017 Pilot Experiment 001

**Status:** Completed preliminary pilot. Not a final benchmark.

## Research question
Can a hybrid supervised and anomaly-detection pipeline identify a network attack family withheld from supervised training?

## Dataset and preprocessing
- Source: uploaded `MachineLearningCSV.zip`, containing 8 CSV files.
- Total source rows: 2,830,743.
- A streaming scan counted labels across all 8 files.
- Pilot sample: 159,616 rows, capped at 30,000 per label; rare classes were retained in full.
- Features retained: 68 numeric flow features.
- Withheld class: `PortScan` (30,000 sampled flows).
- Random seed: 42.

The per-label cap makes this first run manageable, but changes the natural class proportions. Known-class validation metrics must not be interpreted as deployment prevalence or operational precision.

## Method
1. Treat `BENIGN` as class 0 and all represented non-withheld attack classes as class 1.
2. Exclude all `PortScan` rows from supervised training.
3. Split known-class records 75%/25%, stratified by binary class.
4. Train a median-imputed Random Forest (150 trees; balanced class weights).
5. Train a median-imputed, standardized Isolation Forest (100 trees) using benign training flows only.
6. Set the anomaly threshold at the 98th percentile of benign validation anomaly scores (2% target validation false-positive quantile).
7. Evaluate on known-class validation records plus all sampled withheld `PortScan` records.

## Results

| Detector | PortScan recall | PortScan alerts |
|---|---:|---:|
| Random Forest only | 0.00443 | 133 / 30,000 |
| Isolation Forest only | 0.00000 | 0 / 30,000 |
| Hybrid OR | 0.00443 | 133 / 30,000 |

The hybrid did not improve withheld-class detection because the Isolation Forest did not flag any sampled PortScan flows at the selected threshold.

Confusion matrix for the combined evaluation (rows = true benign/attack; columns = predicted benign/attack):

```
[[ 7339,   161],
 [29883, 25021]]
```

Known-class validation results were strong in this capped sample (benign recall approximately 0.98; known-attack recall approximately 1.00). Those metrics are affected by the capped sampling design and should not be presented as real-world operational performance.

## Preliminary interpretation
The baseline recognizes represented attack families but fails to generalize to the withheld `PortScan` sample. This negative result suggests that the current feature representation, training design, or anomaly threshold is insufficient for this unknown-class proxy.

## Next experiments
1. Repeat the leave-one-attack-family-out test for several attack categories.
2. Compare Isolation Forest with other anomaly detectors under the same documented false-positive budget.
3. Test threshold stability across individual capture days/files.
4. Repeat with the full dataset and a clearly documented evaluation protocol.
5. Validate on an independent dataset such as UNSW-NB15.
6. Add per-flow, evidence-grounded explanations and separately evaluate their faithfulness before introducing an LLM/RAG layer.

## Limitations and claims boundary
- A withheld CICIDS2017 label is a controlled unknown-class proxy, not proof of detecting arbitrary real-world zero-day vulnerabilities.
- This is a capped pilot sample, not a full-data benchmark.
- No independent cross-dataset validation has been completed.
- No LLM/RAG explanation quality evaluation has been completed.
- Global feature importance is not causal and is not a per-flow explanation.

**Conclusion:** Experiment 001 is complete as a baseline pilot. Its main finding is a failure case: the hybrid baseline did not detect the withheld `PortScan` sample at a useful rate. This is a concrete starting point for improving SentinelAI, not a successful zero-day-detection claim.
