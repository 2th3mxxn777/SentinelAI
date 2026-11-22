# SentinelAI Experiment 002: Anomaly Detector Comparison and Calibrated Fusion

**Status:** Completed preliminary pilot comparison. Not a final benchmark.

## Objective
Compare Random Forest (RF), Isolation Forest (IF), and Local Outlier Factor (LOF) for a controlled withheld-attack experiment, then test calibrated score fusion.

## Data and protocol
- Dataset: uploaded CICIDS2017 MachineLearningCSV archive.
- Pilot sample: 26,693 rows, stratified by label with a maximum of 2,500 records per label (smaller classes retained in full).
- Numeric features retained: 66.
- Withheld attack family: `PortScan` (2,500 records).
- Random seed: 42.
- Known labels were converted to benign (0) versus represented attack (1); PortScan was excluded from supervised training.
- Known records were split 75% training / 25% validation, stratified by binary label.
- RF: median imputation + 100-tree Random Forest with balanced class weights.
- IF: median imputation, standard scaling, 80-tree Isolation Forest fit on benign training records.
- LOF: novelty detection with 25 neighbors, fit on a random 2,000-row benign training reference subset.
- IF and LOF thresholds were independently set at the 98th percentile of benign validation anomaly scores.
- Fusion: mean of benign-validation z-scores for RF attack probability and LOF anomaly score; threshold set at the 98th percentile of benign validation fused scores.

## Results

| Method | PortScan recall | PortScan alerts | Benign validation FPR | Known attack validation recall |
|---|---:|---:|---:|---:|
| Random Forest | 3.84% | 96 / 2,500 | 1.60% | 99.93% |
| Isolation Forest | 0.00% | 0 / 2,500 | 2.08% | 4.96% |
| Local Outlier Factor | 1.76% | 44 / 2,500 | 2.08% | 13.53% |
| RF + IF (OR rule) | 3.84% | 96 / 2,500 | 3.68% | 99.93% |
| RF + LOF (OR rule) | 5.48% | 137 / 2,500 | 3.68% | 99.93% |
| Calibrated RF + LOF fusion | 3.04% | 76 / 2,500 | 2.08% | Not summarized as a binary fixed threshold |

## Interpretation
- LOF alone detected some withheld PortScan flows, whereas IF detected none in this pilot.
- The uncalibrated RF + LOF OR rule increased PortScan recall from 3.84% to 5.48%, but raised benign validation FPR from 1.60% to 3.68%.
- Calibrated RF + LOF fusion constrained empirical benign validation FPR to 2.08%, but PortScan recall was 3.04%, below RF alone (3.84%).
- Therefore, the simple fusion tested here did **not** demonstrate a better recall/false-positive trade-off than RF alone. Adding an anomaly score does not automatically improve unknown-class detection.

## Conclusion
Experiment 002 is a negative/diagnostic result. LOF provides a distinct anomaly signal, but naive OR fusion trades additional PortScan detections for more false positives. The calibrated fusion tested here did not outperform RF at a similar benign validation false-positive rate.

## Limitations
1. This is a per-label capped pilot sample, not the full dataset or natural traffic distribution.
2. Only PortScan was withheld in this experiment.
3. Thresholds are calibrated on the benign validation slice from the same dataset.
4. A withheld dataset label is an unknown-class proxy, not proof of real-world zero-day vulnerability detection.
5. No independent cross-dataset validation or analyst study has been completed.
