# Methodology

## Dataset
Primary candidate: CICIDS2017, obtained from the Canadian Institute for Cybersecurity's official dataset page: https://www.unb.ca/cic/datasets/ids-2017.html. Record the exact files, download date, checksum, label mapping, and feature schema. Do not assume that every CSV has identical columns.

## Experimental unit and target
Each row represents a network-flow record. The target is the dataset's attack/benign label column. Inspect exact label values before selecting the withheld attack family.

## Leakage-resistant protocol
1. Read and concatenate CSV files while preserving source-file provenance.
2. Normalize column names and remove whitespace from labels.
3. Convert numeric features; replace infinities and missing values using training-derived statistics only.
4. Remove identifier-like columns (for example flow IDs, source/destination IPs, timestamps) unless a justified experiment explicitly studies them.
5. Split the known-label subset into training and validation partitions with stratification where feasible.
6. Fit the supervised classifier only on known-label training data.
7. Fit the anomaly detector only on benign training examples.
8. Choose the anomaly threshold from benign validation scores at a predeclared false-positive quantile. Do not use withheld attack labels to tune it.
9. Evaluate the frozen pipeline on the known-class test partition and the withheld attack family.
10. Preserve run configuration, software versions, and output metrics.

## Models
- Supervised baseline: Random Forest classifier.
- Novelty baseline: Isolation Forest trained on benign training flows.
- Hybrid decision: alert if the supervised model predicts a known attack OR the anomaly score crosses the threshold. A separate benign-vs-known-attack classifier is used for the supervised decision.

## Metrics
Report precision, recall, F1, balanced accuracy, confusion matrix, PR-AUC where meaningful, false-positive rate on benign traffic, withheld-family recall, and inference latency. Report per-class metrics and class support. Do not report accuracy alone under class imbalance.

## Explanation protocol
For each alert, store the predicted class, classifier confidence, anomaly score, threshold, and top global feature importances. The initial template explanation must state that feature importance is global and is not proof of causality. Later SHAP explanations should be evaluated for faithfulness and stability. LLM explanations must be assessed against structured evidence, with unsupported factual claims counted.

## Threats to validity
- Dataset age and synthetic/testbed traffic may not represent current production networks.
- A withheld dataset label is not equivalent to a true zero-day exploit.
- Duplicate or near-duplicate flows can inflate performance.
- Random splitting can leak temporal or scenario-specific patterns.
- Feature distributions and label semantics may differ across datasets.
- Threshold choice materially affects false-positive rate and must be predeclared.
