# SentinelAI

**Research prototype:** hybrid network intrusion detection with controlled unseen-attack evaluation and evidence-grounded explanations.

> Status: working research prototype with preliminary pilot experiments. Results are not full-dataset benchmarks.

## Research question
Can a hybrid detector identify attack families withheld from supervised training, and can structured evidence support reliable explanations of its alerts?

## Implemented
- CSV ingestion for CICIDS-style flow datasets
- Label/feature cleanup
- Random Forest known-class baseline
- Isolation Forest anomaly model fitted only on benign training flows
- Hybrid alert rule and metrics/prediction exports
- Detector comparison pilot (Random Forest, Isolation Forest, LOF, and score fusion)
- Per-flow SHAP explanations for Random Forest predictions
- SHAP contribution export with probability reconstruction audit

## Quick start (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download CICIDS2017 from https://www.unb.ca/cic/datasets/ids-2017.html and place extracted CSV files in `datasets/raw/cicids2017/`. Do not commit raw data.

Inspect exact labels:
```powershell
$env:PYTHONPATH = "src"
python -m sentinelai.experiment --data-dir datasets/raw/cicids2017 --list-labels
```

Run a controlled withheld-label experiment:
```powershell
python -m sentinelai.experiment --data-dir datasets/raw/cicids2017 --holdout-label "PortScan"
```

## Generate local SHAP explanations
Install dependencies from `requirements.txt`, then run:
```powershell
$env:PYTHONPATH = "src"
python -m sentinelai.explainability --data-dir datasets/raw/cicids2017 --holdout-label "PortScan" --max-rows 30000 --explain-count 25
```
Outputs are written to `experiments/results/explanations/`:
- `shap_explanations.json`: per-flow attack probability, base value, local SHAP contributions, and caveats.
- `shap_feature_contributions.csv`: flattened top-feature contributions.

Positive SHAP values increase the model's attack probability relative to the background reference; negative values decrease it. SHAP explains model behavior, not causality or malicious intent. The current script explains known-class test alerts; it does not claim that the held-out class was detected.

## Research results
- `research/results/PILOT_EXPERIMENT_001.md`: initial withheld-PortScan baseline.
- `research/results/PILOT_EXPERIMENT_002.md`: anomaly detector comparison.
- `research/results/SHAP_EXPERIMENT_001.md`: per-flow explainability pilot.

## Research design and limitations
A withheld dataset label is a **controlled unknown-class benchmark**, not proof of detecting arbitrary real-world zero-day vulnerabilities. Pilot samples are capped by label and do not represent natural traffic prevalence. Inspect duplicates, temporal leakage, dataset limitations, class imbalance, and label quality before making strong claims.

LLM/RAG explanations, evidence retrieval, explanation faithfulness evaluation, and independent cross-dataset validation remain future work. Do not invent or pre-fill experimental results.

See `research/RESEARCH_PLAN.md`, `research/METHODOLOGY.md`, and `research/EXPERIMENT_LOG_TEMPLATE.md`.
