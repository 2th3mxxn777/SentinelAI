# SentinelAI

**Research prototype:** hybrid network intrusion detection with controlled unseen-attack evaluation and evidence-grounded explanations.

> Status: initial implementation scaffold. No experimental results are claimed until the pipeline is run on a documented dataset.

## Research question
Can a hybrid detector identify attack families withheld from supervised training, and can structured evidence support reliable explanations of its alerts?

## Implemented scaffold
- CSV ingestion for CICIDS-style flow datasets
- Label/feature cleanup
- Random Forest known-class baseline
- Isolation Forest anomaly model fitted only on benign training flows
- Hybrid alert rule and evidence fields
- Metrics and prediction exports
- Deterministic evidence-bound explanation template

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

Run a controlled withheld-label experiment (replace the label with an exact value printed above):
```powershell
python -m sentinelai.experiment --data-dir datasets/raw/cicids2017 --holdout-label "PortScan"
```
Outputs: `experiments/results/metrics.json` and `predictions.csv`.

## Research design and limitations
The selected attack label is excluded from supervised fitting. The anomaly model is fitted only on benign training examples. Its threshold is selected from benign validation scores before evaluating the withheld class.

A withheld dataset label is a **controlled unknown-class benchmark**, not proof of detecting arbitrary real-world zero-day vulnerabilities. Inspect duplicates, temporal leakage, dataset limitations, class imbalance, and label quality before making strong claims.

The current feature-importance explanation is global model evidence, not causal proof. The template explanation is deterministic, not an LLM. LLM/RAG explanation reliability remains a planned extension. Do not invent or pre-fill experimental results.

See `research/RESEARCH_PLAN.md`, `research/METHODOLOGY.md`, and `research/EXPERIMENT_LOG_TEMPLATE.md`.
