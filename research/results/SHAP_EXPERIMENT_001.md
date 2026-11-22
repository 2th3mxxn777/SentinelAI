# SentinelAI Explainability Pilot — SHAP Experiment 001

**Status:** Executed successfully as a preliminary local pilot.

## Purpose
Add per-flow explanations to SentinelAI's supervised Random Forest detector. SHAP values describe how individual input features move a model prediction relative to a background reference set.

## Protocol
- Input: uploaded, stratified CICIDS2017 pilot CSV.
- Sample after per-label cap: 15,383 flows.
- Numeric flow features retained: 66.
- Withheld family: `PortScan`; withheld records excluded from supervised model training.
- Known-label split: 75% training / 25% test, stratified on benign vs. represented attacks.
- Model: median imputation + 160-tree Random Forest with balanced class weights.
- Explainer: SHAP TreeExplainer, interventional feature perturbation, probability output.
- Background reference: 60 training rows.
- Explained examples: 12 highest-confidence predicted attack alerts from the known-class test partition.

## Numerical consistency check
For each explanation, the base attack probability plus SHAP contributions across all 66 features reconstructs the model's attack probability.
- Maximum absolute reconstruction error: approximately `1.44e-08`.
- Mean absolute reconstruction error: approximately `6.69e-09`.

This supports numerical consistency of the SHAP outputs for this pilot implementation. It does not establish model accuracy, causality, or analyst usefulness.

## Interpretation boundary
- Positive SHAP values increase attack probability relative to the background reference; negative values decrease it.
- SHAP values explain model behavior, not causal mechanisms or ground-truth malicious intent.
- This pilot explains known-class predicted alerts only. The held-out `PortScan` family is not claimed to have been detected or explained.
- Results use a capped pilot sample, not the full dataset or a natural traffic prevalence distribution.

## Next work
1. Add a documented feature-definition knowledge base and retrieval layer.
2. Generate explanations that cite only retrieved feature definitions and observed per-flow evidence.
3. Add an LLM only after evidence retrieval and claim validation are implemented.
4. Evaluate explanation faithfulness, unsupported claims, stability, and analyst usefulness.
