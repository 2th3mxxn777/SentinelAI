# SentinelAI RAG Experiment 001 — Evidence Retrieval Pilot

**Status:** Implemented and executed on the 12 SHAP explanations from SHAP Experiment 001. This is a deterministic retrieval-and-rendering baseline, not an LLM experiment.

## Objective
Test whether each feature cited by a SHAP explanation can be connected to a documented CICFlowMeter feature definition, while leaving unsupported or ambiguous fields unresolved instead of inventing an explanation.

## Knowledge source
A curated feature glossary was created from the CICFlowMeter ReadMe.txt feature reference. Each glossary entry stores its feature name, aliases, definition, source identifier, source URL, and source-line locator. The source describes the measurement represented by each feature; it does not describe whether a value is malicious.

## Retrieval method
1. Read a SHAP explanation record containing observed feature values and signed SHAP contributions.
2. Normalize feature names and perform exact matching against glossary names and aliases.
3. Attach the retrieved definition and source table locator to the feature evidence.
4. Render a deterministic explanation using the model's attack probability, the SHAP contribution direction, observed values, and source-backed definitions.
5. Mark features with no exact glossary match as unresolved. No fuzzy matching or unsupported semantic inference is used.

## Results
- SHAP alerts processed: 12.
- Top-feature mentions presented to retrieval: 120.
- Feature mentions resolved to a glossary entry: 115.
- Unresolved mentions: 5.
- Retrieval coverage: 86.67% (104 / 120).
- All 5 unresolved mentions were Fwd Header Length.1, a duplicate-column suffix. It was intentionally not silently mapped to Fwd Header Length because the duplicate column's exact provenance was not established.
- Automated tests: 3 passed (source citation on exact retrieval, unresolved-field handling, and explanation guardrails).

## Claim boundaries
The generated explanation may report the classifier's recorded attack probability, the SHAP direction, and the documented meaning of a feature. It must not infer maliciousness from a feature value, treat SHAP as causality, present a dataset label as observed telemetry, or diagnose a threat family without separate validated evidence.

## Limitations
1. Exact-name retrieval from one curated feature-reference source, not a broad cybersecurity RAG corpus.
2. Retrieval coverage is not a measure of definition correctness, explanation faithfulness, or analyst usefulness.
3. Deterministic templates only; no LLM was called.
4. No analyst study, adversarial prompt test, or cross-dataset test has been performed.

## Next experiment
Add a constrained LLM renderer that receives only the retrieved evidence object, must cite source IDs, and returns a schema-validated explanation. Evaluate unsupported claims, missing citations, contradictions with SHAP direction, and latency against this deterministic baseline. The LLM must be allowed to abstain when evidence is insufficient.
