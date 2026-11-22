# SentinelAI LLM Explanation Contract — Design and Validation Harness

**Status:** Prompt contract and structural validation code implemented. No LLM provider has been called and no generated LLM outputs or LLM performance results are claimed.

## Purpose
Prepare a constrained interface for a future language-model explanation renderer. The model receives only the structured, retrieved evidence object; it must return a fixed JSON structure, cite retrieved source IDs, preserve observed values and SHAP directions, and abstain when evidence is insufficient.

## Implemented safeguards
- Prompt prohibits attack-family diagnosis, causal claims, and claims of attacker intent from feature values alone.
- Offline ground-truth labels are explicitly excluded as explanation evidence.
- Output must follow a fixed JSON contract.
- Every feature must exist in the retrieved evidence.
- Source IDs, observed values, SHAP directions, and definitions must exactly match retrieved evidence.
- The renderer must abstain if it cannot provide an evidence-backed explanation.
- Invalid JSON and unsupported fields are rejected.

## Automated tests
Five unit tests cover supported output, invented feature rejection, wrong SHAP direction rejection, unsupported citation rejection, and abstention when no evidence is available.

## Important limitation
Structural checks do not establish semantic truth. A language model could still make an unsupported statement in free-text summary fields. A future evaluation must add claim-level review, human annotation, and/or a validated semantic entailment procedure. The current harness is a guardrail, not proof of hallucination-free output.

## Next evaluation
Compare the deterministic RAG explanation with constrained LLM output on the same alerts. Score unsupported claims per explanation, missing or invalid citations, SHAP-direction contradictions, abstention correctness, readability, and latency. Keep the deterministic explanation as the reference baseline.
