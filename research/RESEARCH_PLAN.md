# SentinelAI Research Plan

## Working title
**Evidence-Grounded Explanations for Hybrid Network Intrusion Detection Under Withheld Attack Categories**

## Research question
How can an AI-assisted network intrusion detection system identify previously unseen cyber threats while generating accurate, evidence-grounded explanations that security analysts can inspect?

## Research questions
- RQ1: How well does a supervised baseline detect attack categories represented during training?
- RQ2: How does a hybrid supervised/anomaly detector perform when one attack family is withheld from supervised training?
- RQ3: Do explanations accurately reflect the model evidence available for each alert?
- RQ4: What trade-offs arise among detection performance, false-positive rate, explanation quality, and computational cost?

## Hypotheses (to test, not assume)
- H1: Adding anomaly detection may improve recall for a withheld attack family, potentially at the cost of more false positives.
- H2: Explanations constrained to recorded model evidence will contain fewer unsupported claims than free-form explanations.
- H3: Explanation reliability may differ between known attacks, withheld attacks, and benign flows incorrectly flagged as anomalous.

## Proposed contribution
A reproducible evaluation protocol that separates (1) detection of known classes, (2) anomaly-based recognition of a withheld class, and (3) the factual support of alert explanations. The novelty claim must be refined only after a verified literature review.

## Scope and honesty
A public dataset with a withheld attack family is a controlled proxy for unknown-class generalization. It does not establish detection of every zero-day vulnerability or generalization to live networks. No results are asserted in this plan.

## Planned extensions
1. Compare Random Forest with XGBoost or another justified baseline.
2. Compare Isolation Forest with an autoencoder.
3. Add SHAP local explanations and measure feature-attribution stability.
4. Add an LLM/RAG condition that receives only structured evidence; evaluate unsupported statements, evidence consistency, and latency.
5. Replicate the protocol on a second dataset such as UNSW-NB15, with careful feature mapping.
