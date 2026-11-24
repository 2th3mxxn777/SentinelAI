\# Experiment 006 — Explainability and Evidence-Grounded RAG Explanations



\## 1. Objective



Experiment 006 evaluates the explainability component of SentinelAI. The objective is to generate feature-level explanations for network traffic classifications using SHAP and then convert those explanations into evidence-grounded, analyst-readable summaries using a deterministic Retrieval-Augmented Generation (RAG) pipeline.



This experiment investigates whether model explanations can be connected to documented network-feature meanings without inventing unsupported interpretations.



\## 2. Experimental Setup



The experiment used the processed CICIDS2017 dataset and the previously prepared SentinelAI explainability pipeline.



The SHAP explanation command was:



```bash

python -m sentinelai.explainability --data-dir data/processed --output-dir research/results/experiment\_006\_shap --holdout-label PortScan --max-rows 30000 --explain-count 20 --background-count 100

```



The RAG explanation command was:



```bash

python -m sentinelai.rag\_explain --input research/results/experiment\_006\_shap/shap\_explanations.json --glossary knowledge/feature\_glossary.json --output-dir research/results/experiment\_006\_rag

```



The explainability pipeline used a separate binary benign-versus-attack Random Forest model. Its SHAP results should not be interpreted as explanations of the multiclass entropy-based unknown-class detector evaluated in Experiment 005.



\## 3. SHAP Results



The SHAP pipeline completed successfully.



| Metric | Result |

|---|---:|

| Dataset rows after sampling | 22,143 |

| Feature count | 68 |

| Held-out label | PortScan |

| Explained alerts | 20 |

| Background samples | 100 |

| Maximum additivity error | 0.0000000143 |

| Mean additivity error | 0.0000000058 |



The very small additivity errors indicate that the calculated SHAP contributions closely satisfy the expected additive relationship for the explained model outputs.



These results demonstrate successful generation of feature-attribution explanations for the selected alerts. They do not independently establish that the explanations are causally correct or useful to human security analysts.



\## 4. RAG Explanation Results



The deterministic RAG pipeline processed the generated SHAP explanations using the feature glossary.



| Metric | Result |

|---|---:|

| Explanations generated | 20 |

| Resolved feature mentions | 138 |

| Unresolved feature mentions | 62 |

| Total feature mentions | 200 |

| Feature-mention resolution rate | 69% |



The pipeline successfully retrieved glossary-supported descriptions for 138 feature mentions. The remaining 62 mentions were unresolved and were not treated as verified feature definitions.



This conservative behavior is important: unsupported feature interpretations should not be presented as established facts in a cybersecurity explanation.



\## 5. Interpretation and Discussion



Experiment 006 demonstrates a functioning two-stage explainability workflow:



1\. SHAP generates feature-level contributions for selected network traffic predictions.

2\. A deterministic retrieval pipeline connects recognized features to documented glossary descriptions.

3\. The resulting explanations provide a structured foundation for analyst-facing reporting.



The 69% feature-mention resolution rate identifies a clear

