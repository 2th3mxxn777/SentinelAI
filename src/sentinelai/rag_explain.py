"""Evidence-constrained retrieval and explanation for SentinelAI SHAP records.

This is a deterministic RAG baseline. It retrieves documented feature semantics
and renders a structured explanation; it does not call an LLM.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Any

def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())

def load_glossary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def retrieve_feature(feature: str, glossary: dict[str, Any]) -> dict[str, Any] | None:
    target = normalize(feature)
    for item in glossary["features"]:
        if target and any(target == normalize(n) for n in [item["name"], *item.get("aliases", [])]):
            return item
    # No fuzzy guess: unsupported feature names remain uncited.
    return None

def explain_record(record: dict[str, Any], glossary: dict[str, Any]) -> dict[str, Any]:
    source = glossary["source"]
    evidence, unsupported = [], []
    for item in record.get("top_feature_contributions", []):
        doc = retrieve_feature(item["feature"], glossary)
        if doc is None:
            unsupported.append(item["feature"])
            continue
        evidence.append({
            "feature": item["feature"], "observed_value": item.get("observed_value"),
            "shap_value_attack_probability": item.get("shap_value_attack_probability"),
            "direction": item.get("direction"), "definition": doc["definition"],
            "source_id": source["source_id"], "source_title": source["title"],
            "source_url": source["url"], "source_location": doc["source_location"]
        })
    probability = float(record["attack_probability"])
    decision = "crosses" if probability >= 0.5 else "does not cross"
    summary = (f"The classifier assigned an attack probability of {probability:.4f}; "
        f"this {decision} the prototype's 0.5 alert threshold. The listed SHAP contributions "
        "describe how these features moved this model's attack probability relative to its "
        "SHAP background reference.")
    return {
        "explanation_id": record.get("explanation_id"),
        "dataset_row_index": record.get("dataset_row_index"),
        "true_label_for_offline_evaluation_only": record.get("true_label"),
        "predicted_label": record.get("predicted_label"),
        "attack_probability": probability, "summary": summary,
        "retrieved_evidence": evidence, "unresolved_features": unsupported,
        "guardrails": [
            "A feature definition describes what a field measures; it does not imply that a high or low value is malicious.",
            "SHAP values describe this model's attribution, not causality or proof of attacker intent.",
            "The dataset ground-truth label is retained for offline evaluation and must not be presented as observed telemetry evidence.",
            "No threat-family diagnosis is inferred from feature definitions alone."
        ],
        "retrieval_status": "complete" if not unsupported else "partial_with_unresolved_features"
    }

def run(input_path: Path, glossary_path: Path, output_dir: Path) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    glossary = load_glossary(glossary_path)
    explanations = [explain_record(row, glossary) for row in payload.get("explanations", [])]
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": "SentinelAI evidence retrieval and deterministic RAG pilot",
        "input_experiment": payload.get("experiment"),
        "retrieval_method": "normalized exact feature-name/alias matching against curated CICFlowMeter glossary",
        "source": glossary["source"], "explanation_count": len(explanations),
        "resolved_feature_mentions": sum(len(x["retrieved_evidence"]) for x in explanations),
        "unresolved_feature_mentions": sum(len(x["unresolved_features"]) for x in explanations),
        "explanations": explanations,
        "limitations": [
            "This is retrieval plus deterministic templating, not an LLM-generated explanation.",
            "The glossary covers documented feature definitions; unknown names are left unresolved rather than guessed.",
            "The current corpus contains one CICFlowMeter feature-reference source and is not a broad cybersecurity knowledge base.",
            "No analyst usefulness study or causal explanation validation has been performed."
        ]
    }
    (output_dir/"rag_explanations.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = ["# SentinelAI Evidence-Grounded Explanations", "", f"Generated explanations: {len(explanations)}", ""]
    for e in explanations:
        md += [f"## Alert {e['explanation_id']}", "", e["summary"], "", "Retrieved feature evidence:"]
        for f in e["retrieved_evidence"]:
            sign = "increased" if float(f["shap_value_attack_probability"] or 0) > 0 else "decreased"
            md.append(f"- **{f['feature']} = {f['observed_value']}** — {f['definition']} SHAP {sign} attack probability by {float(f['shap_value_attack_probability']):.6f}. [Source: {f['source_id']}, {f['source_location']}]({f['source_url']})")
        if e["unresolved_features"]:
            md.append("- **Not explained (no exact glossary match):** " + ", ".join(e["unresolved_features"]))
        md += ["", "> SHAP attribution is not causal evidence. Feature definitions alone do not establish maliciousness.", ""]
    (output_dir/"rag_explanations.md").write_text("\n".join(md), encoding="utf-8")
    return {k: result[k] for k in ("experiment","explanation_count","resolved_feature_mentions","unresolved_feature_mentions","limitations")}

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="experiments/results/explanations/shap_explanations.json")
    p.add_argument("--glossary", default="knowledge/feature_glossary.json")
    p.add_argument("--output-dir", default="experiments/results/rag_explanations")
    a = p.parse_args()
    print(json.dumps(run(Path(a.input), Path(a.glossary), Path(a.output_dir)), indent=2))

if __name__ == "__main__":
    main()
