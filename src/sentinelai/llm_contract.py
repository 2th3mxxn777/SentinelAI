"""Prompt contract and deterministic validator for an optional LLM explanation.

This module does not call a hosted model. It validates that a renderer's output
uses only feature names, values, SHAP directions, and source IDs in retrieved
evidence. Semantic truth still requires human review or a separately validated
entailment evaluator.
"""
from __future__ import annotations
import json
from typing import Any

REQUIRED_TOP = {"alert_summary","model_probability_statement","evidence_items","limitations","abstained"}
REQUIRED_ITEM = {"feature","observed_value","definition","shap_effect","source_id"}
ALLOWED_EFFECTS = {"increases_attack_probability","decreases_attack_probability"}

def build_prompt(evidence: dict[str, Any], template: str) -> str:
    return template.replace("{{EVIDENCE_JSON}}", json.dumps(evidence, ensure_ascii=False, indent=2))

def validate_explanation(evidence: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    errors=[]
    if set(output) != REQUIRED_TOP:
        errors.append("top_level_keys_must_match_contract")
    if not isinstance(output.get("alert_summary"), str) or not output["alert_summary"].strip():
        errors.append("alert_summary_must_be_nonempty_string")
    if not isinstance(output.get("model_probability_statement"), str):
        errors.append("model_probability_statement_must_be_string")
    if not isinstance(output.get("limitations"), list) or not all(isinstance(x,str) for x in output.get("limitations",[])):
        errors.append("limitations_must_be_string_array")
    if not isinstance(output.get("abstained"), bool):
        errors.append("abstained_must_be_boolean")
    if not isinstance(output.get("evidence_items"), list):
        errors.append("evidence_items_must_be_array")
        items=[]
    else:
        items=output["evidence_items"]
    available={}
    for item in evidence.get("retrieved_evidence",[]):
        available.setdefault(item["feature"],[]).append(item)
    for i,item in enumerate(items):
        if not isinstance(item,dict) or set(item) != REQUIRED_ITEM:
            errors.append(f"evidence_item_{i}_keys_must_match_contract")
            continue
        matches=available.get(item.get("feature"),[])
        if not matches:
            errors.append(f"evidence_item_{i}_feature_not_retrieved")
            continue
        source_matches=[m for m in matches if m.get("source_id")==item.get("source_id")]
        if not source_matches:
            errors.append(f"evidence_item_{i}_source_id_not_supported")
            continue
        source=source_matches[0]
        if item.get("observed_value") != source.get("observed_value"):
            errors.append(f"evidence_item_{i}_observed_value_mismatch")
        if item.get("shap_effect") != source.get("direction"):
            errors.append(f"evidence_item_{i}_shap_direction_mismatch")
        if item.get("definition") != source.get("definition"):
            errors.append(f"evidence_item_{i}_definition_not_exactly_retrieved")
        if item.get("shap_effect") not in ALLOWED_EFFECTS:
            errors.append(f"evidence_item_{i}_invalid_shap_effect")
    if output.get("abstained") is True and items:
        errors.append("abstained_output_must_not_include_evidence_items")
    if output.get("abstained") is False and not items:
        errors.append("non_abstained_output_requires_evidence_items")
    return {"valid":not errors,"errors":errors,
            "validated_evidence_item_count":len(items) if not errors else 0,
            "semantic_truth_note":"Structural validation cannot prove that a paraphrase is semantically faithful; exact retrieved definitions are required by this baseline."}

def parse_and_validate(evidence: dict[str, Any], raw_output: str) -> dict[str, Any]:
    try:
        parsed=json.loads(raw_output)
    except json.JSONDecodeError as exc:
        return {"valid":False,"errors":[f"invalid_json:{exc.msg}"],"validated_evidence_item_count":0}
    return validate_explanation(evidence,parsed)
