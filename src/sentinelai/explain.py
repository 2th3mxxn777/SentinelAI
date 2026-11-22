"""Evidence-bound, deterministic alert explanations (not an LLM)."""
def explain_alert(predicted_label: str, confidence: float, anomaly_score: float,
                 threshold: float, top_features: list[tuple[str, float]]) -> str:
    evidence = ", ".join(f"{name} (global importance {value:.4f})"
                         for name, value in top_features[:5]) or "No feature attribution available"
    anomaly_text = "exceeded" if anomaly_score >= threshold else "did not exceed"
    return (
        f"Alert classification: {predicted_label}. Classifier confidence: {confidence:.3f}. "
        f"The anomaly score {anomaly_text} the configured threshold "
        f"({anomaly_score:.4f} vs {threshold:.4f}). Top global model features: {evidence}. "
        "These feature importances describe model-level influence, not causal proof about this flow. "
        "Verify the raw flow and surrounding telemetry before taking action."
    )
