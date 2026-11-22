import unittest
from sentinelai.llm_contract import validate_explanation

class LLMContractTests(unittest.TestCase):
    def setUp(self):
        self.evidence={"retrieved_evidence":[{
            "feature":"Flow Duration","observed_value":1000,
            "definition":"Duration of the network flow, measured in microseconds.",
            "direction":"increases_attack_probability","source_id":"CICFlowMeter-README-feature-list"}]}
    def valid_output(self):
        return {"alert_summary":"The model flagged this flow.",
            "model_probability_statement":"Attack probability was 0.91.",
            "evidence_items":[{"feature":"Flow Duration","observed_value":1000,
                "definition":"Duration of the network flow, measured in microseconds.",
                "shap_effect":"increases_attack_probability",
                "source_id":"CICFlowMeter-README-feature-list"}],
            "limitations":["SHAP is not causal evidence."],"abstained":False}
    def test_accepts_supported_output(self):
        self.assertTrue(validate_explanation(self.evidence,self.valid_output())["valid"])
    def test_rejects_invented_feature(self):
        out=self.valid_output(); out["evidence_items"][0]["feature"]="Destination Port"
        self.assertFalse(validate_explanation(self.evidence,out)["valid"])
    def test_rejects_wrong_shap_direction(self):
        out=self.valid_output(); out["evidence_items"][0]["shap_effect"]="decreases_attack_probability"
        self.assertFalse(validate_explanation(self.evidence,out)["valid"])
    def test_rejects_unsupported_source(self):
        out=self.valid_output(); out["evidence_items"][0]["source_id"]="made-up-source"
        self.assertFalse(validate_explanation(self.evidence,out)["valid"])
    def test_requires_abstention_without_evidence(self):
        out={"alert_summary":"Cannot explain.","model_probability_statement":"Unknown.",
             "evidence_items":[],"limitations":["Insufficient evidence."],"abstained":True}
        self.assertTrue(validate_explanation({"retrieved_evidence":[]},out)["valid"])

if __name__=="__main__": unittest.main()
