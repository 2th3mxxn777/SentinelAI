import unittest
from pathlib import Path
from sentinelai.rag_explain import explain_record, load_glossary

ROOT=Path(__file__).resolve().parents[1]
GLOSSARY=load_glossary(ROOT/"knowledge"/"feature_glossary.json")

class RagExplainTests(unittest.TestCase):
    def sample(self):
        return {"explanation_id":1,"dataset_row_index":123,"true_label":"DDoS",
          "predicted_label":"attack","attack_probability":0.91,
          "top_feature_contributions":[
            {"feature":"Flow Duration","observed_value":1200,
             "shap_value_attack_probability":0.12,"direction":"increases_attack_probability"},
            {"feature":"Fwd Header Length.1","observed_value":64,
             "shap_value_attack_probability":-0.02,"direction":"decreases_attack_probability"}]}
    def test_exact_retrieval_has_source_citation(self):
        r=explain_record(self.sample(),GLOSSARY)
        self.assertEqual(len(r["retrieved_evidence"]),1)
        self.assertEqual(r["retrieved_evidence"][0]["source_id"],"CICFlowMeter-README-feature-list")
        self.assertTrue(r["retrieved_evidence"][0]["source_url"].startswith("https://"))
    def test_unmatched_feature_is_not_guessed(self):
        r=explain_record(self.sample(),GLOSSARY)
        self.assertEqual(r["unresolved_features"],["Fwd Header Length.1"])
        self.assertEqual(r["retrieval_status"],"partial_with_unresolved_features")
    def test_guardrails_are_present(self):
        r=explain_record(self.sample(),GLOSSARY)
        self.assertTrue(any("not causality" in x.lower() or "not causal" in x.lower() for x in r["guardrails"]))

if __name__=="__main__": unittest.main()
