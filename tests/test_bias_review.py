import pytest
from sentinel.pipeline import SentinelPipeline

class TestBiasAndReview:
    def setup_method(self):
        self.pipeline = SentinelPipeline()

    def test_religious_context_bias(self):
        # This sentence uses 'jihad' in its spiritual/personal sense, not violent.
        text = "He is performing his jihad by struggling to be a better person."
        result = self.pipeline.classify(text)
        
        print(f"\nText: {text}")
        print(f"Label: {result['label']}")
        print(f"Risk Score: {result['risk_score']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Human Review: {result['requires_human_review']}")
        print(f"Bias Risk: {result['bias_metadata']['high_bias_risk']}")
        print(f"Reasoning: {result['reasoning']}")
        
        # It should require human review because of high bias risk terms
        assert result['requires_human_review'] is True
        assert result['bias_metadata']['high_bias_risk'] is True

    def test_political_struggle_bias(self):
        # 'Struggle', 'resist', 'fight' are in low_risk but can be used in peaceful political context.
        text = "We must resist these new tax laws through peaceful protest and struggle for our rights."
        result = self.pipeline.classify(text)
        
        print(f"\nText: {text}")
        print(f"Label: {result['label']}")
        print(f"Risk Score: {result['risk_score']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Human Review: {result['requires_human_review']}")
        print(f"Bias Risk: {result['bias_metadata']['high_bias_risk']}")
        
        # It should require human review because of political keywords and potential bias
        assert result['requires_human_review'] is True
        assert result['bias_metadata']['high_bias_risk'] is True

    def test_low_confidence_requires_review(self):
        # A case that is ambiguous should ideally be flagged for human review.
        text = "The situation is complex and requires a strong response."
        result = self.pipeline.classify(text)
        
        print(f"\nText: {text}")
        print(f"Label: {result['label']}")
        print(f"Risk Score: {result['risk_score']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Human Review: {result['requires_human_review']}")
        
        # It should require human review if confidence is low
        assert result['requires_human_review'] is True
