import pytest
from sentinel.pipeline import SentinelPipeline

class TestBiasAndReview:
    def setup_method(self):
        self.pipeline = SentinelPipeline()

    def test_religious_context_bias(self):
        # Nonviolent religious sense of "jihad"; baseline for false-positive review.
        text = "He is performing his jihad by struggling to be a better person."
        result = self.pipeline.classify(text)

        print(f"\nText: {text}")
        print(f"Label: {result['label']}")
        print(f"Risk Score: {result['risk_score']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Reasoning: {result['reasoning']}")

    def test_political_struggle_bias(self):
        # Peaceful civic language overlapping low_risk lexicon ("resist", "struggle").
        text = "We must resist these new tax laws through peaceful protest and struggle for our rights."
        result = self.pipeline.classify(text)
        
        print(f"\nText: {text}")
        print(f"Label: {result['label']}")
        print(f"Risk Score: {result['risk_score']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Reasoning: {result['reasoning']}")

    def test_low_confidence_requires_review(self):
        # Intentionally vague phrasing; output is the regression baseline until review flags exist.
        text = "The situation is complex and requires a strong response."
        result = self.pipeline.classify(text)

        print(f"\nText: {text}")
        print(f"Label: {result['label']}")
        print(f"Risk Score: {result['risk_score']}")
        print(f"Confidence: {result['confidence']}")
