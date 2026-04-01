import pytest
from sentinel.pipeline import SentinelPipeline


@pytest.mark.integration
class TestFullPipeline:
    def setup_method(self):
        self.pipeline = SentinelPipeline()

    def test_end_to_end_single_text(self):
        text = "We will fight for our freedom and resist oppression"
        result = self.pipeline.classify(text)
        
        assert result['label'] in ['Non-Radical', 'Mildly Radical', 'Moderately Radical', 'Highly Radical']
        assert 0 <= result['risk_score'] <= 100
        assert 0 <= result['confidence'] <= 1
        assert 'audit_id' in result
        assert 'timestamp' in result

    def test_end_to_end_batch_processing(self):
        texts = [
            "This is a peaceful message about nothing",
            "We must fight against injustice",
            "The radicals will attack by any means necessary"
        ]
        results = self.pipeline.classify_batch(texts)
        
        assert len(results) == 3
        for result in results:
            assert 'label' in result
            assert 'risk_score' in result
            assert 'confidence' in result

    def test_high_risk_text_classification(self):
        text = "They will kill and destroy the enemy by all means necessary"
        result = self.pipeline.classify(text)
        
        assert result['risk_score'] > 0
        assert len(result['flagged_terms']) > 0

    def test_low_risk_text_classification(self):
        text = "I love reading books about history and culture"
        result = self.pipeline.classify(text)
        
        assert result['label'] == 'Non-Radical' or result['label'] == 'Mildly Radical'
        assert result['confidence'] > 0

    def test_classify_batch_with_parallel_flag(self):
        texts = ["Test one", "Test two", "Test three"]
        results = self.pipeline.classify_batch(texts, parallel=False)
        assert len(results) == 3

    def test_pipeline_handles_empty_string(self):
        result = self.pipeline.classify("")
        assert 'label' in result
        assert 'risk_score' in result

    def test_pipeline_handles_unicode_text(self):
        text = "This is a test with émoji 😀 and unicode ✓"
        result = self.pipeline.classify(text)
        assert result is not None

    def test_pipeline_handles_long_text(self):
        text = " ".join(["word"] * 1000)
        result = self.pipeline.classify(text)
        assert result['label'] is not None
