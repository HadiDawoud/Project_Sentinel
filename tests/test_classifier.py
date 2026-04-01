import pytest
from unittest.mock import patch, MagicMock
from sentinel.classifier import RadicalClassifier


class TestRadicalClassifier:
    def setup_method(self):
        self.classifier = RadicalClassifier()

    def test_classifier_initializes(self):
        assert self.classifier.model_name == "distilbert-base-uncased"
        assert self.classifier.num_labels == 4
        assert self.classifier.labels is not None

    def test_predict_returns_label(self):
        result = self.classifier.predict("This is a test")
        assert 'label' in result
        assert result['label'] in self.classifier.labels.values()

    def test_predict_returns_confidence(self):
        result = self.classifier.predict("Test text")
        assert 'confidence' in result
        assert 0 <= result['confidence'] <= 1

    def test_predict_returns_probabilities(self):
        result = self.classifier.predict("Sample text")
        assert 'probabilities' in result
        assert len(result['probabilities']) == 4

    def test_predict_batch_returns_list(self):
        texts = ["Test one", "Test two", "Test three"]
        results = self.classifier.predict_batch(texts)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_predict_batch_each_has_label(self):
        texts = ["First", "Second"]
        results = self.classifier.predict_batch(texts)
        for result in results:
            assert 'label' in result

    def test_get_fine_grained_scores_returns_dict(self):
        scores = self.classifier.get_fine_grained_scores("Test")
        assert isinstance(scores, dict)
        assert len(scores) == 4

    def test_probabilities_sum_to_one(self):
        result = self.classifier.predict("Test text")
        prob_sum = sum(result['probabilities'].values())
        assert abs(prob_sum - 1.0) < 0.01

    def test_predict_batch_probabilities_sum(self):
        texts = ["a", "b", "c"]
        results = self.classifier.predict_batch(texts)
        for result in results:
            prob_sum = sum(result['probabilities'].values())
            assert abs(prob_sum - 1.0) < 0.01
