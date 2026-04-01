import pytest
from sentinel.fusion import ScoreFusion


class TestConfidenceScoring:
    def setup_method(self):
        self.fusion = ScoreFusion()

    def test_confidence_bounds_zero(self):
        rule_result = {
            'flagged': False,
            'matched_terms': [],
            'risk_score': 0,
            'has_high_risk_terms': False
        }
        ml_result = {
            'label': 'Non-Radical',
            'confidence': 0.5,
            'probabilities': {'Non-Radical': 0.5, 'Mildly Radical': 0.2, 'Moderately Radical': 0.2, 'Highly Radical': 0.1}
        }
        result = self.fusion.fuse(rule_result, ml_result)
        assert 0 <= result['confidence'] <= 1

    def test_confidence_bounds_high(self):
        rule_result = {
            'flagged': True,
            'matched_terms': ['kill'],
            'risk_score': 80,
            'has_high_risk_terms': True
        }
        ml_result = {
            'label': 'Highly Radical',
            'confidence': 0.99,
            'probabilities': {'Non-Radical': 0.0, 'Mildly Radical': 0.0, 'Moderately Radical': 0.01, 'Highly Radical': 0.99}
        }
        result = self.fusion.fuse(rule_result, ml_result)
        assert 0 <= result['confidence'] <= 1

    def test_high_confidence_with_rule_certainty(self):
        rule_result = {
            'flagged': True,
            'matched_terms': ['attack', 'bomb'],
            'risk_score': 60,
            'has_high_risk_terms': True
        }
        ml_result = {
            'label': 'Moderately Radical',
            'confidence': 0.85,
            'probabilities': {'Non-Radical': 0.05, 'Mildly Radical': 0.1, 'Moderately Radical': 0.85, 'Highly Radical': 0.0}
        }
        result = self.fusion.fuse(rule_result, ml_result)
        assert result['confidence'] > 0.7

    def test_low_confidence_uncertain(self):
        rule_result = {
            'flagged': False,
            'matched_terms': [],
            'risk_score': 0,
            'has_high_risk_terms': False
        }
        ml_result = {
            'label': 'Non-Radical',
            'confidence': 0.35,
            'probabilities': {'Non-Radical': 0.35, 'Mildly Radical': 0.3, 'Moderately Radical': 0.2, 'Highly Radical': 0.15}
        }
        result = self.fusion.fuse(rule_result, ml_result)
        assert result['confidence'] < 0.5
