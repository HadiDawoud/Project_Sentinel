import pytest
from sentinel.fusion import ScoreFusion


class TestScoreFusion:
    def setup_method(self):
        self.fusion = ScoreFusion()

    def test_fuse_combines_scores(self):
        rule_result = {
            'risk_score': 50,
            'matched_terms': ['enemy'],
            'has_high_risk_terms': False,
            'flagged': True
        }
        ml_result = {
            'label': 'Mildly Radical',
            'confidence': 0.7,
            'probabilities': {
                'Non-Radical': 0.2,
                'Mildly Radical': 0.6,
                'Moderately Radical': 0.15,
                'Highly Radical': 0.05
            }
        }
        result = self.fusion.fuse(rule_result, ml_result)
        assert 'label' in result
        assert 'risk_score' in result
        assert 'confidence' in result

    def test_fuse_amplifies_high_risk(self):
        rule_result = {
            'risk_score': 30,
            'matched_terms': ['kill'],
            'has_high_risk_terms': True,
            'flagged': True
        }
        ml_result = {
            'label': 'Moderately Radical',
            'confidence': 0.6,
            'probabilities': {
                'Non-Radical': 0.1,
                'Mildly Radical': 0.2,
                'Moderately Radical': 0.5,
                'Highly Radical': 0.2
            }
        }
        result = self.fusion.fuse(rule_result, ml_result)
        assert result['rule_amplification'] is True

    def test_compute_ml_risk_score(self):
        ml_scores = {
            'Non-Radical': 0.5,
            'Mildly Radical': 0.3,
            'Moderately Radical': 0.15,
            'Highly Radical': 0.05
        }
        score = self.fusion._compute_ml_risk_score(ml_scores)
        assert 0 <= score <= 1

    def test_determine_label_ranges(self):
        assert self.fusion._determine_label(10) == "Non-Radical"
        assert self.fusion._determine_label(30) == "Mildly Radical"
        assert self.fusion._determine_label(60) == "Moderately Radical"
        assert self.fusion._determine_label(90) == "Highly Radical"

    def test_generate_reasoning(self):
        rule_result = {
            'matched_terms': ['enemy'],
            'keyword_details': {'medium_risk': ['enemy']},
            'has_high_risk_terms': False
        }
        ml_result = {
            'label': 'Mildly Radical',
            'confidence': 0.75
        }
        reasoning = self.fusion._generate_reasoning(rule_result, ml_result, 35)
        assert 'Mildly Radical' in reasoning
        assert isinstance(reasoning, str)
