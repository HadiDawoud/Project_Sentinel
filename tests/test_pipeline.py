import pytest
from sentinel.preprocessor import TextPreprocessor
from sentinel.rule_engine import RuleEngine
from sentinel.fusion import ScoreFusion


class TestTextPreprocessor:
    def setup_method(self):
        self.preprocessor = TextPreprocessor()

    def test_clean_removes_urls(self):
        text = "Check this https://example.com awesome link"
        result = self.preprocessor.clean(text)
        assert "https://" not in result

    def test_clean_removes_mentions(self):
        text = "Hello @user123 how are you?"
        result = self.preprocessor.clean(text)
        assert "@user123" not in result

    def test_clean_lowercases_text(self):
        text = "HELLO WORLD"
        result = self.preprocessor.clean(text)
        assert result == "hello world"

    def test_tokenize_splits_text(self):
        text = "hello world test"
        result = self.preprocessor.tokenize(text)
        assert result == ["hello", "world", "test"]

    def test_preprocess_returns_all_fields(self):
        text = "Test input"
        result = self.preprocessor.preprocess(text)
        assert 'original' in result
        assert 'cleaned' in result
        assert 'tokens' in result
        assert 'token_count' in result

    def test_preprocess_empty_string(self):
        result = self.preprocessor.preprocess("")
        assert result["cleaned"] == ""
        assert result["tokens"] == []
        assert result["token_count"] == 0

    def test_preprocess_whitespace_only(self):
        result = self.preprocessor.preprocess("   \n\t  ")
        assert result["cleaned"] == ""
        assert result["token_count"] == 0


class TestRuleEngine:
    def setup_method(self):
        self.engine = RuleEngine("data/rules/keywords.yaml")

    def test_high_risk_keywords_detected(self):
        text = "We will kill and destroy our enemies"
        result = self.engine.check_keywords(text)
        assert result['flagged'] is True
        assert 'kill' in result['matched_terms']
        assert 'destroy' in result['matched_terms']

    def test_no_keywords_in_normal_text(self):
        text = "I had a nice day at the park"
        result = self.engine.check_keywords(text)
        assert result['flagged'] is False
        assert len(result['matched_terms']) == 0

    def test_empty_input_no_flags(self):
        result = self.engine.analyze("")
        assert result["flagged"] is False
        assert result["risk_score"] == 0

    def test_pattern_matching(self):
        text = "We must rise and eliminate those who oppose us"
        result = self.engine.check_patterns(text)
        assert 'must rise' in [p['pattern'] for p in result['matched_patterns']]

    def test_dehumanization_pattern_subhuman(self):
        text = "They are subhuman and unworthy of mercy"
        result = self.engine.check_patterns(text)
        patterns = [p['pattern'] for p in result['matched_patterns']]
        assert 'subhuman' in patterns

    def test_full_analysis(self):
        text = "The enemy must be destroyed by any means necessary"
        result = self.engine.analyze(text)
        assert result['flagged'] is True
        assert len(result['matched_terms']) > 0
        assert result['risk_score'] > 0


class TestScoreFusion:
    def setup_method(self):
        self.fusion = ScoreFusion()

    def test_fuses_rule_and_ml_results(self):
        rule_result = {
            'flagged': True,
            'matched_terms': ['kill', 'attack'],
            'risk_score': 50,
            'has_high_risk_terms': False
        }
        ml_result = {
            'label': 'Moderately Radical',
            'confidence': 0.7,
            'probabilities': {
                'Non-Radical': 0.1,
                'Mildly Radical': 0.1,
                'Moderately Radical': 0.6,
                'Highly Radical': 0.2
            }
        }
        result = self.fusion.fuse(rule_result, ml_result)
        assert 'label' in result
        assert 'risk_score' in result
        assert 'confidence' in result
        assert 'flagged_terms' in result

    def test_amplification_for_high_risk(self):
        rule_result = {
            'flagged': True,
            'matched_terms': ['kill'],
            'risk_score': 60,
            'has_high_risk_terms': True
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
        assert result['risk_score'] > 60

    def test_determine_label_scores(self):
        assert self.fusion._determine_label(10) == "Non-Radical"
        assert self.fusion._determine_label(35) == "Mildly Radical"
        assert self.fusion._determine_label(60) == "Moderately Radical"
        assert self.fusion._determine_label(85) == "Highly Radical"

    def test_fused_confidence_and_risk_within_bounds(self):
        rule_result = {
            'flagged': False,
            'matched_terms': [],
            'risk_score': 0,
            'has_high_risk_terms': False,
        }
        ml_result = {
            'label': 'Non-Radical',
            'confidence': 0.95,
            'probabilities': {
                'Non-Radical': 0.95,
                'Mildly Radical': 0.03,
                'Moderately Radical': 0.01,
                'Highly Radical': 0.01,
            },
        }
        result = self.fusion.fuse(rule_result, ml_result)
        assert 0 <= result['confidence'] <= 1
        assert 0 <= result['risk_score'] <= 100

    def test_high_risk_rules_cap_confidence_at_one(self):
        rule_result = {
            'flagged': True,
            'matched_terms': ['kill'],
            'risk_score': 90,
            'has_high_risk_terms': True,
        }
        ml_result = {
            'label': 'Highly Radical',
            'confidence': 0.99,
            'probabilities': {
                'Non-Radical': 0.0,
                'Mildly Radical': 0.0,
                'Moderately Radical': 0.01,
                'Highly Radical': 0.99,
            },
        }
        result = self.fusion.fuse(rule_result, ml_result)
        assert result['confidence'] <= 1.0
        assert result['risk_score'] <= 100
