import pytest
from sentinel.rule_engine import RuleEngine


class TestRuleEngine:
    def setup_method(self):
        self.engine = RuleEngine()

    def test_check_keywords_high_risk(self):
        text = "They will kill and destroy the enemy"
        result = self.engine.check_keywords(text)
        assert result["flagged"] is True
        assert "kill" in result["matched_terms"]
        assert "destroy" in result["matched_terms"]
        assert result["risk_score"] > 0

    def test_check_keywords_low_risk(self):
        text = "We must fight for our freedom"
        result = self.engine.check_keywords(text)
        assert result["flagged"] is True
        assert "fight" in result["matched_terms"]

    def test_check_keywords_no_match(self):
        text = "This is a normal peaceful message"
        result = self.engine.check_keywords(text)
        assert result["flagged"] is False
        assert result["matched_terms"] == []

    def test_check_patterns_call_to_violence(self):
        text = "We must rise up and take up arms"
        result = self.engine.check_patterns(text)
        assert result["flagged"] is True
        assert result["pattern_count"] > 0

    def test_check_patterns_no_match(self):
        text = "Hello world"
        result = self.engine.check_patterns(text)
        assert result["flagged"] is False
        assert result["pattern_count"] == 0

    def test_analyze_combines_results(self):
        text = "They will kill us all"
        result = self.engine.analyze(text)
        assert "flagged" in result
        assert "risk_score" in result
        assert "matched_terms" in result

    def test_analyze_has_high_risk_terms(self):
        text = "We must eliminate the threat"
        result = self.engine.analyze(text)
        assert result["has_high_risk_terms"] is True
