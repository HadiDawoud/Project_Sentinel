import pytest
from sentinel.preprocessor import TextPreprocessor
from sentinel.rule_engine import RuleEngine


class TestEdgeCases:
    def setup_method(self):
        self.preprocessor = TextPreprocessor()
        self.engine = RuleEngine()

    def test_empty_string_preprocessing(self):
        result = self.preprocessor.preprocess("")
        assert result['cleaned'] == ""
        assert result['tokens'] == []

    def test_only_whitespace(self):
        result = self.preprocessor.preprocess("   \n\t   ")
        assert result['cleaned'] == ""
        assert result['token_count'] == 0

    def test_only_special_characters(self):
        result = self.preprocessor.preprocess("!@#$%^&*()")
        assert result['token_count'] == 0

    def test_only_unicode_emojis(self):
        result = self.preprocessor.preprocess("😀😁😂")
        assert "😀" not in result['cleaned']

    def test_very_long_text(self):
        text = "word " * 10000
        result = self.preprocessor.preprocess(text)
        assert result['token_count'] > 0

    def test_empty_rule_engine_analyze(self):
        result = self.engine.analyze("")
        assert result['flagged'] is False
        assert result['risk_score'] == 0

    def test_only_numbers(self):
        result = self.engine.analyze("123 456 789")
        assert result['flagged'] is False

    def test_only_punctuation(self):
        result = self.engine.analyze("!!! ??? ...")
        assert result['flagged'] is False

    def test_mixed_case_keywords(self):
        result = self.engine.check_keywords("KILL and DESTROY the enemy")
        assert result['flagged'] is True

    def test_keyword_partial_match(self):
        result = self.engine.check_keywords("killing in the morning")
        assert len(result['matched_terms']) == 0
