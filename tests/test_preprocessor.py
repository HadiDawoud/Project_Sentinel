import pytest
from sentinel.preprocessor import TextPreprocessor


class TestTextPreprocessor:
    def setup_method(self):
        self.preprocessor = TextPreprocessor()

    def test_clean_removes_urls(self):
        text = "Check this link http://example.com for more info"
        result = self.preprocessor.clean(text)
        assert "http://example.com" not in result

    def test_clean_removes_mentions(self):
        text = "Hello @user123 and @anotheruser"
        result = self.preprocessor.clean(text)
        assert "@user123" not in result

    def test_clean_removes_emojis(self):
        text = "This is a test 😀 with emojis 🎉"
        result = self.preprocessor.clean(text)
        assert "😀" not in result
        assert "🎉" not in result

    def test_clean_lowercases_text(self):
        text = "UPPERCASE TEXT"
        result = self.preprocessor.clean(text)
        assert result == "uppercase text"

    def test_tokenize_splits_text(self):
        text = "hello world test"
        tokens = self.preprocessor.tokenize(text)
        assert tokens == ["hello", "world", "test"]

    def test_preprocess_returns_dict(self):
        text = "Test input text"
        result = self.preprocessor.preprocess(text)
        assert "original" in result
        assert "cleaned" in result
        assert "tokens" in result
        assert "token_count" in result

    def test_preprocess_token_count(self):
        text = "one two three four"
        result = self.preprocessor.preprocess(text)
        assert result["token_count"] == 4
