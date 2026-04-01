import re
import string
from typing import List


class TextPreprocessor:
    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.email_pattern = re.compile(r'\S+@\S+\.\S+')
        self.whitespace_pattern = re.compile(r'\s+')
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )

    def clean(self, text: str) -> str:
        text = text.lower()
        text = self.emoji_pattern.sub(' ', text)
        text = self.url_pattern.sub(' ', text)
        text = self.mention_pattern.sub(' ', text)
        text = self.email_pattern.sub(' ', text)
        text = self._remove_special_chars(text)
        text = self.whitespace_pattern.sub(' ', text)
        return text.strip()

    def _remove_special_chars(self, text: str) -> str:
        allowed = set(string.ascii_lowercase + string.digits + string.whitespace + "'")
        return ''.join(c if c in allowed else ' ' for c in text)

    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def preprocess(self, text: str) -> dict:
        cleaned = self.clean(text)
        tokens = self.tokenize(cleaned)
        return {
            'original': text,
            'cleaned': cleaned,
            'tokens': tokens,
            'token_count': len(tokens)
        }
