import re
import string
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor


class TextPreprocessor:
    def __init__(self, num_workers: int = 1):
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
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
        self._num_workers = max(1, num_workers)

    def __repr__(self):
        return f"TextPreprocessor()"

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

    def preprocess(self, text: str) -> Dict[str, Any]:
        cleaned = self.clean(text)
        tokens = self.tokenize(cleaned)
        return {
            'original': text,
            'cleaned': cleaned,
            'tokens': tokens,
            'token_count': len(tokens)
        }

    def preprocess_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not texts:
            return []
        if len(texts) < 10 or self._num_workers <= 1:
            return [self.preprocess(t) for t in texts]
        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            results = list(executor.map(self.preprocess, texts))
        return results
