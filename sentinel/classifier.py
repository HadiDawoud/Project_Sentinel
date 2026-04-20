import torch
import numpy as np
import signal
import platform
import os
import time
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

from .constants import LABEL_MAP, DEFAULT_MAX_LENGTH
from .exceptions import ModelLoadError, PredictionError


class TimeoutError(Exception):
    pass


def _compute_device() -> torch.device:
    if os.environ.get('SENTINEL_DEVICE'):
        return torch.device(os.environ['SENTINEL_DEVICE'])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RadicalClassifier:
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 4,
        checkpoint_path: Optional[str] = None,
        lazy_load: bool = False,
        max_retries: int = 3,
        retry_delay: float = 0.5
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.checkpoint_path = checkpoint_path
        self.device = _compute_device()
        self.tokenizer = None
        self.model = None
        self._is_loaded = False
        self._inference_count = 0
        self._retry_count = 0
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        if not lazy_load:
            self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> None:
        if self._is_loaded:
            return
        self._load_model()
        self._is_loaded = True

    def _load_model(self) -> None:
        valid_checkpoint = False
        try:
            if self.checkpoint_path and Path(self.checkpoint_path).exists():
                if any(Path(self.checkpoint_path).iterdir()):
                    valid_checkpoint = True
            
            if valid_checkpoint:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.checkpoint_path,
                    num_labels=self.num_labels
                )
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=self.num_labels
                )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}") from e

    def warmup(self, num_inferences: int = 3) -> None:
        self._ensure_model_loaded()
        warmup_texts = [
            "This is a normal conversation about everyday topics.",
            "We should discuss the importance of community safety.",
            "Sharing knowledge about historical events and cultures."
        ]
        for _ in range(num_inferences):
            for text in warmup_texts:
                self._dummy_inference(text)

    def _dummy_inference(self, text: str) -> None:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=DEFAULT_MAX_LENGTH,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = self.model(**inputs)

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def predict(self, text: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        self._ensure_model_loaded()
        
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                if timeout and platform.system() != 'Windows':
                    def timeout_handler(signum, frame):
                        raise TimeoutError(f"Prediction timed out after {timeout}s")
                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(timeout))
                    try:
                        return self._predict_impl(text)
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
                else:
                    return self._predict_impl(text)
            except (RuntimeError, torch.cuda.OutOfMemoryError, OSError) as e:
                last_exception = e
                self._retry_count += 1
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise PredictionError(f"Prediction failed after {self.max_retries} attempts: {e}") from e
        
        raise PredictionError(f"Prediction failed: {last_exception}")
    
    def _predict_impl(self, text: str) -> Dict[str, Any]:
        self._inference_count += 1
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=DEFAULT_MAX_LENGTH,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()

        probs = probabilities[0].cpu().numpy()
        
        return {
            'label': LABEL_MAP[predicted_class],
            'label_id': predicted_class,
            'confidence': confidence,
            'probabilities': {
                LABEL_MAP[i]: float(probs[i]) for i in range(self.num_labels)
            }
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        self._ensure_model_loaded()
        
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return self._predict_batch_impl(texts)
            except (RuntimeError, torch.cuda.OutOfMemoryError, OSError) as e:
                last_exception = e
                self._retry_count += 1
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise PredictionError(f"Batch prediction failed after {self.max_retries} attempts: {e}") from e
        
        raise PredictionError(f"Batch prediction failed: {last_exception}")

    def _predict_batch_impl(self, texts: List[str]) -> List[Dict[str, Any]]:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=DEFAULT_MAX_LENGTH,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

        results = []
        for i, text in enumerate(texts):
            probs = probabilities[i].cpu().numpy()
            predicted_class = np.argmax(probs)
            confidence = float(probs[predicted_class])
            
            results.append({
                'text': text,
                'label': LABEL_MAP[predicted_class],
                'label_id': int(predicted_class),
                'confidence': confidence,
                'probabilities': {
                    LABEL_MAP[j]: float(probs[j]) for j in range(self.num_labels)
                }
            })

        return results

    def get_fine_grained_scores(self, text: str) -> Dict[int, float]:
        self._ensure_model_loaded()
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=DEFAULT_MAX_LENGTH,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)

        return {i: float(probabilities[0][i]) for i in range(self.num_labels)}

    def get_stats(self) -> Dict[str, Any]:
        return {
            "inference_count": self._inference_count,
            "retry_count": self._retry_count,
            "device": str(self.device),
            "model_name": self.model_name,
            "is_loaded": self._is_loaded,
            "max_retries": self.max_retries,
        }
