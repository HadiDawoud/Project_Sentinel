import torch
import numpy as np
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path


class RadicalClassifier:
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 4,
        checkpoint_path: Optional[str] = None
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._load_model(checkpoint_path)
        
        self.labels = {
            0: "Non-Radical",
            1: "Mildly Radical",
            2: "Moderately Radical",
            3: "Highly Radical"
        }

    def _load_model(self, checkpoint_path: Optional[str] = None) -> None:
        valid_checkpoint = False
        if checkpoint_path and Path(checkpoint_path).exists():
            # Check if directory is not empty (at least config.json should be there)
            if any(Path(checkpoint_path).iterdir()):
                valid_checkpoint = True
        
        if valid_checkpoint:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint_path,
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

    def predict(self, text: str) -> Dict[str, any]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
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
            'label': self.labels[predicted_class],
            'label_id': predicted_class,
            'confidence': confidence,
            'probabilities': {
                self.labels[i]: float(probs[i]) for i in range(self.num_labels)
            }
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=256,
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
                'label': self.labels[predicted_class],
                'label_id': int(predicted_class),
                'confidence': confidence,
                'probabilities': {
                    self.labels[j]: float(probs[j]) for j in range(self.num_labels)
                }
            })

        return results

    def get_fine_grained_scores(self, text: str) -> Dict[int, float]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)

        return {i: float(probabilities[0][i]) for i in range(self.num_labels)}
