import torch
import pandas as pd
import argparse
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from models.metrics import compute_metrics, get_classification_report


def evaluate_model(
    model_path: str,
    test_file: str,
    batch_size: int = 16
):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    test_df = pd.read_csv(test_file)
    texts = test_df['text'].tolist()
    true_labels = test_df['label'].tolist()
    
    predictions = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(preds.cpu().tolist())
    
    metrics = compute_metrics(true_labels, predictions)
    labels = ["Non-Radical", "Mildly Radical", "Moderately Radical", "Highly Radical"]
    report = get_classification_report(true_labels, predictions, labels)
    
    return metrics, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument('--model-path', required=True, help="Path to model checkpoint")
    parser.add_argument('--test-file', required=True, help="Test CSV file")
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    
    metrics, report = evaluate_model(args.model_path, args.test_file, args.batch_size)
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nClassification Report:")
    print(report)
