import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import numpy as np


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': accuracy, 'f1': f1}


def train_model(
    model_name: str = "distilbert-base-uncased",
    train_file: str = "data/processed/train.csv",
    val_file: str = "data/processed/val.csv",
    output_dir: str = "models/checkpoints",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5
):
    from datasets import Dataset
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4
    )
    
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    if "label" not in train_df.columns:
        raise ValueError("Training CSV must include a 'label' column (int class index).")
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )

    train_dataset = train_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="models/logs",
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    
    return trainer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train radical content classifier")
    parser.add_argument('--train', default='data/processed/train.csv')
    parser.add_argument('--val', default='data/processed/val.csv')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    
    train_model(
        train_file=args.train,
        val_file=args.val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
