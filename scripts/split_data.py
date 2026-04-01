import pandas as pd
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_data(
    input_file: str,
    output_dir: str = "data/processed",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
):
    df = pd.read_csv(input_file)
    
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")
    
    train_df, temp_df = train_test_split(
        df, 
        test_size=(1 - train_ratio),
        random_state=random_seed,
        stratify=df['label']
    )
    
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_adjusted),
        random_state=random_seed,
        stratify=temp_df['label']
    )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
    
    print(f"Split complete:")
    print(f"  Train: {len(train_df)} samples ({train_ratio:.0%})")
    print(f"  Val:   {len(val_df)} samples ({val_ratio:.0%})")
    print(f"  Test:  {len(test_df)} samples ({test_ratio:.0%})")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument('--input', required=True, help="Input CSV file")
    parser.add_argument('--output-dir', default='data/processed', help="Output directory")
    parser.add_argument('--train-ratio', type=float, default=0.8, help="Training set ratio")
    parser.add_argument('--val-ratio', type=float, default=0.1, help="Validation set ratio")
    parser.add_argument('--test-ratio', type=float, default=0.1, help="Test set ratio")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    split_data(
        args.input,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
