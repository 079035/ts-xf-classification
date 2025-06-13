import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import argparse

def prepare_data_from_csv(input_path, output_dir, label_col='label', 
                         test_size=0.2, val_size=0.1, random_state=42):
    """
    Convert CSV data to Parquet format with train/val/test split.
    """
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    # Check class distribution
    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n{df[label_col].value_counts()}")
    print(f"Positive ratio: {(df[label_col] == 1).mean():.4f}")
    
    # Create temporal split (important for time series)
    n_samples = len(df)
    train_end = int(n_samples * (1 - test_size - val_size))
    val_end = int(n_samples * (1 - test_size))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    # Save as Parquet
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_parquet(os.path.join(output_dir, 'train.parquet'))
    val_df.to_parquet(os.path.join(output_dir, 'val.parquet'))
    test_df.to_parquet(os.path.join(output_dir, 'test.parquet'))
    
    print(f"\nData saved to {output_dir}")
    print(f"Train: {len(train_df)} samples, positive ratio: {(train_df[label_col] == 1).mean():.4f}")
    print(f"Val: {len(val_df)} samples, positive ratio: {(val_df[label_col] == 1).mean():.4f}")
    print(f"Test: {len(test_df)} samples, positive ratio: {(test_df[label_col] == 1).mean():.4f}")

def create_synthetic_imbalanced_data(output_dir, n_samples=10000, n_features=20, 
                                   seq_len=96, pos_ratio=0.01):
    """
    Create synthetic imbalanced time series data for testing.
    """
    print(f"Creating synthetic imbalanced time series data...")
    
    # Generate features
    np.random.seed(42)
    
    # Create time series features with some temporal patterns
    time = np.arange(n_samples)
    features = []
    
    for i in range(n_features):
        # Mix of different patterns
        if i < 5:  # Trend components
            feature = 0.1 * time + np.random.randn(n_samples) * 10
        elif i < 10:  # Seasonal components
            feature = 10 * np.sin(2 * np.pi * time / 100 + np.random.rand() * 2 * np.pi) + \
                     np.random.randn(n_samples) * 2
        else:  # Random walk
            feature = np.cumsum(np.random.randn(n_samples)) + np.random.randn(n_samples) * 5
        
        features.append(feature)
    
    features = np.array(features).T
    
    # Create imbalanced labels with some pattern
    # Positive examples are more likely when certain features have extreme values
    labels = np.zeros(n_samples, dtype=int)
    
    # Create a score based on feature patterns
    score = np.abs(features[:, 0]) + np.abs(features[:, 5])  # Use specific features
    threshold = np.percentile(score, 100 * (1 - pos_ratio))
    labels[score > threshold] = 1
    
    # Add some random positive examples
    n_additional = max(0, int(n_samples * pos_ratio) - labels.sum())
    if n_additional > 0:
        negative_indices = np.where(labels == 0)[0]
        random_positives = np.random.choice(negative_indices, n_additional, replace=False)
        labels[random_positives] = 1
    
    # Create DataFrame
    df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(n_features)])
    df['label'] = labels
    
    # Create temporal split
    train_end = int(0.7 * n_samples)
    val_end = int(0.85 * n_samples)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    # Save as Parquet
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_parquet(os.path.join(output_dir, 'train.parquet'))
    val_df.to_parquet(os.path.join(output_dir, 'val.parquet'))
    test_df.to_parquet(os.path.join(output_dir, 'test.parquet'))
    
    print(f"\nSynthetic data saved to {output_dir}")
    print(f"Train: {len(train_df)} samples, positive ratio: {(train_df['label'] == 1).mean():.4f}")
    print(f"Val: {len(val_df)} samples, positive ratio: {(val_df['label'] == 1).mean():.4f}")
    print(f"Test: {len(test_df)} samples, positive ratio: {(test_df['label'] == 1).mean():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare imbalanced data for classification')
    parser.add_argument('--input_path', type=str, help='Input CSV file path')
    parser.add_argument('--output_dir', type=str, default='./data/', help='Output directory')
    parser.add_argument('--label_col', type=str, default='label', help='Label column name')
    parser.add_argument('--synthetic', action='store_true', help='Create synthetic data')
    parser.add_argument('--n_samples', type=int, default=10000, help='Number of samples for synthetic data')
    parser.add_argument('--pos_ratio', type=float, default=0.01, help='Positive class ratio')
    
    args = parser.parse_args()
    
    if args.synthetic:
        create_synthetic_imbalanced_data(
            args.output_dir, 
            n_samples=args.n_samples,
            pos_ratio=args.pos_ratio
        )
    elif args.input_path:
        prepare_data_from_csv(
            args.input_path,
            args.output_dir,
            label_col=args.label_col
        )
    else:
        print("Please provide --input_path or use --synthetic flag")
