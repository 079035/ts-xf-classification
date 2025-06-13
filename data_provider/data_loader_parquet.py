import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from sklearn.preprocessing import StandardScaler

class ParquetTimeSeriesDataset(Dataset):
    """
    Dataset class for loading time series data from Parquet files.
    Designed for imbalanced classification tasks.
    """
    
    def __init__(self, root_path, data_path, flag='train', 
                 seq_len=96, label_col='label', 
                 scale=True, timeenc=0, freq='h'):
        """
        Args:
            root_path: Root directory containing data
            data_path: Specific parquet file name or pattern
            flag: 'train', 'val', or 'test'
            seq_len: Sequence length for time series
            label_col: Name of the label column
            scale: Whether to scale features
            timeenc: Time encoding type (0 or 1)
            freq: Frequency of time series
        """
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.seq_len = seq_len
        self.label_col = label_col
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        
        self.__read_data__()
    
    def __read_data__(self):
        """
        Read and preprocess Parquet data files.
        """
        # Construct file path
        if self.flag == 'train':
            file_name = 'train.parquet'
        elif self.flag == 'val':
            file_name = 'val.parquet'
        else:
            file_name = 'test.parquet'
        
        file_path = os.path.join(self.root_path, file_name)
        
        # Check if specific file exists, otherwise try data_path
        if not os.path.exists(file_path):
            file_path = os.path.join(self.root_path, self.data_path)
        
        print(f"Loading data from: {file_path}")
        
        # Read parquet file
        df_raw = pd.read_parquet(file_path)
        
        # Extract labels
        if self.label_col in df_raw.columns:
            self.labels = df_raw[self.label_col].values
            # Remove label column from features
            df_features = df_raw.drop(columns=[self.label_col])
        else:
            raise ValueError(f"Label column '{self.label_col}' not found in data")
        
        # Get feature columns
        self.feature_names = df_features.columns.tolist()
        self.enc_in = len(self.feature_names)
        
        # Convert to numpy array
        data = df_features.values
        
        # Scale features if requested
        if self.scale:
            self.scaler = StandardScaler()
            if self.flag == 'train':
                self.scaler.fit(data)
            data = self.scaler.transform(data)
        
        # Create sequences
        self.data_x = []
        self.data_y = []
        
        for i in range(len(data) - self.seq_len + 1):
            sequence = data[i:i + self.seq_len]
            label = self.labels[i + self.seq_len - 1]  # Use label at end of sequence
            
            self.data_x.append(sequence)
            self.data_y.append(label)
        
        self.data_x = np.array(self.data_x)
        self.data_y = np.array(self.data_y)
        
        # Calculate class statistics
        unique_classes, class_counts = np.unique(self.data_y, return_counts=True)
        self.num_class = len(unique_classes)
        self.class_names = [f"Class_{i}" for i in unique_classes]
        
        print(f"Dataset {self.flag} - Shape: {self.data_x.shape}")
        print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
        print(f"Positive ratio: {class_counts[1] / len(self.data_y):.4f}")
    
    def __getitem__(self, index):
        """
        Get a single sample.
        
        Returns:
            sequence: Time series sequence
            label: Class label
            seq_mask: Padding mask (all ones for this implementation)
        """
        sequence = self.data_x[index]
        label = self.data_y[index]
        seq_mask = np.ones(self.seq_len)  # No padding in this implementation
        
        return sequence, label, seq_mask
    
    def __len__(self):
        return len(self.data_x)
    
    @property
    def max_seq_len(self):
        return self.seq_len
    
    @property
    def feature_df(self):
        # Return a dummy DataFrame with feature info for compatibility
        return pd.DataFrame(columns=self.feature_names)
