"""
Data loading utilities for TH-GFD
Supports: Customs, Amazon, Elliptic++ datasets
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch


class CustomsDataLoader:
    """Load and preprocess Customs dataset"""
    
    def __init__(self, data_path: str, config):
        self.data_path = data_path
        self.config = config
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load(self) -> pd.DataFrame:
        """Load customs data from CSV"""
        # Try different file names
        possible_names = ['customs.csv', 'synthetic.csv', 'data.csv']
        
        for name in possible_names:
            filepath = os.path.join(self.data_path, name)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                print(f"Loaded {len(df)} transactions from {filepath}")
                return df
        
        raise FileNotFoundError(f"No customs data found in {self.data_path}")
    
    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Preprocess customs data"""
        df = df.copy()
        
        # Parse date if present
        if 'sgd.date' in df.columns:
            df['date'] = pd.to_datetime(df['sgd.date'], format='%y-%m-%d', errors='coerce')
            df['day_of_year'] = df['date'].dt.dayofyear
            df['month'] = df['date'].dt.month
            df['week'] = df['date'].dt.isocalendar().week
        
        # Encode categorical columns
        categorical_cols = self.config.graph.entity_columns
        for col in categorical_cols:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                    df[col].astype(str).fillna('UNKNOWN')
                )
        
        # Create derived features (following DATE paper)
        if all(c in df.columns for c in ['cif.value', 'quantity']):
            df['unit_value'] = df['cif.value'] / (df['quantity'] + 1)
        if all(c in df.columns for c in ['cif.value', 'gross.weight']):
            df['value_per_kg'] = df['cif.value'] / (df['gross.weight'] + 1)
        if all(c in df.columns for c in ['total.taxes', 'cif.value']):
            df['tax_ratio'] = df['total.taxes'] / (df['cif.value'] + 1)
        if all(c in df.columns for c in ['fob.value', 'cif.value']):
            df['fob_cif_ratio'] = df['fob.value'] / (df['cif.value'] + 1)
        
        # Scale numerical features
        numerical_cols = self.config.graph.numerical_columns
        extra_numerical = ['unit_value', 'value_per_kg', 'tax_ratio', 'fob_cif_ratio', 
                          'day_of_year', 'month', 'week']
        numerical_cols = [c for c in numerical_cols + extra_numerical if c in df.columns]
        
        if numerical_cols:
            df[numerical_cols] = df[numerical_cols].fillna(0)
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        # Store metadata
        metadata = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'numerical_cols': numerical_cols,
            'categorical_cols': categorical_cols,
            'n_samples': len(df)
        }
        
        return df, metadata
    
    def temporal_split(self, df: pd.DataFrame, 
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data temporally (train on past, test on future)"""
        if 'date' not in df.columns:
            print("Warning: No date column, using random split")
            return self.random_split(df, train_ratio, val_ratio)
        
        df_sorted = df.sort_values('date').reset_index(drop=True)
        n = len(df_sorted)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df_sorted.iloc[:train_end]
        val_df = df_sorted.iloc[train_end:val_end]
        test_df = df_sorted.iloc[val_end:]
        
        print(f"Temporal split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df
    
    def random_split(self, df: pd.DataFrame,
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Random split"""
        train_val, test_df = train_test_split(df, test_size=1-train_ratio-val_ratio, 
                                              random_state=self.config.data.random_seed)
        train_df, val_df = train_test_split(train_val, test_size=val_ratio/(train_ratio+val_ratio),
                                            random_state=self.config.data.random_seed)
        
        print(f"Random split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df
    
    def mask_labels(self, df: pd.DataFrame, label_ratio: float) -> np.ndarray:
        """
        Mask labels to simulate limited inspection rate.
        Returns mask where True = labeled, False = unlabeled
        """
        n = len(df)
        n_labeled = int(n * label_ratio)
        
        # Randomly select which samples are labeled
        labeled_indices = np.random.choice(n, size=n_labeled, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[labeled_indices] = True
        
        print(f"Label masking: {mask.sum()}/{n} samples labeled ({label_ratio*100:.1f}%)")
        return mask


class AmazonDataLoader:
    """Load Amazon fraud dataset"""
    
    def __init__(self, data_path: str, config):
        self.data_path = data_path
        self.config = config
        
    def load(self):
        """
        Load Amazon dataset from PyTorch Geometric or local files
        Expected format: node features, edge indices for 3 relation types, labels
        """
        try:
            from torch_geometric.datasets import Amazon
            # Note: The actual Amazon fraud dataset might need custom loading
            # This is a placeholder - use the actual dataset source
            pass
        except ImportError:
            pass
        
        # Try loading from local files
        features_path = os.path.join(self.data_path, 'amazon_features.npy')
        labels_path = os.path.join(self.data_path, 'amazon_labels.npy')
        
        if os.path.exists(features_path):
            features = np.load(features_path)
            labels = np.load(labels_path)
            
            # Load edge indices for each relation type
            edges = {}
            for rel in ['u_p_u', 'u_s_u', 'u_v_u']:
                edge_path = os.path.join(self.data_path, f'amazon_edges_{rel}.npy')
                if os.path.exists(edge_path):
                    edges[rel] = np.load(edge_path)
            
            return {
                'features': features,
                'labels': labels,
                'edges': edges
            }
        
        raise FileNotFoundError(f"Amazon data not found in {self.data_path}")


class EllipticDataLoader:
    """Load Elliptic++ dataset"""
    
    def __init__(self, data_path: str, config):
        self.data_path = data_path
        self.config = config
        
    def load(self):
        """
        Load Elliptic dataset
        Expected files: elliptic_txs_features.csv, elliptic_txs_edgelist.csv, elliptic_txs_classes.csv
        """
        # Features
        features_path = os.path.join(self.data_path, 'elliptic_txs_features.csv')
        if os.path.exists(features_path):
            features_df = pd.read_csv(features_path, header=None)
            # First column is txId, second is time step, rest are features
            tx_ids = features_df.iloc[:, 0].values
            time_steps = features_df.iloc[:, 1].values
            features = features_df.iloc[:, 2:].values
        else:
            raise FileNotFoundError(f"Elliptic features not found at {features_path}")
        
        # Edges
        edges_path = os.path.join(self.data_path, 'elliptic_txs_edgelist.csv')
        if os.path.exists(edges_path):
            edges_df = pd.read_csv(edges_path)
            edges = edges_df.values
        else:
            edges = None
        
        # Labels
        labels_path = os.path.join(self.data_path, 'elliptic_txs_classes.csv')
        if os.path.exists(labels_path):
            labels_df = pd.read_csv(labels_path)
            # Map: '1' -> illicit (1), '2' -> licit (0), 'unknown' -> -1
            label_map = {'1': 1, '2': 0, 'unknown': -1}
            labels = labels_df.iloc[:, 1].map(label_map).values
        else:
            labels = None
        
        return {
            'tx_ids': tx_ids,
            'time_steps': time_steps,
            'features': features,
            'edges': edges,
            'labels': labels
        }


def get_data_loader(dataset: str, data_path: str, config):
    """Factory function to get appropriate data loader"""
    loaders = {
        'customs': CustomsDataLoader,
        'amazon': AmazonDataLoader,
        'elliptic': EllipticDataLoader
    }
    
    if dataset not in loaders:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(loaders.keys())}")
    
    return loaders[dataset](data_path, config)


# Utility function to create sample customs data for testing
def create_sample_customs_data(n_samples: int = 1000, save_path: str = None) -> pd.DataFrame:
    """Create synthetic customs data for testing"""
    np.random.seed(42)
    
    n_importers = 100
    n_declarants = 50
    n_offices = 10
    n_countries = 20
    
    data = {
        'sgd.id': [f'SGD{i+1}' for i in range(n_samples)],
        'sgd.date': pd.date_range('2013-01-01', periods=n_samples, freq='H').strftime('%y-%m-%d'),
        'importer.id': [f'IMP{np.random.randint(1, n_importers+1):06d}' for _ in range(n_samples)],
        'declarant.id': [f'DEC{np.random.randint(1, n_declarants+1):04d}' for _ in range(n_samples)],
        'country': [f'CNTRY{np.random.randint(1, n_countries+1):03d}' for _ in range(n_samples)],
        'office.id': [f'OFFICE{np.random.randint(1, n_offices+1):02d}' for _ in range(n_samples)],
        'tariff.code': [np.random.randint(1000000000, 9999999999) for _ in range(n_samples)],
        'quantity': np.random.randint(1, 1000, n_samples),
        'gross.weight': np.random.uniform(100, 100000, n_samples),
        'fob.value': np.random.uniform(1000, 500000, n_samples),
        'cif.value': np.random.uniform(1000, 600000, n_samples),
        'total.taxes': np.random.uniform(100, 50000, n_samples),
        'illicit': np.random.binomial(1, 0.075, n_samples),  # ~7.5% fraud rate
        'revenue': np.zeros(n_samples)
    }
    
    # Revenue only for illicit transactions
    illicit_mask = data['illicit'] == 1
    data['revenue'][illicit_mask] = np.random.uniform(500, 5000, illicit_mask.sum())
    
    df = pd.DataFrame(data)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved sample data to {save_path}")
    
    return df
