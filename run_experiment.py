"""
TH-GFD Main Experiment Runner
Run this script to train and evaluate all models.

Usage:
    python run_experiment.py --dataset customs --label_ratio 0.05
    python run_experiment.py --dataset customs --label_ratio 0.01 --baselines_only
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.graph_builder import TemporalHeteroGraphBuilder, HomogeneousGraphBuilder
from models.baselines import (
    XGBoostBaseline, LightGBMBaseline, MLPBaseline,
    GCNBaseline, GraphSAGEBaseline, GATBaseline,
    train_gnn_baseline
)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_customs_data(data_path: str = None) -> pd.DataFrame:
    """Load or create customs dataset"""
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} transactions from {data_path}")
    else:
        print("Creating synthetic customs data for testing...")
        df = create_synthetic_customs_data(n_samples=10000)
    
    return df


def create_synthetic_customs_data(n_samples: int = 10000) -> pd.DataFrame:
    """Create synthetic customs data for testing"""
    np.random.seed(42)
    
    n_importers = 500
    n_declarants = 100
    n_offices = 20
    n_countries = 50
    
    # Generate data
    data = {
        'sgd.id': [f'SGD{i+1}' for i in range(n_samples)],
        'sgd.date': pd.date_range('2013-01-01', periods=n_samples, freq='30min').strftime('%y-%m-%d'),
        'importer.id': [f'IMP{np.random.randint(1, n_importers+1):06d}' for _ in range(n_samples)],
        'declarant.id': [f'DEC{np.random.randint(1, n_declarants+1):04d}' for _ in range(n_samples)],
        'country': [f'CNTRY{np.random.randint(1, n_countries+1):03d}' for _ in range(n_samples)],
        'office.id': [f'OFFICE{np.random.randint(1, n_offices+1):02d}' for _ in range(n_samples)],
        'tariff.code': [np.random.randint(1000000000, 9999999999) for _ in range(n_samples)],
        'quantity': np.random.randint(1, 1000, n_samples),
        'gross.weight': np.random.lognormal(7, 1.5, n_samples),
        'fob.value': np.random.lognormal(9, 1.5, n_samples),
        'cif.value': np.random.lognormal(9.2, 1.5, n_samples),
        'total.taxes': np.random.lognormal(7, 1.5, n_samples),
    }
    
    # Generate fraud labels (~7.5% fraud rate)
    # Fraud more likely for: high value, certain importers, certain countries
    fraud_prob = np.zeros(n_samples)
    fraud_prob += 0.03  # Base rate
    
    # High-value transactions more likely to be fraud
    value_percentile = np.percentile(data['cif.value'], 90)
    fraud_prob[data['cif.value'] > value_percentile] += 0.1
    
    # Some importers are more likely to commit fraud
    risky_importers = set(np.random.choice(
        [f'IMP{i:06d}' for i in range(1, n_importers+1)], 
        size=int(n_importers * 0.1), replace=False
    ))
    for i, imp in enumerate(data['importer.id']):
        if imp in risky_importers:
            fraud_prob[i] += 0.15
    
    fraud_prob = np.clip(fraud_prob, 0, 1)
    data['illicit'] = np.random.binomial(1, fraud_prob)
    
    # Revenue (only for fraudulent transactions)
    data['revenue'] = np.where(
        data['illicit'] == 1,
        data['total.taxes'] * np.random.uniform(0.5, 2.0, n_samples),
        0
    )
    
    df = pd.DataFrame(data)
    print(f"Created synthetic data: {len(df)} transactions, {data['illicit'].sum()} illicit ({data['illicit'].mean()*100:.2f}%)")
    
    return df


def preprocess_data(df: pd.DataFrame) -> Tuple:
    """Preprocess customs data"""
    df = df.copy()
    
    # Parse date
    if 'sgd.date' in df.columns:
        df['date'] = pd.to_datetime(df['sgd.date'], format='%y-%m-%d', errors='coerce')
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
    
    # Feature engineering
    numerical_cols = ['quantity', 'gross.weight', 'fob.value', 'cif.value', 'total.taxes']
    
    if all(c in df.columns for c in ['cif.value', 'quantity']):
        df['unit_value'] = df['cif.value'] / (df['quantity'] + 1)
        numerical_cols.append('unit_value')
    
    if all(c in df.columns for c in ['cif.value', 'gross.weight']):
        df['value_per_kg'] = df['cif.value'] / (df['gross.weight'] + 1)
        numerical_cols.append('value_per_kg')
    
    if all(c in df.columns for c in ['total.taxes', 'cif.value']):
        df['tax_ratio'] = df['total.taxes'] / (df['cif.value'] + 1)
        numerical_cols.append('tax_ratio')
    
    if 'day_of_year' in df.columns:
        numerical_cols.extend(['day_of_year', 'month'])
    
    # Scale features
    scaler = StandardScaler()
    df[numerical_cols] = df[numerical_cols].fillna(0)
    X = scaler.fit_transform(df[numerical_cols])
    
    # Labels
    y = df['illicit'].values if 'illicit' in df.columns else np.zeros(len(df))
    
    # Revenue
    revenue = df['revenue'].values if 'revenue' in df.columns else np.zeros(len(df))
    
    return df, X, y, revenue, numerical_cols


def temporal_train_val_test_split(df: pd.DataFrame, 
                                   train_ratio: float = 0.7,
                                   val_ratio: float = 0.1):
    """Split data temporally"""
    if 'date' in df.columns:
        df_sorted = df.sort_values('date').reset_index(drop=True)
    else:
        df_sorted = df.reset_index(drop=True)
    
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_idx = np.arange(train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n)
    
    return train_idx, val_idx, test_idx


def create_label_mask(n_samples: int, train_idx: np.ndarray, label_ratio: float) -> np.ndarray:
    """Create mask for labeled samples (simulating limited inspection)"""
    n_train = len(train_idx)
    n_labeled = int(n_train * label_ratio)
    
    labeled_train_idx = np.random.choice(train_idx, size=n_labeled, replace=False)
    
    label_mask = np.zeros(n_samples, dtype=bool)
    label_mask[labeled_train_idx] = True
    
    return label_mask


def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                    revenue: np.ndarray = None) -> dict:
    """Compute evaluation metrics"""
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    metrics = {
        'AUC': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5,
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
    }
    
    # Precision@k% and Recall@k%
    for k in [1, 5]:
        top_k_idx = np.argsort(y_pred_proba)[-int(len(y_true) * k / 100):]
        precision_at_k = y_true[top_k_idx].mean()
        recall_at_k = y_true[top_k_idx].sum() / max(y_true.sum(), 1)
        
        metrics[f'Precision@{k}%'] = precision_at_k
        metrics[f'Recall@{k}%'] = recall_at_k
        
        if revenue is not None:
            revenue_at_k = revenue[top_k_idx].sum() / max(revenue.sum(), 1)
            metrics[f'Revenue@{k}%'] = revenue_at_k
    
    return metrics


def run_xgboost_experiment(X, y, train_idx, val_idx, test_idx, label_mask):
    """Run XGBoost baseline"""
    print("\n" + "="*50)
    print("Running XGBoost Baseline")
    print("="*50)
    
    # Get labeled training data
    labeled_train_idx = train_idx[label_mask[train_idx]]
    
    X_train = X[labeled_train_idx]
    y_train = y[labeled_train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    print(f"Training on {len(X_train)} labeled samples")
    
    model = XGBoostBaseline(
        n_estimators=100,
        max_depth=6,
        scale_pos_weight=sum(y_train == 0) / max(sum(y_train == 1), 1)
    )
    model.fit(X_train, y_train, X_val, y_val)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return y_pred_proba


def run_gnn_experiment(X, y, edge_index, train_idx, val_idx, test_idx, 
                       label_mask, model_class, model_name, device='cpu'):
    """Run GNN baseline experiment"""
    print("\n" + "="*50)
    print(f"Running {model_name} Baseline")
    print("="*50)
    
    from torch_geometric.data import Data
    
    # Create masks
    n = len(y)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    
    # Only use labeled training samples
    labeled_train_idx = train_idx[label_mask[train_idx]]
    train_mask[labeled_train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    print(f"Training on {train_mask.sum().item()} labeled samples")
    
    # Create PyG data object
    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long)
    )
    
    # Create model
    model = model_class(
        num_features=X.shape[1],
        hidden_dim=64,
        num_layers=2,
        dropout=0.3
    )
    
    # Train
    model, metrics = train_gnn_baseline(
        model, data, train_mask, val_mask, test_mask,
        epochs=200, lr=0.01, device=device
    )
    
    # Get predictions
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        y_pred_proba = torch.sigmoid(out[test_mask]).cpu().numpy()
    
    return y_pred_proba


def build_homogeneous_graph(df, entity_columns):
    """Build homogeneous graph for GNN baselines"""
    print("Building homogeneous graph...")
    
    n = len(df)
    edges = []
    
    for entity_col in entity_columns:
        if entity_col not in df.columns:
            continue
        
        # Group by entity
        groups = df.groupby(entity_col).indices
        
        for entity, indices in groups.items():
            if len(indices) < 2:
                continue
            
            # Sample to avoid too many edges
            if len(indices) > 20:
                indices = np.random.choice(indices, 20, replace=False)
            
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    edges.append([indices[i], indices[j]])
                    edges.append([indices[j], indices[i]])
    
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # Remove duplicates
        edge_index = torch.unique(edge_index, dim=1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    print(f"Graph: {n} nodes, {edge_index.shape[1]} edges")
    return edge_index


def main():
    parser = argparse.ArgumentParser(description='TH-GFD Experiment Runner')
    parser.add_argument('--dataset', type=str, default='customs', 
                        choices=['customs', 'amazon', 'elliptic'])
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to dataset CSV file')
    parser.add_argument('--label_ratio', type=float, default=0.05,
                        help='Fraction of labeled training data (default: 0.05)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--baselines_only', action='store_true',
                        help='Run only baseline models')
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    print("\n" + "="*50)
    print("Loading and Preprocessing Data")
    print("="*50)
    
    df = load_customs_data(args.data_path)
    df, X, y, revenue, feature_cols = preprocess_data(df)
    
    print(f"Total samples: {len(df)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Fraud rate: {y.mean()*100:.2f}%")
    
    # Split data
    train_idx, val_idx, test_idx = temporal_train_val_test_split(df)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Create label mask
    label_mask = create_label_mask(len(df), train_idx, args.label_ratio)
    print(f"Labeled training samples: {label_mask.sum()} ({args.label_ratio*100:.1f}%)")
    
    # Store results
    results = {}
    
    # =========================================================================
    # XGBoost Baseline
    # =========================================================================
    y_pred_xgb = run_xgboost_experiment(X, y, train_idx, val_idx, test_idx, label_mask)
    results['XGBoost'] = compute_metrics(y[test_idx], y_pred_xgb, revenue[test_idx])
    
    # =========================================================================
    # Build graph for GNN baselines
    # =========================================================================
    entity_columns = ['importer.id', 'declarant.id', 'office.id', 'country']
    edge_index = build_homogeneous_graph(df, entity_columns)
    edge_index = edge_index.to(device)
    
    # =========================================================================
    # GCN Baseline
    # =========================================================================
    y_pred_gcn = run_gnn_experiment(
        X, y, edge_index, train_idx, val_idx, test_idx, 
        label_mask, GCNBaseline, "GCN", device
    )
    results['GCN'] = compute_metrics(y[test_idx], y_pred_gcn, revenue[test_idx])
    
    # =========================================================================
    # GraphSAGE Baseline
    # =========================================================================
    y_pred_sage = run_gnn_experiment(
        X, y, edge_index, train_idx, val_idx, test_idx,
        label_mask, GraphSAGEBaseline, "GraphSAGE", device
    )
    results['GraphSAGE'] = compute_metrics(y[test_idx], y_pred_sage, revenue[test_idx])
    
    # =========================================================================
    # GAT Baseline
    # =========================================================================
    y_pred_gat = run_gnn_experiment(
        X, y, edge_index, train_idx, val_idx, test_idx,
        label_mask, GATBaseline, "GAT", device
    )
    results['GAT'] = compute_metrics(y[test_idx], y_pred_gat, revenue[test_idx])
    
    # =========================================================================
    # Print Results
    # =========================================================================
    print("\n" + "="*80)
    print(f"RESULTS SUMMARY (Label Ratio: {args.label_ratio*100:.1f}%)")
    print("="*80)
    
    # Create results table
    metrics_order = ['AUC', 'F1', 'Precision@1%', 'Recall@5%', 'Revenue@5%']
    
    print(f"\n{'Method':<15}", end='')
    for m in metrics_order:
        print(f"{m:>12}", end='')
    print()
    print("-"*75)
    
    for method, metrics in results.items():
        print(f"{method:<15}", end='')
        for m in metrics_order:
            val = metrics.get(m, 0)
            print(f"{val:>12.4f}", end='')
        print()
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(f'results_label{int(args.label_ratio*100)}pct.csv')
    print(f"\nResults saved to results_label{int(args.label_ratio*100)}pct.csv")
    
    return results


if __name__ == "__main__":
    from typing import Tuple
    main()
