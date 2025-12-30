"""
TH-GFD Quick Start Script
Run this first to test the complete pipeline.

Usage:
    python quick_start.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Ensure we can import from project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    print("="*60)
    print("TH-GFD Quick Start - Testing Complete Pipeline")
    print("="*60)
    
    # Check dependencies
    print("\n[1/7] Checking dependencies...")
    try:
        import torch_geometric
        print(f"  ✓ PyTorch: {torch.__version__}")
        print(f"  ✓ PyTorch Geometric: {torch_geometric.__version__}")
    except ImportError:
        print("  ✗ PyTorch Geometric not found!")
        print("  Install with: pip install torch-geometric")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  ✓ Device: {device}")
    
    # Create synthetic data
    print("\n[2/7] Creating synthetic customs data...")
    np.random.seed(42)
    n_samples = 5000  # Small for quick testing
    
    df = pd.DataFrame({
        'sgd.date': pd.date_range('2013-01-01', periods=n_samples, freq='30min').strftime('%y-%m-%d'),
        'importer.id': [f'IMP{np.random.randint(1, 101):04d}' for _ in range(n_samples)],
        'declarant.id': [f'DEC{np.random.randint(1, 21):02d}' for _ in range(n_samples)],
        'country': [f'CTY{np.random.randint(1, 11):02d}' for _ in range(n_samples)],
        'office.id': [f'OFF{np.random.randint(1, 6):02d}' for _ in range(n_samples)],
        'quantity': np.random.randint(1, 500, n_samples),
        'gross.weight': np.random.lognormal(6, 1.5, n_samples),
        'fob.value': np.random.lognormal(8, 1.5, n_samples),
        'cif.value': np.random.lognormal(8.2, 1.5, n_samples),
        'total.taxes': np.random.lognormal(6, 1.5, n_samples),
    })
    
    # Generate realistic fraud labels
    fraud_prob = np.full(n_samples, 0.03)
    high_value = df['cif.value'] > np.percentile(df['cif.value'], 85)
    fraud_prob[high_value] += 0.1
    
    risky_imps = np.random.choice(df['importer.id'].unique(), size=10, replace=False)
    for imp in risky_imps:
        fraud_prob[df['importer.id'] == imp] += 0.15
    
    df['illicit'] = np.random.binomial(1, np.clip(fraud_prob, 0, 1))
    df['revenue'] = np.where(df['illicit'] == 1, 
                             df['total.taxes'] * np.random.uniform(0.5, 2, n_samples), 
                             0)
    
    print(f"  ✓ Created {n_samples} transactions")
    print(f"  ✓ Fraud rate: {df['illicit'].mean()*100:.2f}%")
    
    # Build graph
    print("\n[3/7] Building temporal-heterogeneous graph...")
    from data.graph_builder import TemporalHeteroGraphBuilder
    
    # Prepare features
    feature_cols = ['quantity', 'gross.weight', 'fob.value', 'cif.value', 'total.taxes']
    df['unit_value'] = df['cif.value'] / (df['quantity'] + 1)
    df['value_per_kg'] = df['cif.value'] / (df['gross.weight'] + 1)
    feature_cols.extend(['unit_value', 'value_per_kg'])
    
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols].fillna(0))
    
    builder = TemporalHeteroGraphBuilder(
        entity_columns=['importer.id', 'declarant.id', 'office.id', 'country'],
        numerical_columns=feature_cols
    )
    
    data = builder.build(df, label_col='illicit', revenue_col='revenue')
    
    print(f"  ✓ Node types: {data.node_types}")
    print(f"  ✓ Edge types: {len(data.edge_types)}")
    print(f"  ✓ Transaction features: {data['transaction'].x.shape}")
    
    # Create splits
    print("\n[4/7] Creating train/val/test splits...")
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    
    train_idx = np.arange(train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n)
    
    # Label mask (5% labeled)
    label_ratio = 0.05
    n_labeled = int(len(train_idx) * label_ratio)
    labeled_idx = np.random.choice(train_idx, size=n_labeled, replace=False)
    label_mask = np.zeros(n, dtype=bool)
    label_mask[labeled_idx] = True
    
    print(f"  ✓ Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"  ✓ Labeled training samples: {n_labeled} ({label_ratio*100}%)")
    
    # Run baselines
    print("\n[5/7] Running XGBoost baseline...")
    from models.baselines import XGBoostBaseline
    
    X = df[feature_cols].values
    y = df['illicit'].values
    
    xgb_model = XGBoostBaseline(n_estimators=50, max_depth=4)
    xgb_model.fit(X[labeled_idx], y[labeled_idx], X[val_idx], y[val_idx])
    xgb_pred = xgb_model.predict_proba(X[test_idx])[:, 1]
    
    xgb_auc = roc_auc_score(y[test_idx], xgb_pred)
    print(f"  ✓ XGBoost AUC: {xgb_auc:.4f}")
    
    # Run GraphSAGE baseline
    print("\n[6/7] Running GraphSAGE baseline...")
    from data.graph_builder import HomogeneousGraphBuilder
    from models.baselines import GraphSAGEBaseline, train_gnn_baseline
    from torch_geometric.data import Data
    
    homo_builder = HomogeneousGraphBuilder(
        entity_columns=['importer.id', 'declarant.id', 'office.id', 'country'],
        numerical_columns=feature_cols
    )
    homo_data = homo_builder.build(df, label_col='illicit')
    
    # Create masks
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[labeled_idx] = True  # Only labeled samples
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    sage_model = GraphSAGEBaseline(
        num_features=len(feature_cols),
        hidden_dim=64,
        num_layers=2
    )
    
    sage_model, sage_metrics = train_gnn_baseline(
        sage_model, homo_data, train_mask, val_mask, test_mask,
        epochs=100, lr=0.01, device=device
    )
    
    print(f"  ✓ GraphSAGE AUC: {sage_metrics['auc']:.4f}")
    
    # Run TH-GFD
    print("\n[7/7] Running TH-GFD (simplified version)...")
    from models.thgfd import SimplifiedTHGFD
    
    # Prepare data for simplified model
    x_dict = {node_type: data[node_type].x for node_type in data.node_types}
    edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
    
    thgfd_model = SimplifiedTHGFD(
        num_features=len(feature_cols),
        hidden_dim=64,
        num_layers=2,
        edge_types=list(data.edge_types)
    ).to(device)
    
    optimizer = torch.optim.Adam(thgfd_model.parameters(), lr=0.01)
    
    # Move data to device
    x_dict = {k: v.to(device) for k, v in x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}
    y = data['transaction'].y.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    
    # Training loop
    best_val_auc = 0
    best_state = None
    
    for epoch in range(100):
        thgfd_model.train()
        optimizer.zero_grad()
        
        fraud_logits, _ = thgfd_model(x_dict, edge_index_dict)
        
        # Loss only on labeled data
        pos_weight = (train_mask.sum() - y[train_mask].sum()) / (y[train_mask].sum() + 1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            fraud_logits[train_mask],
            y[train_mask].float(),
            pos_weight=pos_weight.clamp(max=20)
        )
        
        loss.backward()
        optimizer.step()
        
        # Validation
        thgfd_model.eval()
        with torch.no_grad():
            fraud_logits, _ = thgfd_model(x_dict, edge_index_dict)
            val_probs = torch.sigmoid(fraud_logits[val_mask]).cpu().numpy()
            val_labels = y[val_mask].cpu().numpy()
            
            if len(np.unique(val_labels)) > 1:
                val_auc = roc_auc_score(val_labels, val_probs)
            else:
                val_auc = 0.5
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in thgfd_model.state_dict().items()}
    
    # Evaluate
    thgfd_model.load_state_dict(best_state)
    thgfd_model.to(device)
    thgfd_model.eval()
    
    with torch.no_grad():
        fraud_logits, _ = thgfd_model(x_dict, edge_index_dict)
        test_probs = torch.sigmoid(fraud_logits[test_mask]).cpu().numpy()
        test_labels = y[test_mask].cpu().numpy()
        
        thgfd_auc = roc_auc_score(test_labels, test_probs)
    
    print(f"  ✓ TH-GFD AUC: {thgfd_auc:.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\n{'Method':<20} {'Test AUC':>12}")
    print("-"*32)
    print(f"{'XGBoost':<20} {xgb_auc:>12.4f}")
    print(f"{'GraphSAGE':<20} {sage_metrics['auc']:>12.4f}")
    print(f"{'TH-GFD':<20} {thgfd_auc:>12.4f}")
    print("-"*32)
    
    print("\n✓ Quick start completed successfully!")
    print("\nNext steps:")
    print("  1. Run full experiments: python run_experiment.py")
    print("  2. Try different label ratios: python run_experiment.py --label_ratio 0.01")
    print("  3. Use your own data: python run_experiment.py --data_path /path/to/customs.csv")
    
    return {
        'xgboost_auc': xgb_auc,
        'graphsage_auc': sage_metrics['auc'],
        'thgfd_auc': thgfd_auc
    }


if __name__ == "__main__":
    main()
