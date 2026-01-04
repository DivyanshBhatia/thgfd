"""
Complete Fraud Detection Benchmark Experiments
==============================================

This script runs comprehensive experiments on Amazon FDB datasets.

Setup:
------
1. Install dependencies:
   pip install xgboost scikit-learn pandas numpy
   pip install git+https://github.com/amazon-science/fraud-dataset-benchmark.git
   
2. Setup Kaggle CLI (required for FDB):
   - Create account at kaggle.com
   - Download kaggle.json from Account Settings
   - Place at ~/.kaggle/kaggle.json
   - Join IEEE-CIS competition: https://www.kaggle.com/c/ieee-fraud-detection

3. Run experiments:
   python fdb_experiments.py --dataset all
   python fdb_experiments.py --dataset ieeecis
   python fdb_experiments.py --dataset ccfraud

Metrics Computed:
-----------------
- AUC-ROC, Average Precision (AP), NDCG
- F1 Score (at 0.5 and best threshold)
- Precision@K, Recall@K, Lift@K (K = 1%, 5%, 10%)
"""

import numpy as np
import pandas as pd
import argparse
import warnings
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# =============================================================================
# COMPREHENSIVE METRICS
# =============================================================================

def compute_all_metrics(y_true, y_pred_proba, k_percentages=[1, 5, 10]):
    """
    Compute comprehensive fraud detection metrics.
    
    Returns dict with:
    - AUC, AP, NDCG
    - Best_F1, F1@0.5
    - P@K%, R@K%, Lift@K% for each K
    """
    metrics = {}
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    n_samples = len(y_true)
    n_positives = y_true.sum()
    fraud_rate = n_positives / n_samples if n_samples > 0 else 0
    
    # AUC-ROC
    try:
        metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
    except:
        metrics['AUC'] = 0.5
    
    # Average Precision
    try:
        metrics['AP'] = average_precision_score(y_true, y_pred_proba)
    except:
        metrics['AP'] = fraud_rate
    
    # Best F1 and F1@0.5
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        metrics['Best_F1'] = np.max(f1_scores)
    except:
        metrics['Best_F1'] = 0
    
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    metrics['F1@0.5'] = f1_score(y_true, y_pred_binary, zero_division=0)
    
    # Sort by predicted probability (descending)
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    # Precision@K, Recall@K, Lift@K
    for k_pct in k_percentages:
        k = max(1, int(n_samples * k_pct / 100))
        top_k_true = y_true_sorted[:k]
        
        p_at_k = top_k_true.sum() / k
        metrics[f'P@{k_pct}%'] = p_at_k
        
        r_at_k = top_k_true.sum() / max(n_positives, 1)
        metrics[f'R@{k_pct}%'] = r_at_k
        
        metrics[f'Lift@{k_pct}%'] = p_at_k / fraud_rate if fraud_rate > 0 else 0
    
    # NDCG
    dcg = np.sum(y_true_sorted / np.log2(np.arange(2, n_samples + 2)))
    ideal_sorted = np.sort(y_true)[::-1]
    idcg = np.sum(ideal_sorted / np.log2(np.arange(2, n_samples + 2)))
    metrics['NDCG'] = dcg / idcg if idcg > 0 else 0
    
    return metrics


def print_metrics(metrics, name):
    """Pretty print metrics."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print('='*60)
    print(f"  AUC:     {metrics['AUC']:.4f}")
    print(f"  AP:      {metrics['AP']:.4f}")
    print(f"  Best F1: {metrics['Best_F1']:.4f}")
    print(f"  NDCG:    {metrics['NDCG']:.4f}")
    print(f"  ─────────────────────────────")
    print(f"  P@1%:    {metrics['P@1%']:.4f}  (Lift: {metrics['Lift@1%']:.1f}x)")
    print(f"  P@5%:    {metrics['P@5%']:.4f}  (Lift: {metrics['Lift@5%']:.1f}x)")
    print(f"  P@10%:   {metrics['P@10%']:.4f} (Lift: {metrics['Lift@10%']:.1f}x)")
    print(f"  ─────────────────────────────")
    print(f"  R@1%:    {metrics['R@1%']:.4f}")
    print(f"  R@5%:    {metrics['R@5%']:.4f}")
    print(f"  R@10%:   {metrics['R@10%']:.4f}")


# =============================================================================
# FDB DATASET LOADER
# =============================================================================

FDB_DATASETS = {
    'ieeecis': {'name': 'IEEE-CIS Fraud Detection', 'graph_utility': 0.0236},
    'ccfraud': {'name': 'Credit Card Fraud', 'graph_utility': 0.0041},
    'fraudecom': {'name': 'Fraud Ecommerce', 'graph_utility': 0.0568},
    'sparknov': {'name': 'Simulated CC Transactions', 'graph_utility': 0.0045},
    'twitterbot': {'name': 'Twitter Bots', 'graph_utility': 0.0113},
    'malurl': {'name': 'Malicious URLs', 'graph_utility': 0.0015},
    'fakejob': {'name': 'Fake Job Postings', 'graph_utility': 0.0984},
    'vehicleloan': {'name': 'Vehicle Loan Default', 'graph_utility': 0.0194},
    'ipblock': {'name': 'IP Blocklist (CI-BadGuys)', 'graph_utility': 0.2895},
}


def load_fdb_dataset(key):
    """Load dataset from FDB."""
    try:
        from fdb.datasets import FraudDatasetBenchmark
        
        print(f"Loading {key} from FDB...")
        obj = FraudDatasetBenchmark(
            key=key,
            load_pre_downloaded=False,
            delete_downloaded=False
        )
        
        # Get train/test data
        train_df = obj.train.copy()
        test_df = obj.test.copy()
        test_labels = obj.test_labels.copy()
        
        # Identify label column
        label_col = 'EVENT_LABEL'
        
        # Identify columns to drop
        drop_cols = ['EVENT_LABEL', 'EVENT_TIMESTAMP', 'LABEL_TIMESTAMP', 
                     'EVENT_ID', 'ENTITY_ID', 'ENTITY_TYPE']
        
        # Prepare features
        feature_cols = [c for c in train_df.columns if c not in drop_cols]
        
        X_train = train_df[feature_cols].copy()
        y_train = train_df[label_col].values
        X_test = test_df[feature_cols].copy()
        y_test = test_labels[label_col].values
        
        # Encode categorical columns
        for col in X_train.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)])
            le.fit(combined)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
        
        # Fill NaN
        X_train = X_train.fillna(-999)
        X_test = X_test.fillna(-999)
        
        return X_train.values, y_train, X_test.values, y_test
        
    except ImportError:
        print("FDB not installed. Install with:")
        print("pip install git+https://github.com/amazon-science/fraud-dataset-benchmark.git")
        return None
    except Exception as e:
        print(f"Error loading {key}: {e}")
        return None


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier."""
    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost not installed. pip install xgboost")
        return None, None
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Class weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train, 
              eval_set=[(X_test_scaled, y_test)],
              verbose=False)
    
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    return y_pred_proba, model


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest classifier."""
    from sklearn.ensemble import RandomForestClassifier
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    return y_pred_proba, model


def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM classifier."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBM not installed. pip install lightgbm")
        return None, None
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)
    
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    return y_pred_proba, model


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(key, run_gnn=False):
    """Run full experiment on a dataset."""
    
    info = FDB_DATASETS.get(key, {'name': key, 'graph_utility': 0.0})
    
    print("\n" + "="*80)
    print(f"DATASET: {info['name']} ({key})")
    print(f"Graph Utility: {info['graph_utility']:.4f}")
    print("="*80)
    
    # Load data
    data = load_fdb_dataset(key)
    if data is None:
        return None
    
    X_train, y_train, X_test, y_test = data
    
    print(f"\nData loaded:")
    print(f"  Train: {len(y_train):,} samples, {y_train.sum():,} frauds ({y_train.mean()*100:.2f}%)")
    print(f"  Test:  {len(y_test):,} samples, {y_test.sum():,} frauds ({y_test.mean()*100:.2f}%)")
    print(f"  Features: {X_train.shape[1]}")
    
    results = {
        'dataset': key,
        'name': info['name'],
        'graph_utility': info['graph_utility'],
        'n_train': len(y_train),
        'n_test': len(y_test),
        'fraud_rate': y_test.mean()
    }
    
    # XGBoost
    print("\n" + "-"*40)
    print("Training XGBoost...")
    xgb_pred, _ = train_xgboost(X_train, y_train, X_test, y_test)
    if xgb_pred is not None:
        xgb_metrics = compute_all_metrics(y_test, xgb_pred)
        print_metrics(xgb_metrics, "XGBoost Results")
        results['xgb'] = xgb_metrics
    
    # Random Forest
    print("\n" + "-"*40)
    print("Training Random Forest...")
    rf_pred, _ = train_random_forest(X_train, y_train, X_test, y_test)
    rf_metrics = compute_all_metrics(y_test, rf_pred)
    print_metrics(rf_metrics, "Random Forest Results")
    results['rf'] = rf_metrics
    
    # LightGBM (optional)
    try:
        print("\n" + "-"*40)
        print("Training LightGBM...")
        lgb_pred, _ = train_lightgbm(X_train, y_train, X_test, y_test)
        if lgb_pred is not None:
            lgb_metrics = compute_all_metrics(y_test, lgb_pred)
            print_metrics(lgb_metrics, "LightGBM Results")
            results['lgb'] = lgb_metrics
    except:
        pass
    
    return results


def run_all_experiments():
    """Run experiments on all FDB datasets."""
    
    print("="*80)
    print("FDB FRAUD DETECTION BENCHMARK")
    print("="*80)
    print(f"Datasets: {len(FDB_DATASETS)}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = []
    
    for key in FDB_DATASETS.keys():
        try:
            result = run_experiment(key)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"Error with {key}: {e}")
    
    # Create summary
    if all_results:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        summary_data = []
        for r in all_results:
            row = {
                'Dataset': r['name'],
                'Graph_Utility': r['graph_utility'],
                'Fraud_Rate': r['fraud_rate'],
            }
            
            if 'xgb' in r:
                row['XGB_AUC'] = r['xgb']['AUC']
                row['XGB_AP'] = r['xgb']['AP']
                row['XGB_P@1%'] = r['xgb']['P@1%']
            
            if 'rf' in r:
                row['RF_AUC'] = r['rf']['AUC']
                row['RF_AP'] = r['rf']['AP']
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'fdb_results_{timestamp}.csv'
        summary_df.to_csv(filename, index=False)
        print(f"\nResults saved to: {filename}")
    
    return all_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='FDB Fraud Detection Experiments')
    parser.add_argument('--dataset', type=str, default='all',
                        help='Dataset key (or "all" for all datasets)')
    parser.add_argument('--gnn', action='store_true',
                        help='Include GNN experiments')
    
    args = parser.parse_args()
    
    if args.dataset == 'all':
        run_all_experiments()
    elif args.dataset in FDB_DATASETS:
        run_experiment(args.dataset, run_gnn=args.gnn)
    else:
        print(f"Unknown dataset: {args.dataset}")
        print(f"Available: {list(FDB_DATASETS.keys())}")


if __name__ == "__main__":
    main()
