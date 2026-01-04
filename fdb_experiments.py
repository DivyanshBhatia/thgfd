"""
Complete Fraud Detection Benchmark Experiments
==============================================

This script runs comprehensive experiments on fraud detection datasets.

Setup:
------
1. Install dependencies:
   pip install xgboost scikit-learn pandas numpy lightgbm
   
2. Run experiments:
   python fdb_experiments.py --dataset all
   python fdb_experiments.py --dataset ipblock
   python fdb_experiments.py --dataset credit_card

Metrics Computed:
-----------------
- AUC-ROC, Average Precision (AP), NDCG
- F1 Score (at 0.5 and best threshold)
- Precision@K, Recall@K, Lift@K (K = 1%, 5%, 10%)
"""

import os
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
# DATASET CONFIGURATION
# =============================================================================

# All available datasets with their configuration
# All loaders from analyze_datasets.py are mapped here
FDB_DATASETS = {
    # Custom loaders from analyze_datasets.py (primary loaders)
    'elliptic': {'name': 'Elliptic Bitcoin', 'graph_utility': 0.15, 'source': 'custom', 'loader': 'load_elliptic'},
    'amazon': {'name': 'Amazon Fraud', 'graph_utility': 0.12, 'source': 'custom', 'loader': 'load_amazon'},
    'yelp': {'name': 'Yelp Fraud', 'graph_utility': 0.10, 'source': 'custom', 'loader': 'load_yelp'},
    'bitcoin_otc': {'name': 'Bitcoin-OTC', 'graph_utility': 0.08, 'source': 'custom', 'loader': 'load_bitcoin_otc'},
    'bitcoin_alpha': {'name': 'Bitcoin-Alpha', 'graph_utility': 0.08, 'source': 'custom', 'loader': 'load_bitcoin_alpha'},
    'ieee_cis': {'name': 'IEEE-CIS Fraud Detection', 'graph_utility': 0.0236, 'source': 'custom', 'loader': 'load_ieee_cis'},
    'customs': {'name': 'Customs Fraud', 'graph_utility': 0.11, 'source': 'custom', 'loader': 'load_customs'},
    'credit_card': {'name': 'Credit Card (Kaggle)', 'graph_utility': 0.05, 'source': 'custom', 'loader': 'load_credit_card'},
    'ecommerce': {'name': 'Ecommerce Fraud', 'graph_utility': 0.06, 'source': 'custom', 'loader': 'load_ecommerce'},
    'cc_transactions': {'name': 'CC Transactions Fraud', 'graph_utility': 0.07, 'source': 'custom', 'loader': 'load_cc_transactions'},
    'twitter_bots': {'name': 'Twitter Bots', 'graph_utility': 0.09, 'source': 'custom', 'loader': 'load_twitter_bots'},
    'malicious_urls': {'name': 'Malicious URLs', 'graph_utility': 0.04, 'source': 'custom', 'loader': 'load_malicious_urls'},
    'vehicle_loan': {'name': 'Vehicle Loan Default', 'graph_utility': 0.05, 'source': 'custom', 'loader': 'load_vehicle_loan_default'},
    'fake_job': {'name': 'Fake Job Postings', 'graph_utility': 0.10, 'source': 'custom', 'loader': 'load_fake_job_postings'},
    'ipblock': {'name': 'IP Blocklist (CI-BadGuys)', 'graph_utility': 0.2895, 'source': 'custom', 'loader': 'load_ci_badguys'},
    
    # From FDB (Amazon Fraud Detection Benchmark) - alternative loaders if FDB is installed
    'ieeecis_fdb': {'name': 'IEEE-CIS (FDB)', 'graph_utility': 0.0236, 'source': 'fdb'},
    'ccfraud_fdb': {'name': 'Credit Card Fraud (FDB)', 'graph_utility': 0.0041, 'source': 'fdb'},
    'fraudecom_fdb': {'name': 'Fraud Ecommerce (FDB)', 'graph_utility': 0.0568, 'source': 'fdb'},
    'sparknov_fdb': {'name': 'Simulated CC Transactions (FDB)', 'graph_utility': 0.0045, 'source': 'fdb'},
    'twitterbot_fdb': {'name': 'Twitter Bots (FDB)', 'graph_utility': 0.0113, 'source': 'fdb'},
    'malurl_fdb': {'name': 'Malicious URLs (FDB)', 'graph_utility': 0.0015, 'source': 'fdb'},
    'fakejob_fdb': {'name': 'Fake Job Postings (FDB)', 'graph_utility': 0.0984, 'source': 'fdb'},
    'vehicleloan_fdb': {'name': 'Vehicle Loan Default (FDB)', 'graph_utility': 0.0194, 'source': 'fdb'},
}


# =============================================================================
# DATASET LOADER
# =============================================================================

def load_fdb_dataset(key):
    """Load dataset from FDB or custom loaders."""
    
    info = FDB_DATASETS.get(key)
    if info is None:
        print(f"Unknown dataset: {key}")
        return None
    
    source = info.get('source', 'fdb')
    
    # Handle custom loaders from analyze_datasets.py
    if source == 'custom':
        try:
            loader_name = info.get('loader')
            
            if loader_name == 'load_ci_badguys':
                from analyze_datasets import load_ci_badguys
                print(f"Loading {key} using custom loader (CI-BadGuys)...")
                data = load_ci_badguys()
                
            elif loader_name == 'load_elliptic':
                from analyze_datasets import load_elliptic
                print(f"Loading {key} using custom loader (Elliptic)...")
                data = load_elliptic()
                
            elif loader_name == 'load_amazon':
                from analyze_datasets import load_amazon
                print(f"Loading {key} using custom loader (Amazon)...")
                data = load_amazon()
                
            elif loader_name == 'load_yelp':
                from analyze_datasets import load_yelp
                print(f"Loading {key} using custom loader (Yelp)...")
                data = load_yelp()
                
            elif loader_name == 'load_bitcoin_otc':
                from analyze_datasets import load_bitcoin
                print(f"Loading {key} using custom loader (Bitcoin-OTC)...")
                data = load_bitcoin('otc')
                
            elif loader_name == 'load_bitcoin_alpha':
                from analyze_datasets import load_bitcoin
                print(f"Loading {key} using custom loader (Bitcoin-Alpha)...")
                data = load_bitcoin('alpha')
                
            elif loader_name == 'load_credit_card':
                from analyze_datasets import load_credit_card
                print(f"Loading {key} using custom loader (Credit Card)...")
                data = load_credit_card()
                
            elif loader_name == 'load_ecommerce':
                from analyze_datasets import load_ecommerce
                print(f"Loading {key} using custom loader (Ecommerce)...")
                data = load_ecommerce()
                
            elif loader_name == 'load_cc_transactions':
                from analyze_datasets import load_cc_transactions
                print(f"Loading {key} using custom loader (CC Transactions)...")
                data = load_cc_transactions()
                
            elif loader_name == 'load_twitter_bots':
                from analyze_datasets import load_twitter_bots
                print(f"Loading {key} using custom loader (Twitter Bots)...")
                data = load_twitter_bots()
                
            elif loader_name == 'load_malicious_urls':
                from analyze_datasets import load_malicious_urls
                print(f"Loading {key} using custom loader (Malicious URLs)...")
                data = load_malicious_urls()
                
            elif loader_name == 'load_vehicle_loan_default':
                from analyze_datasets import load_vehicle_loan_default
                print(f"Loading {key} using custom loader (Vehicle Loan Default)...")
                data = load_vehicle_loan_default()
                
            elif loader_name == 'load_fake_job_postings':
                from analyze_datasets import load_fake_job_postings
                print(f"Loading {key} using custom loader (Fake Job Postings)...")
                data = load_fake_job_postings()
                
            elif loader_name == 'load_customs':
                from analyze_datasets import load_customs
                print(f"Loading {key} using custom loader (Customs)...")
                data = load_customs()
                
            elif loader_name == 'load_ieee_cis':
                from analyze_datasets import load_ieee_cis
                print(f"Loading {key} using custom loader (IEEE-CIS)...")
                data = load_ieee_cis()
                
            else:
                print(f"Unknown loader: {loader_name}")
                return None
            
            if data is None:
                print(f"Failed to load {key} dataset")
                return None
            
            X = data['X']
            y = data['y']
            
            # Handle labeled_mask if present (for semi-supervised datasets like Elliptic)
            if 'labeled_mask' in data:
                labeled_mask = data['labeled_mask']
                if hasattr(labeled_mask, 'numpy'):
                    labeled_mask = labeled_mask.numpy()
                X = X[labeled_mask]
                y = y[labeled_mask]
            
            # Split into train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            return X_train, y_train, X_test, y_test
            
        except Exception as e:
            print(f"Error loading {key}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Default: Load from FDB
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
# RESULTS SAVING
# =============================================================================

def save_results_to_csv(results, output_dir='.'):
    """
    Save experiment results to CSV files.
    
    Creates two files:
    1. Individual experiment file: experiment_results_{dataset}_{timestamp}.csv
    2. Master results file: all_experiment_results.csv (appends to existing)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Prepare rows for the CSV
    rows = []
    
    # Base info for all models
    base_info = {
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Dataset': results['name'],
        'Dataset_Key': results['dataset'],
        'Graph_Utility': results['graph_utility'],
        'N_Train': results['n_train'],
        'N_Test': results['n_test'],
        'N_Features': results.get('n_features', 0),
        'Fraud_Rate': results['fraud_rate']
    }
    
    # Add metrics for each model
    model_names = {'xgb': 'XGBoost', 'rf': 'RandomForest', 'lgb': 'LightGBM'}
    
    for model_key, model_name in model_names.items():
        if model_key in results and results[model_key] is not None:
            row = base_info.copy()
            row['Model'] = model_name
            
            # Add all metrics
            metrics = results[model_key]
            for metric_name, metric_value in metrics.items():
                row[metric_name] = metric_value
            
            rows.append(row)
    
    if not rows:
        print("No results to save.")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Reorder columns for better readability
    priority_cols = ['Timestamp', 'Dataset', 'Dataset_Key', 'Model', 
                     'AUC', 'AP', 'Best_F1', 'NDCG', 
                     'P@1%', 'P@5%', 'P@10%', 
                     'R@1%', 'R@5%', 'R@10%',
                     'Lift@1%', 'Lift@5%', 'Lift@10%', 
                     'F1@0.5',
                     'Graph_Utility', 'Fraud_Rate', 'N_Train', 'N_Test', 'N_Features']
    
    # Order columns: priority columns first, then any remaining
    ordered_cols = [c for c in priority_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + remaining_cols]
    
    # Save individual experiment file
    individual_file = os.path.join(output_dir, f"experiment_results_{results['dataset']}_{timestamp}.csv")
    df.to_csv(individual_file, index=False)
    print(f"\n✓ Results saved to: {individual_file}")
    
    # Append to master results file
    master_file = os.path.join(output_dir, 'all_experiment_results.csv')
    if os.path.exists(master_file):
        # Read existing and append
        existing_df = pd.read_csv(master_file)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_csv(master_file, index=False)
        print(f"✓ Results appended to: {master_file}")
    else:
        df.to_csv(master_file, index=False)
        print(f"✓ Master results file created: {master_file}")
    
    # Print a summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    summary_cols = ['Model', 'AUC', 'AP', 'Best_F1', 'P@1%', 'Lift@1%']
    summary_cols = [c for c in summary_cols if c in df.columns]
    print(df[summary_cols].to_string(index=False))
    
    return individual_file


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(key, output_dir='.'):
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
        'n_features': X_train.shape[1],
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
    else:
        results['xgb'] = None
    
    # Random Forest
    print("\n" + "-"*40)
    print("Training Random Forest...")
    rf_pred, _ = train_random_forest(X_train, y_train, X_test, y_test)
    rf_metrics = compute_all_metrics(y_test, rf_pred)
    print_metrics(rf_metrics, "Random Forest Results")
    results['rf'] = rf_metrics
    
    # LightGBM
    print("\n" + "-"*40)
    print("Training LightGBM...")
    lgb_pred, _ = train_lightgbm(X_train, y_train, X_test, y_test)
    if lgb_pred is not None:
        lgb_metrics = compute_all_metrics(y_test, lgb_pred)
        print_metrics(lgb_metrics, "LightGBM Results")
        results['lgb'] = lgb_metrics
    else:
        results['lgb'] = None
    
    # Save results to CSV
    save_results_to_csv(results, output_dir)
    
    return results


def run_all_experiments(output_dir='.'):
    """Run experiments on all available datasets."""
    
    print("="*80)
    print("FRAUD DETECTION BENCHMARK EXPERIMENTS")
    print("="*80)
    print(f"Total Datasets: {len(FDB_DATASETS)}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output Directory: {output_dir}")
    
    # List all datasets
    print("\nAvailable Datasets:")
    for key, info in FDB_DATASETS.items():
        source = info.get('source', 'fdb')
        print(f"  - {key}: {info['name']} (source: {source})")
    
    all_results = []
    
    for key in FDB_DATASETS.keys():
        try:
            result = run_experiment(key, output_dir)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"Error with {key}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create final summary
    if all_results:
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        
        summary_data = []
        for r in all_results:
            row = {
                'Dataset': r['name'],
                'Graph_Utility': r['graph_utility'],
                'Fraud_Rate': f"{r['fraud_rate']*100:.2f}%",
            }
            
            if r.get('xgb') is not None:
                row['XGB_AUC'] = f"{r['xgb']['AUC']:.4f}"
                row['XGB_AP'] = f"{r['xgb']['AP']:.4f}"
            else:
                row['XGB_AUC'] = 'N/A'
                row['XGB_AP'] = 'N/A'
            
            if r.get('rf') is not None:
                row['RF_AUC'] = f"{r['rf']['AUC']:.4f}"
                row['RF_AP'] = f"{r['rf']['AP']:.4f}"
            else:
                row['RF_AUC'] = 'N/A'
                row['RF_AP'] = 'N/A'
            
            if r.get('lgb') is not None:
                row['LGB_AUC'] = f"{r['lgb']['AUC']:.4f}"
                row['LGB_AP'] = f"{r['lgb']['AP']:.4f}"
            else:
                row['LGB_AUC'] = 'N/A'
                row['LGB_AP'] = 'N/A'
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Save final summary
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = os.path.join(output_dir, f'experiment_summary_{timestamp}.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"\nFinal summary saved to: {summary_file}")
    
    return all_results


def list_datasets():
    """Print list of all available datasets."""
    print("\n" + "="*80)
    print("AVAILABLE DATASETS")
    print("="*80)
    
    print("\nFDB Datasets (Amazon Fraud Detection Benchmark):")
    for key, info in FDB_DATASETS.items():
        if info.get('source') == 'fdb':
            print(f"  {key:<20} - {info['name']}")
    
    print("\nCustom Datasets (from analyze_datasets.py):")
    for key, info in FDB_DATASETS.items():
        if info.get('source') == 'custom':
            print(f"  {key:<20} - {info['name']}")
    
    print("\nUsage:")
    print("  python fdb_experiments.py --dataset <dataset_key>")
    print("  python fdb_experiments.py --dataset all")
    print("  python fdb_experiments.py --list")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Fraud Detection Benchmark Experiments')
    parser.add_argument('--dataset', type=str, default='all',
                        help='Dataset key (or "all" for all datasets)')
    parser.add_argument('--output', type=str, default='.',
                        help='Output directory for results')
    parser.add_argument('--list', action='store_true',
                        help='List all available datasets')
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
        return
    
    if args.dataset == 'all':
        run_all_experiments(args.output)
    elif args.dataset in FDB_DATASETS:
        run_experiment(args.dataset, args.output)
    else:
        print(f"Unknown dataset: {args.dataset}")
        print(f"Available datasets: {list(FDB_DATASETS.keys())}")
        print("\nUse --list to see all available datasets")


if __name__ == "__main__":
    main()
