#!/usr/bin/env python3
"""
DATASET CHARACTERISTICS ANALYSIS
================================

Analyzes key properties of fraud detection datasets to understand
why certain methods work better on certain datasets.

Metrics computed:
1. Homophily - Do connected nodes share labels?
2. Feature Predictiveness - Can features alone predict labels?
3. Graph Utility - Does graph add value beyond features?
4. Density - How dense is the graph?
5. Class Imbalance - Fraud rate
6. Degree Statistics - Connectivity patterns
7. Clustering Coefficient - Local structure
8. Label Propagation Accuracy - Simple graph-based prediction
9. Feature-Label Correlation - Feature informativeness

Author: IJCAI 2025
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# METRIC COMPUTATION FUNCTIONS
# =============================================================================

def compute_homophily(edge_index, y, labeled_mask=None):
    """
    Compute edge homophily: fraction of edges connecting same-label nodes.
    
    High homophily (>0.5) = GNNs should help (neighbors have same label)
    Low homophily (<0.5) = GNNs may hurt (neighbors have different label)
    """
    if isinstance(edge_index, torch.Tensor):
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
    else:
        src, dst = edge_index[0], edge_index[1]
    
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    if labeled_mask is not None:
        if isinstance(labeled_mask, torch.Tensor):
            labeled_mask = labeled_mask.cpu().numpy()
        # Only consider edges between labeled nodes
        valid_edges = labeled_mask[src] & labeled_mask[dst]
        src = src[valid_edges]
        dst = dst[valid_edges]
    
    if len(src) == 0:
        return 0.5
    
    same_label = (y[src] == y[dst]).mean()
    return same_label


def compute_class_homophily(edge_index, y, target_class=1, labeled_mask=None):
    """
    Compute homophily specifically for fraud class.
    
    Returns: fraction of fraud node neighbors that are also fraud
    """
    if isinstance(edge_index, torch.Tensor):
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
    else:
        src, dst = edge_index[0], edge_index[1]
    
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    n_nodes = len(y)
    
    # Build adjacency
    adj = defaultdict(list)
    for s, d in zip(src, dst):
        adj[s].append(d)
        adj[d].append(s)
    
    # For each fraud node, compute fraction of fraud neighbors
    fraud_nodes = np.where(y == target_class)[0]
    if labeled_mask is not None:
        if isinstance(labeled_mask, torch.Tensor):
            labeled_mask = labeled_mask.cpu().numpy()
        fraud_nodes = fraud_nodes[labeled_mask[fraud_nodes]]
    
    if len(fraud_nodes) == 0:
        return 0.0
    
    fraud_neighbor_ratios = []
    for node in fraud_nodes:
        neighbors = adj[node]
        if len(neighbors) > 0:
            if labeled_mask is not None:
                neighbors = [n for n in neighbors if labeled_mask[n]]
            if len(neighbors) > 0:
                fraud_ratio = sum(y[n] == target_class for n in neighbors) / len(neighbors)
                fraud_neighbor_ratios.append(fraud_ratio)
    
    if len(fraud_neighbor_ratios) == 0:
        return 0.0
    
    return np.mean(fraud_neighbor_ratios)


def compute_adjusted_homophily(edge_index, y, labeled_mask=None):
    """
    Adjusted homophily that accounts for class imbalance.
    
    H_adj = (H - H_random) / (1 - H_random)
    where H_random = p^2 + (1-p)^2 for binary classification
    
    Adjusted > 0 means graph structure is informative
    """
    h = compute_homophily(edge_index, y, labeled_mask)
    
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    if labeled_mask is not None:
        if isinstance(labeled_mask, torch.Tensor):
            labeled_mask = labeled_mask.cpu().numpy()
        y = y[labeled_mask]
    
    p = (y == 1).mean()
    h_random = p**2 + (1-p)**2
    
    if h_random >= 1:
        return 0.0
    
    h_adj = (h - h_random) / (1 - h_random)
    return h_adj


def compute_feature_predictiveness(X, y, labeled_mask=None, cv=3):
    """
    Compute how well features alone can predict labels using cross-validation.
    
    Returns AUC of Logistic Regression on features only.
    High value = features are informative, XGBoost should work well
    """
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    if labeled_mask is not None:
        if isinstance(labeled_mask, torch.Tensor):
            labeled_mask = labeled_mask.cpu().numpy()
        X = X[labeled_mask]
        y = y[labeled_mask]
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use Logistic Regression with CV
    model = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
    
    try:
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
        return np.mean(scores)
    except:
        return 0.5


def compute_feature_predictiveness_rf(X, y, labeled_mask=None, cv=3):
    """
    Feature predictiveness using Random Forest (handles non-linear relationships)
    """
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    if labeled_mask is not None:
        if isinstance(labeled_mask, torch.Tensor):
            labeled_mask = labeled_mask.cpu().numpy()
        X = X[labeled_mask]
        y = y[labeled_mask]
    
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    model = RandomForestClassifier(n_estimators=50, max_depth=4, class_weight='balanced', 
                                   random_state=42, n_jobs=-1)
    
    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        return np.mean(scores)
    except:
        return 0.5


def compute_graph_utility(X, edge_index, y, labeled_mask=None):
    """
    Compute graph utility: additional AUC gain from using graph features.
    
    Graph Utility = AUC(features + graph) - AUC(features only)
    
    High value = graph structure adds significant value
    """
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    if isinstance(edge_index, torch.Tensor):
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
    else:
        src, dst = edge_index[0], edge_index[1]
    
    n_nodes = X.shape[0]
    
    # Build adjacency
    adj = defaultdict(list)
    for s, d in zip(src, dst):
        adj[s].append(d)
        adj[d].append(s)
    
    # Compute simple graph features
    in_degree = np.array([len(adj[i]) for i in range(n_nodes)], dtype=np.float32)
    
    # Fraud neighbor ratio (if we had labels)
    fraud_neighbor_ratio = np.zeros(n_nodes, dtype=np.float32)
    for node in range(n_nodes):
        neighbors = adj[node]
        if len(neighbors) > 0:
            fraud_neighbor_ratio[node] = sum(y[n] for n in neighbors) / len(neighbors)
    
    # Combine features
    graph_features = np.column_stack([in_degree, np.log1p(in_degree), fraud_neighbor_ratio])
    X_combined = np.concatenate([X, graph_features], axis=1)
    
    if labeled_mask is not None:
        if isinstance(labeled_mask, torch.Tensor):
            labeled_mask = labeled_mask.cpu().numpy()
        X = X[labeled_mask]
        X_combined = X_combined[labeled_mask]
        y = y[labeled_mask]
    
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    X_combined = np.nan_to_num(X_combined, nan=0, posinf=0, neginf=0)
    
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    X_scaled = scaler1.fit_transform(X)
    X_comb_scaled = scaler2.fit_transform(X_combined)
    
    model = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
    
    try:
        auc_features = cross_val_score(model, X_scaled, y, cv=3, scoring='roc_auc').mean()
        auc_combined = cross_val_score(model, X_comb_scaled, y, cv=3, scoring='roc_auc').mean()
        return auc_combined - auc_features
    except:
        return 0.0


def compute_density(edge_index, n_nodes):
    """
    Compute graph density: E / (N * (N-1))
    
    Higher density = more edges per node
    """
    if isinstance(edge_index, torch.Tensor):
        n_edges = edge_index.shape[1]
    else:
        n_edges = len(edge_index[0])
    
    max_edges = n_nodes * (n_nodes - 1)
    if max_edges == 0:
        return 0
    
    return n_edges / max_edges


def compute_degree_stats(edge_index, n_nodes):
    """
    Compute degree statistics: mean, std, max, min
    """
    if isinstance(edge_index, torch.Tensor):
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
    else:
        src, dst = edge_index[0], edge_index[1]
    
    # Count degrees
    degree = np.zeros(n_nodes)
    for s in src:
        degree[s] += 1
    for d in dst:
        degree[d] += 1
    
    return {
        'mean': np.mean(degree),
        'std': np.std(degree),
        'median': np.median(degree),
        'max': np.max(degree),
        'min': np.min(degree),
        'isolated': (degree == 0).sum()
    }


def compute_clustering_coefficient(edge_index, n_nodes, sample_size=10000):
    """
    Compute average local clustering coefficient.
    
    High clustering = dense local neighborhoods
    """
    if isinstance(edge_index, torch.Tensor):
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
    else:
        src, dst = edge_index[0], edge_index[1]
    
    # Build adjacency
    adj = defaultdict(set)
    for s, d in zip(src, dst):
        adj[s].add(d)
        adj[d].add(s)
    
    # Sample nodes for efficiency
    nodes = np.random.choice(n_nodes, min(sample_size, n_nodes), replace=False)
    
    clustering_coeffs = []
    for node in nodes:
        neighbors = list(adj[node])
        k = len(neighbors)
        if k < 2:
            continue
        
        # Count edges between neighbors
        triangles = 0
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i+1:]:
                if n2 in adj[n1]:
                    triangles += 1
        
        max_triangles = k * (k - 1) / 2
        if max_triangles > 0:
            clustering_coeffs.append(triangles / max_triangles)
    
    if len(clustering_coeffs) == 0:
        return 0.0
    
    return np.mean(clustering_coeffs)


def compute_label_propagation_accuracy(edge_index, y, labeled_mask=None, iterations=10):
    """
    Compute accuracy of simple label propagation.
    
    High accuracy = graph structure is useful for prediction
    """
    if isinstance(edge_index, torch.Tensor):
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
    else:
        src, dst = edge_index[0], edge_index[1]
    
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    n_nodes = len(y)
    
    # Build adjacency
    adj = defaultdict(list)
    for s, d in zip(src, dst):
        adj[s].append(d)
        adj[d].append(s)
    
    # Use 50% as "labeled" for LP, test on other 50%
    if labeled_mask is None:
        labeled_mask = np.ones(n_nodes, dtype=bool)
    elif isinstance(labeled_mask, torch.Tensor):
        labeled_mask = labeled_mask.cpu().numpy()
    
    labeled_idx = np.where(labeled_mask)[0]
    np.random.seed(42)
    np.random.shuffle(labeled_idx)
    
    split = len(labeled_idx) // 2
    train_idx = labeled_idx[:split]
    test_idx = labeled_idx[split:]
    
    train_mask = np.zeros(n_nodes, dtype=bool)
    train_mask[train_idx] = True
    
    # Initialize LP scores
    scores = np.zeros(n_nodes, dtype=np.float32)
    scores[train_idx] = y[train_idx].astype(np.float32)
    
    # Propagate
    for _ in range(iterations):
        new_scores = scores.copy()
        for node in range(n_nodes):
            if train_mask[node]:
                continue
            neighbors = adj[node]
            if len(neighbors) > 0:
                new_scores[node] = 0.5 * scores[node] + 0.5 * np.mean(scores[neighbors])
        scores = new_scores
    
    # Compute AUC on test set
    y_test = y[test_idx]
    scores_test = scores[test_idx]
    
    try:
        return roc_auc_score(y_test, scores_test)
    except:
        return 0.5


def compute_feature_label_correlation(X, y, labeled_mask=None):
    """
    Compute average absolute correlation between features and labels.
    
    High correlation = features are informative
    """
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    if labeled_mask is not None:
        if isinstance(labeled_mask, torch.Tensor):
            labeled_mask = labeled_mask.cpu().numpy()
        X = X[labeled_mask]
        y = y[labeled_mask]
    
    correlations = []
    for i in range(min(X.shape[1], 100)):  # Limit to first 100 features
        col = X[:, i]
        if np.std(col) > 0:
            corr = np.abs(np.corrcoef(col, y)[0, 1])
            if not np.isnan(corr):
                correlations.append(corr)
    
    if len(correlations) == 0:
        return 0.0
    
    return np.mean(correlations)


def compute_neighbor_label_entropy(edge_index, y, labeled_mask=None):
    """
    Compute average entropy of neighbor labels.
    
    Low entropy = neighbors have similar labels (good for GNNs)
    High entropy = neighbors have diverse labels (bad for GNNs)
    """
    if isinstance(edge_index, torch.Tensor):
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
    else:
        src, dst = edge_index[0], edge_index[1]
    
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    n_nodes = len(y)
    
    # Build adjacency
    adj = defaultdict(list)
    for s, d in zip(src, dst):
        adj[s].append(d)
        adj[d].append(s)
    
    if labeled_mask is not None:
        if isinstance(labeled_mask, torch.Tensor):
            labeled_mask = labeled_mask.cpu().numpy()
    else:
        labeled_mask = np.ones(n_nodes, dtype=bool)
    
    entropies = []
    for node in range(n_nodes):
        if not labeled_mask[node]:
            continue
        neighbors = [n for n in adj[node] if labeled_mask[n]]
        if len(neighbors) < 2:
            continue
        
        neighbor_labels = y[neighbors]
        p_fraud = neighbor_labels.mean()
        
        if p_fraud > 0 and p_fraud < 1:
            entropy = -p_fraud * np.log2(p_fraud) - (1-p_fraud) * np.log2(1-p_fraud)
            entropies.append(entropy)
    
    if len(entropies) == 0:
        return 0.0
    
    return np.mean(entropies)


# =============================================================================
# DATA LOADERS
# =============================================================================

def load_elliptic():
    """Load Elliptic Bitcoin dataset"""
    try:
        from torch_geometric.datasets import EllipticBitcoinDataset
        dataset = EllipticBitcoinDataset(root='./data/elliptic')
        data = dataset[0]
        X = data.x.numpy()
        y_raw = data.y.numpy()
        y = np.full(len(y_raw), -1, dtype=np.int64)
        y[y_raw == 0] = 0
        y[y_raw == 1] = 1
        labeled_mask = y >= 0
        return {'X': X.astype(np.float32), 'y': y, 'edge_index': data.edge_index,
                'labeled_mask': labeled_mask, 'name': 'Elliptic'}
    except Exception as e:
        print(f"Error loading Elliptic: {e}")
        return None


def load_amazon():
    """Load Amazon fraud dataset"""
    import scipy.io as sio
    data_dir = './data/amazon_fraud'
    os.makedirs(data_dir, exist_ok=True)
    
    mat_file = None
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.mat'):
                mat_file = os.path.join(root, f)
    
    if mat_file is None:
        import urllib.request
        import zipfile
        url = "https://data.dgl.ai/dataset/FraudAmazon.zip"
        zip_file = os.path.join(data_dir, 'FraudAmazon.zip')
        urllib.request.urlretrieve(url, zip_file)
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(data_dir)
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith('.mat'):
                    mat_file = os.path.join(root, f)
    
    data = sio.loadmat(mat_file)
    X = data['features']
    if hasattr(X, 'toarray'):
        X = X.toarray()
    y = data['label'].flatten().astype(np.int64)
    
    src_list, dst_list = [], []
    for key in ['net_upu', 'net_usu', 'net_uvu']:
        if key in data:
            adj = data[key]
            if hasattr(adj, 'tocoo'):
                adj = adj.tocoo()
                src_list.append(adj.row)
                dst_list.append(adj.col)
    
    edge_index = torch.tensor([np.concatenate(src_list), np.concatenate(dst_list)], dtype=torch.long)
    return {'X': X.astype(np.float32), 'y': y, 'edge_index': edge_index, 'name': 'Amazon'}


def load_yelp():
    """Load Yelp fraud dataset"""
    import scipy.io as sio
    data_dir = './data/yelp_fraud'
    os.makedirs(data_dir, exist_ok=True)
    
    mat_file = None
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.mat'):
                mat_file = os.path.join(root, f)
    
    if mat_file is None:
        import urllib.request
        import zipfile
        url = "https://data.dgl.ai/dataset/FraudYelp.zip"
        zip_file = os.path.join(data_dir, 'FraudYelp.zip')
        urllib.request.urlretrieve(url, zip_file)
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(data_dir)
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith('.mat'):
                    mat_file = os.path.join(root, f)
    
    data = sio.loadmat(mat_file)
    X = data['features']
    if hasattr(X, 'toarray'):
        X = X.toarray()
    y = data['label'].flatten().astype(np.int64)
    
    src_list, dst_list = [], []
    for key in ['net_rur', 'net_rtr', 'net_rsr']:
        if key in data:
            adj = data[key]
            if hasattr(adj, 'tocoo'):
                adj = adj.tocoo()
                src_list.append(adj.row)
                dst_list.append(adj.col)
    
    edge_index = torch.tensor([np.concatenate(src_list), np.concatenate(dst_list)], dtype=torch.long)
    return {'X': X.astype(np.float32), 'y': y, 'edge_index': edge_index, 'name': 'Yelp'}


def load_bitcoin(dataset_type='otc'):
    """Load Bitcoin trust network dataset"""
    try:
        from torch_geometric.datasets import BitcoinOTC
        
        if dataset_type.lower() == 'otc':
            dataset = BitcoinOTC(root='./data/bitcoin_otc')
        else:
            dataset = BitcoinOTC(root='./data/bitcoin_alpha')
        
        data = dataset[0]
        n_nodes = data.edge_index.max().item() + 1
        
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            in_weights = np.zeros(n_nodes, dtype=np.float32)
            out_weights = np.zeros(n_nodes, dtype=np.float32)
            in_count = np.zeros(n_nodes, dtype=np.float32)
            out_count = np.zeros(n_nodes, dtype=np.float32)
            
            edge_attr = data.edge_attr.numpy().flatten()
            src = data.edge_index[0].numpy()
            dst = data.edge_index[1].numpy()
            
            for s, d, w in zip(src, dst, edge_attr):
                out_weights[s] += w
                in_weights[d] += w
                out_count[s] += 1
                in_count[d] += 1
            
            X = np.column_stack([
                in_weights, out_weights, in_count, out_count,
                np.where(in_count > 0, in_weights / in_count, 0),
                np.where(out_count > 0, out_weights / out_count, 0)
            ])
            
            y = (in_weights < -2).astype(np.int64)
        else:
            X = np.random.randn(n_nodes, 32).astype(np.float32)
            y = np.random.binomial(1, 0.1, n_nodes)
        
        return {'X': X, 'y': y, 'edge_index': data.edge_index, 'name': f'Bitcoin-{dataset_type.upper()}'}
    except Exception as e:
        print(f"Error loading Bitcoin-{dataset_type}: {e}")
        return None


def load_ieee_cis(data_path=None):
    """Load IEEE-CIS Fraud Detection dataset"""
    if data_path is None:
        for p in ['./data/ieee-fraud-detection', './data/ieee-cis', './ieee-cis-fraud-detection', 
                  '/kaggle/input/ieee-cis-fraud-detection', './data/ieee-cis-fraud-detection']:
            if os.path.exists(p):
                data_path = p
                break
    
    if data_path is None or not os.path.exists(data_path):
        return None
    
    try:
        train_trans = pd.read_csv(os.path.join(data_path, 'train_transaction.csv'))
        train_id = pd.read_csv(os.path.join(data_path, 'train_identity.csv'))
        
        data = train_trans.merge(train_id, on='TransactionID', how='left')
        
        y = data['isFraud'].values.astype(np.int64)
        
        num_cols = data.select_dtypes(include=[np.number]).columns.drop(['TransactionID', 'isFraud'])
        X = data[num_cols].fillna(-999).values.astype(np.float32)
        
        n_nodes = len(data)
        src_list, dst_list = [], []
        
        for col in ['card1', 'card2', 'P_emaildomain']:
            if col in data.columns:
                groups = data.groupby(col).indices
                for idx_list in groups.values():
                    if len(idx_list) > 1 and len(idx_list) < 1000:
                        for i in range(len(idx_list)):
                            for j in range(i+1, min(i+10, len(idx_list))):
                                src_list.extend([idx_list[i], idx_list[j]])
                                dst_list.extend([idx_list[j], idx_list[i]])
        
        if len(src_list) == 0:
            src_list = np.random.randint(0, n_nodes, n_nodes * 10)
            dst_list = np.random.randint(0, n_nodes, n_nodes * 10)
        
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        
        # Limit size
        if n_nodes > 100000:
            idx = np.random.choice(n_nodes, 100000, replace=False)
            X = X[idx]
            y = y[idx]
            idx_set = set(idx)
            idx_map = {old: new for new, old in enumerate(idx)}
            new_src, new_dst = [], []
            for s, d in zip(src_list, dst_list):
                if s in idx_set and d in idx_set:
                    new_src.append(idx_map[s])
                    new_dst.append(idx_map[d])
            edge_index = torch.tensor([new_src, new_dst], dtype=torch.long)
        
        return {'X': X, 'y': y, 'edge_index': edge_index, 'name': 'IEEE-CIS'}
    except Exception as e:
        print(f"Error loading IEEE-CIS: {e}")
        return None


def load_credit_card(credit_card_path=None):
    """
    Load Credit Card fraud dataset.
    
    Expected columns:
    - transaction_id: transaction ID
    - time: transaction time/timestamp
    - merchant_id: merchant identifier (used for graph)
    - card_id: card identifier (used for graph)
    - user_id: user/customer identifier (used for graph)
    - merchant_category: merchant category code (used for graph)
    - amount: transaction amount
    - is_fraud: label (0/1)
    
    Additional feature columns (V1-V28 for anonymized features, or custom features)
    """
    if credit_card_path is None:
        for p in ['./data/credit_card_fraud/creditcard.csv', './data/credit_card/credit_card.csv', 
                  './credit_card.csv', '../credit_card.csv', 
                  './data/credit_card.csv', '../data/credit_card.csv',
                  './data/creditcard.csv', './creditcard.csv']:
            if os.path.exists(p):
                credit_card_path = p
                break
    
    if credit_card_path is None or not os.path.exists(credit_card_path):
        print(f"    Credit Card file not found. Searched paths:")
        print(f"      ./data/credit_card_fraud/creditcard.csv")
        print(f"      ./data/credit_card/credit_card.csv")
        print(f"      ./data/creditcard.csv")
        print(f"    Please specify path with --credit_card_path")
        return None
    
    try:
        print(f"    Loading from: {credit_card_path}")
        data = pd.read_csv(credit_card_path)
        print(f"    Loaded {len(data):,} rows, {len(data.columns)} columns")
        print(f"    Columns: {list(data.columns)}")
        
        # Identify label column
        label_col = None
        for col in ['is_fraud', 'isFraud', 'fraud', 'Class', 'label', 'target']:
            if col in data.columns:
                label_col = col
                break
        
        if label_col is None:
            print(f"    ERROR: Could not find label column. Expected one of: is_fraud, isFraud, fraud, Class, label, target")
            return None
        
        y = data[label_col].values.astype(np.int64)
        print(f"    Label column: {label_col}, Fraud rate: {y.mean():.2%}")
        
        # Identify ID and exclude columns
        id_cols = ['transaction_id', 'id', 'ID', 'TransactionID', 'trans_id']
        exclude_cols = [label_col] + id_cols
        
        # Identify categorical columns for graph construction
        graph_cols = []
        for col in ['merchant_id', 'card_id', 'user_id', 'merchant_category', 'category',
                    'cc_num', 'merchant', 'zip', 'city', 'state']:
            if col in data.columns:
                graph_cols.append(col)
        print(f"    Graph columns: {graph_cols}")
        
        # Prepare features
        feature_cols = [c for c in data.columns if c not in exclude_cols]
        
        # Handle time/datetime columns
        for time_col in ['time', 'Time', 'trans_date_trans_time', 'datetime', 'timestamp']:
            if time_col in feature_cols:
                try:
                    # Try to parse as datetime
                    if data[time_col].dtype == 'object':
                        dates = pd.to_datetime(data[time_col])
                        data['hour'] = dates.dt.hour
                        data['day_of_week'] = dates.dt.dayofweek
                        data['day_of_month'] = dates.dt.day
                        feature_cols.remove(time_col)
                        feature_cols.extend(['hour', 'day_of_week', 'day_of_month'])
                    # If numeric (like seconds), keep as is
                except:
                    if data[time_col].dtype == 'object':
                        feature_cols.remove(time_col)
        
        # Encode categorical features
        for col in feature_cols:
            if data[col].dtype == 'object' or str(data[col].dtype) == 'category':
                data[col] = pd.factorize(data[col])[0]
        
        X = data[feature_cols].fillna(-999).values.astype(np.float32)
        print(f"    Features: {len(feature_cols)} columns")
        
        # Build graph from categorical columns or time-based proximity
        n_nodes = len(data)
        src_list, dst_list = [], []
        
        if len(graph_cols) > 0:
            # Use categorical columns for graph construction
            for col in graph_cols:
                if col in data.columns:
                    print(f"      Building edges from {col} ({data[col].nunique()} unique values)...")
                    groups = data.groupby(col).indices
                    edges_added = 0
                    for group_name, idx_list in groups.items():
                        idx_array = np.array(list(idx_list))
                        if len(idx_array) > 1 and len(idx_array) < 1000:
                            # Connect nodes in same group (limit connections for large groups)
                            n_connect = min(len(idx_array), 20)
                            for i in range(len(idx_array)):
                                # Connect to next few nodes in group
                                for j in range(1, min(n_connect, len(idx_array) - i)):
                                    src_list.append(idx_array[i])
                                    dst_list.append(idx_array[(i + j) % len(idx_array)])
                                    src_list.append(idx_array[(i + j) % len(idx_array)])
                                    dst_list.append(idx_array[i])
                                    edges_added += 2
                    print(f"        Added {edges_added:,} edges")
        else:
            # No categorical columns - use time-based proximity for Kaggle-style datasets
            print(f"    No categorical graph columns found. Building time-based proximity graph...")
            
            # Check for Time column (common in Kaggle credit card dataset)
            time_col = None
            for tc in ['Time', 'time', 'timestamp']:
                if tc in data.columns:
                    time_col = tc
                    break
            
            if time_col is not None:
                # Sort by time and connect nearby transactions
                time_values = data[time_col].values
                sorted_indices = np.argsort(time_values)
                
                # Connect each transaction to k nearest neighbors in time
                k_neighbors = 5
                edges_added = 0
                for i, idx in enumerate(sorted_indices):
                    # Connect to previous k and next k transactions
                    for j in range(max(0, i - k_neighbors), min(len(sorted_indices), i + k_neighbors + 1)):
                        if i != j:
                            neighbor_idx = sorted_indices[j]
                            src_list.append(idx)
                            dst_list.append(neighbor_idx)
                            edges_added += 1
                print(f"      Added {edges_added:,} time-proximity edges")
            
            # Also add Amount-based similarity edges (connect transactions with similar amounts)
            if 'Amount' in data.columns:
                print(f"    Building amount-based similarity edges...")
                amounts = data['Amount'].values
                # Bin amounts and connect transactions in same bin
                amount_bins = pd.qcut(amounts, q=50, labels=False, duplicates='drop')
                for bin_val in np.unique(amount_bins):
                    bin_indices = np.where(amount_bins == bin_val)[0]
                    if len(bin_indices) > 1 and len(bin_indices) < 500:
                        # Sample connections within bin
                        n_sample = min(len(bin_indices), 10)
                        for i in range(len(bin_indices)):
                            for j in range(1, min(n_sample, len(bin_indices) - i)):
                                src_list.append(bin_indices[i])
                                dst_list.append(bin_indices[i + j])
                                src_list.append(bin_indices[i + j])
                                dst_list.append(bin_indices[i])
        
        if len(src_list) == 0:
            print(f"    WARNING: No edges created. Creating random edges.")
            src_list = np.random.randint(0, n_nodes, n_nodes * 5).tolist()
            dst_list = np.random.randint(0, n_nodes, n_nodes * 5).tolist()
        
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        print(f"    Total edges: {edge_index.shape[1]:,}")
        print(f"    Avg degree: {edge_index.shape[1] / n_nodes:.1f}")
        
        return {'X': X, 'y': y, 'edge_index': edge_index, 'name': 'Credit-Card'}
    
    except Exception as e:
        import traceback
        print(f"    Error loading Credit Card: {e}")
        traceback.print_exc()
        return None


def load_cc_transactions(cc_transactions_path=None):
    """
    Load Credit Card Transactions Fraud Detection dataset.
    
    Expected columns:
    - trans_date_trans_time: transaction datetime
    - cc_num: credit card number (used for graph)
    - merchant: merchant name (used for graph)
    - category: merchant category (used for graph)
    - amt: transaction amount
    - first, last: customer name
    - gender: customer gender
    - street, city, state, zip: customer address (city/state/zip used for graph)
    - lat, long: customer location
    - city_pop: city population
    - job: customer job
    - dob: date of birth
    - trans_num: transaction ID
    - unix_time: unix timestamp
    - merch_lat, merch_long: merchant location
    - is_fraud: label (0/1)
    """
    if cc_transactions_path is None:
        for p in ['./data/credit_card_transaction_fraud/fraudTrain.csv', 
                  './data/credit_card_transaction_fraud/fraudTest.csv',
                  './data/cc_transactions/fraudTrain.csv',
                  './fraudTrain.csv', '../fraudTrain.csv',
                  './data/fraudTrain.csv']:
            if os.path.exists(p):
                cc_transactions_path = p
                break
    
    if cc_transactions_path is None or not os.path.exists(cc_transactions_path):
        print(f"    CC Transactions file not found. Searched paths:")
        print(f"      ./data/credit_card_transaction_fraud/fraudTrain.csv")
        print(f"      ./data/cc_transactions/fraudTrain.csv")
        print(f"    Please specify path with --cc_transactions_path")
        return None
    
    try:
        print(f"    Loading from: {cc_transactions_path}")
        data = pd.read_csv(cc_transactions_path)
        print(f"    Loaded {len(data):,} rows, {len(data.columns)} columns")
        print(f"    Columns: {list(data.columns)[:15]}...")  # Truncate for readability
        
        # This dataset is large, sample if needed
        max_rows = 200000
        if len(data) > max_rows:
            print(f"    Sampling {max_rows:,} rows from {len(data):,} total...")
            data = data.sample(n=max_rows, random_state=42).reset_index(drop=True)
        
        # Identify label column
        label_col = None
        for col in ['is_fraud', 'isFraud', 'fraud', 'Class', 'label', 'target']:
            if col in data.columns:
                label_col = col
                break
        
        if label_col is None:
            print(f"    ERROR: Could not find label column. Expected one of: is_fraud, isFraud, fraud, Class, label, target")
            return None
        
        y = data[label_col].values.astype(np.int64)
        print(f"    Label column: {label_col}, Fraud rate: {y.mean():.2%}")
        
        # Identify ID and exclude columns
        id_cols = ['trans_num', 'Unnamed: 0', 'first', 'last', 'street', 'dob']
        exclude_cols = [label_col] + id_cols
        
        # Identify categorical columns for graph construction
        graph_cols = []
        for col in ['cc_num', 'merchant', 'category', 'city', 'state', 'zip', 'job']:
            if col in data.columns:
                # Only use columns with reasonable cardinality for graph
                n_unique = data[col].nunique()
                if n_unique > 1 and n_unique < len(data) * 0.5:
                    graph_cols.append(col)
        print(f"    Graph columns: {graph_cols}")
        
        # Prepare features
        feature_cols = [c for c in data.columns if c not in exclude_cols]
        
        # Handle datetime column
        if 'trans_date_trans_time' in feature_cols:
            try:
                dates = pd.to_datetime(data['trans_date_trans_time'])
                data['trans_hour'] = dates.dt.hour
                data['trans_dow'] = dates.dt.dayofweek
                data['trans_day'] = dates.dt.day
                data['trans_month'] = dates.dt.month
                feature_cols.remove('trans_date_trans_time')
                feature_cols.extend(['trans_hour', 'trans_dow', 'trans_day', 'trans_month'])
            except:
                if 'trans_date_trans_time' in feature_cols:
                    feature_cols.remove('trans_date_trans_time')
        
        # Encode categorical features
        for col in feature_cols:
            if col in data.columns:
                if data[col].dtype == 'object' or str(data[col].dtype) == 'category':
                    data[col] = pd.factorize(data[col])[0]
        
        # Filter to valid feature columns
        feature_cols = [c for c in feature_cols if c in data.columns]
        
        # Select numeric features only
        numeric_cols = []
        for col in feature_cols:
            if col in data.columns:
                try:
                    data[col].astype(np.float32)
                    numeric_cols.append(col)
                except:
                    pass
        
        X = data[numeric_cols].fillna(-999).values.astype(np.float32)
        print(f"    Features: {len(numeric_cols)} columns")
        
        # Build graph from categorical columns
        n_nodes = len(data)
        src_list, dst_list = [], []
        
        for col in graph_cols:
            if col in data.columns:
                n_unique = data[col].nunique()
                print(f"      Building edges from {col} ({n_unique} unique values)...")
                groups = data.groupby(col).indices
                edges_added = 0
                for group_name, idx_list in groups.items():
                    idx_array = np.array(list(idx_list))
                    if len(idx_array) > 1 and len(idx_array) < 500:
                        # Connect nodes in same group (limit connections for large groups)
                        n_connect = min(len(idx_array), 15)
                        for i in range(len(idx_array)):
                            # Connect to next few nodes in group
                            for j in range(1, min(n_connect, len(idx_array) - i)):
                                src_list.append(idx_array[i])
                                dst_list.append(idx_array[(i + j) % len(idx_array)])
                                src_list.append(idx_array[(i + j) % len(idx_array)])
                                dst_list.append(idx_array[i])
                                edges_added += 2
                print(f"        Added {edges_added:,} edges")
        
        if len(src_list) == 0:
            print(f"    WARNING: No edges created from graph columns. Creating random edges.")
            src_list = np.random.randint(0, n_nodes, n_nodes * 5).tolist()
            dst_list = np.random.randint(0, n_nodes, n_nodes * 5).tolist()
        
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        print(f"    Total edges: {edge_index.shape[1]:,}")
        print(f"    Avg degree: {edge_index.shape[1] / n_nodes:.1f}")
        
        return {'X': X, 'y': y, 'edge_index': edge_index, 'name': 'CC-Transactions'}
    
    except Exception as e:
        import traceback
        print(f"    Error loading CC Transactions: {e}")
        traceback.print_exc()
        return None


def load_twitter_bots(twitter_bots_path=None):
    """
    Load Twitter Human/Bots dataset.
    
    Expected columns:
    - id: Twitter account ID
    - created_at: account creation date
    - default_profile, default_profile_image: profile settings
    - description: bio text
    - favourites_count, followers_count, friends_count: engagement metrics
    - geo_enabled: location enabled
    - lang: language (used for graph)
    - location: location text (used for graph)
    - screen_name: username
    - statuses_count: tweet count
    - verified: verification status
    - average_tweets_per_day, account_age_days: derived metrics
    - account_type: "bot" or "human" (label)
    """
    if twitter_bots_path is None:
        for p in ['./data/twitter_human_bots/twitter_human_bots_dataset.csv',
                  './data/twitter_bots_accounts/twitter_human_bots_dataset.csv',
                  './data/twitter_bots/twitter_human_bots_dataset.csv',
                  './twitter_human_bots_dataset.csv', '../twitter_human_bots_dataset.csv',
                  './data/twitter_human_bots_dataset.csv']:
            if os.path.exists(p):
                twitter_bots_path = p
                break
    
    if twitter_bots_path is None or not os.path.exists(twitter_bots_path):
        print(f"    Twitter Bots file not found. Searched paths:")
        print(f"      ./data/twitter_human_bots/twitter_human_bots_dataset.csv")
        print(f"      ./data/twitter_bots_accounts/twitter_human_bots_dataset.csv")
        print(f"    Please specify path with --twitter_bots_path")
        return None
    
    try:
        print(f"    Loading from: {twitter_bots_path}")
        data = pd.read_csv(twitter_bots_path)
        print(f"    Loaded {len(data):,} rows, {len(data.columns)} columns")
        print(f"    Columns: {list(data.columns)}")
        
        # Identify label column
        label_col = None
        for col in ['account_type', 'label', 'is_bot', 'bot', 'class', 'target']:
            if col in data.columns:
                label_col = col
                break
        
        if label_col is None:
            print(f"    ERROR: Could not find label column. Expected one of: account_type, is_bot, bot, label, class")
            return None
        
        # Convert label to binary (bot=1, human=0)
        if data[label_col].dtype == 'object':
            label_map = {'bot': 1, 'human': 0, 'Bot': 1, 'Human': 0}
            y = data[label_col].map(label_map).values.astype(np.int64)
        else:
            y = data[label_col].values.astype(np.int64)
        
        print(f"    Label column: {label_col}, Bot rate: {y.mean():.2%}")
        
        # Identify columns to exclude from features
        id_cols = ['id', 'Unnamed: 0', 'screen_name', 'description', 'location', 
                   'profile_background_image_url', 'profile_image_url']
        exclude_cols = [label_col] + id_cols
        
        # Identify categorical columns for graph construction
        graph_cols = []
        for col in ['lang', 'verified', 'default_profile', 'default_profile_image', 'geo_enabled']:
            if col in data.columns:
                graph_cols.append(col)
        print(f"    Graph columns: {graph_cols}")
        
        # Prepare features
        feature_cols = [c for c in data.columns if c not in exclude_cols]
        
        # Handle datetime column
        if 'created_at' in feature_cols:
            try:
                dates = pd.to_datetime(data['created_at'])
                data['created_hour'] = dates.dt.hour
                data['created_dow'] = dates.dt.dayofweek
                data['created_month'] = dates.dt.month
                data['created_year'] = dates.dt.year
                feature_cols.remove('created_at')
                feature_cols.extend(['created_hour', 'created_dow', 'created_month', 'created_year'])
            except:
                if 'created_at' in feature_cols:
                    feature_cols.remove('created_at')
        
        # Encode categorical features
        for col in feature_cols:
            if col in data.columns:
                if data[col].dtype == 'object' or str(data[col].dtype) == 'category':
                    data[col] = pd.factorize(data[col])[0]
                elif data[col].dtype == 'bool':
                    data[col] = data[col].astype(int)
        
        # Filter to valid feature columns
        feature_cols = [c for c in feature_cols if c in data.columns]
        
        # Select numeric features only
        numeric_cols = []
        for col in feature_cols:
            if col in data.columns:
                try:
                    data[col].astype(np.float32)
                    numeric_cols.append(col)
                except:
                    pass
        
        X = data[numeric_cols].fillna(-999).values.astype(np.float32)
        print(f"    Features: {len(numeric_cols)} columns ({numeric_cols[:5]}...)")
        
        # Build graph from categorical columns
        n_nodes = len(data)
        src_list, dst_list = [], []
        
        for col in graph_cols:
            if col in data.columns:
                print(f"      Building edges from {col} ({data[col].nunique()} unique values)...")
                groups = data.groupby(col).indices
                edges_added = 0
                for group_name, idx_list in groups.items():
                    idx_array = np.array(list(idx_list))
                    if len(idx_array) > 1 and len(idx_array) < 500:
                        # Connect nodes in same group (limit connections for large groups)
                        n_connect = min(len(idx_array), 15)
                        for i in range(len(idx_array)):
                            # Connect to next few nodes in group
                            for j in range(1, min(n_connect, len(idx_array) - i)):
                                src_list.append(idx_array[i])
                                dst_list.append(idx_array[(i + j) % len(idx_array)])
                                src_list.append(idx_array[(i + j) % len(idx_array)])
                                dst_list.append(idx_array[i])
                                edges_added += 2
                print(f"        Added {edges_added:,} edges")
        
        # Also add edges based on account_age_days proximity (accounts created around same time)
        if 'account_age_days' in data.columns:
            print(f"    Building edges based on account age proximity...")
            age_values = data['account_age_days'].values
            sorted_indices = np.argsort(age_values)
            k_neighbors = 5
            age_edges = 0
            for i, idx in enumerate(sorted_indices):
                for j in range(max(0, i - k_neighbors), min(len(sorted_indices), i + k_neighbors + 1)):
                    if i != j:
                        neighbor_idx = sorted_indices[j]
                        src_list.append(idx)
                        dst_list.append(neighbor_idx)
                        age_edges += 1
            print(f"      Added {age_edges:,} age-proximity edges")
        
        if len(src_list) == 0:
            print(f"    WARNING: No edges created. Creating random edges.")
            src_list = np.random.randint(0, n_nodes, n_nodes * 5).tolist()
            dst_list = np.random.randint(0, n_nodes, n_nodes * 5).tolist()
        
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        print(f"    Total edges: {edge_index.shape[1]:,}")
        print(f"    Avg degree: {edge_index.shape[1] / n_nodes:.1f}")
        
        return {'X': X, 'y': y, 'edge_index': edge_index, 'name': 'Twitter-Bots'}
    
    except Exception as e:
        import traceback
        print(f"    Error loading Twitter Bots: {e}")
        traceback.print_exc()
        return None


def load_fake_job_postings(fake_job_path=None):
    """
    Load Real/Fake Job Posting Prediction dataset.
    
    Expected columns:
    - job_id: job posting ID
    - title: job title (text - extract features)
    - location: location (used for graph - country extraction)
    - department: department (used for graph)
    - salary_range: salary range
    - company_profile, description, requirements, benefits: text columns
    - telecommuting, has_company_logo, has_questions: binary flags
    - employment_type: Full-time, Part-time, etc. (used for graph)
    - required_experience: experience level (used for graph)
    - required_education: education level (used for graph)
    - industry: industry type (used for graph)
    - function: job function (used for graph)
    - fraudulent: label (0/1)
    """
    if fake_job_path is None:
        for p in ['./data/real_fake_job_posting_prediction/fake_job_postings.csv',
                  './data/fake_job_postings/fake_job_postings.csv',
                  './data/job_postings/fake_job_postings.csv',
                  './fake_job_postings.csv', '../fake_job_postings.csv',
                  './data/fake_job_postings.csv']:
            if os.path.exists(p):
                fake_job_path = p
                break
    
    if fake_job_path is None or not os.path.exists(fake_job_path):
        print(f"    Fake Job Postings file not found. Searched paths:")
        print(f"      ./data/real_fake_job_posting_prediction/fake_job_postings.csv")
        print(f"      ./data/fake_job_postings/fake_job_postings.csv")
        print(f"    Please specify path with --fake_job_path")
        return None
    
    try:
        print(f"    Loading from: {fake_job_path}")
        data = pd.read_csv(fake_job_path)
        print(f"    Loaded {len(data):,} rows, {len(data.columns)} columns")
        print(f"    Columns: {list(data.columns)}")
        
        # Identify label column
        label_col = None
        for col in ['fraudulent', 'is_fraud', 'fraud', 'fake', 'is_fake', 'label', 'target']:
            if col in data.columns:
                label_col = col
                break
        
        if label_col is None:
            print(f"    ERROR: Could not find label column. Expected one of: fraudulent, is_fraud, fraud, fake")
            return None
        
        y = data[label_col].values.astype(np.int64)
        print(f"    Label column: {label_col}, Fraudulent rate: {y.mean():.2%}")
        
        # Identify ID and exclude columns
        id_cols = ['job_id', 'id', 'ID']
        text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
        exclude_cols = [label_col] + id_cols
        
        # Identify categorical columns for graph construction
        graph_cols = []
        for col in ['employment_type', 'required_experience', 'required_education', 
                    'industry', 'function', 'department']:
            if col in data.columns:
                # Only use columns with reasonable cardinality for graph
                n_unique = data[col].nunique()
                if n_unique > 1 and n_unique < len(data) * 0.3:
                    graph_cols.append(col)
        print(f"    Graph columns: {graph_cols}")
        
        # Extract country from location
        if 'location' in data.columns:
            def extract_country(loc):
                if pd.isna(loc) or loc == '':
                    return 'unknown'
                parts = str(loc).split(',')
                if len(parts) >= 1:
                    return parts[0].strip()
                return 'unknown'
            data['country'] = data['location'].apply(extract_country)
            n_countries = data['country'].nunique()
            if n_countries > 1 and n_countries < len(data) * 0.3:
                graph_cols.append('country')
            print(f"    Extracted {n_countries} unique countries from location")
        
        # Extract features from text columns
        print(f"    Extracting features from text columns...")
        feature_arrays = []
        feature_names = []
        
        for text_col in text_cols:
            if text_col in data.columns:
                # Fill NA with empty string
                texts = data[text_col].fillna('').astype(str)
                
                # Text length
                col_len = texts.apply(len).values.astype(np.float32)
                feature_arrays.append(col_len)
                feature_names.append(f'{text_col}_len')
                
                # Word count
                col_words = texts.apply(lambda x: len(x.split())).values.astype(np.float32)
                feature_arrays.append(col_words)
                feature_names.append(f'{text_col}_words')
                
                # Has content flag
                col_has = (texts.apply(len) > 0).astype(np.float32).values
                feature_arrays.append(col_has)
                feature_names.append(f'{text_col}_has')
        
        # Add binary flags
        for flag_col in ['telecommuting', 'has_company_logo', 'has_questions']:
            if flag_col in data.columns:
                flag_values = data[flag_col].fillna(0).values.astype(np.float32)
                feature_arrays.append(flag_values)
                feature_names.append(flag_col)
        
        # Add salary range features
        if 'salary_range' in data.columns:
            salary = data['salary_range'].fillna('')
            has_salary = (salary.apply(len) > 0).astype(np.float32).values
            feature_arrays.append(has_salary)
            feature_names.append('has_salary')
        
        # Encode categorical columns as features
        for cat_col in ['employment_type', 'required_experience', 'required_education']:
            if cat_col in data.columns:
                encoded = pd.factorize(data[cat_col].fillna('unknown'))[0].astype(np.float32)
                feature_arrays.append(encoded)
                feature_names.append(f'{cat_col}_encoded')
        
        X = np.column_stack(feature_arrays).astype(np.float32)
        print(f"    Features: {X.shape[1]} columns ({feature_names[:5]}...)")
        
        # Build graph from categorical columns
        n_nodes = len(data)
        src_list, dst_list = [], []
        
        for col in graph_cols:
            if col in data.columns:
                n_unique = data[col].nunique()
                print(f"      Building edges from {col} ({n_unique} unique values)...")
                # Fill NA values for grouping
                col_data = data[col].fillna('_NA_')
                groups = col_data.groupby(col_data).indices
                edges_added = 0
                for group_name, idx_list in groups.items():
                    idx_array = np.array(list(idx_list))
                    if len(idx_array) > 1 and len(idx_array) < 500:
                        # Connect nodes in same group (limit connections for large groups)
                        n_connect = min(len(idx_array), 15)
                        for i in range(len(idx_array)):
                            # Connect to next few nodes in group
                            for j in range(1, min(n_connect, len(idx_array) - i)):
                                src_list.append(idx_array[i])
                                dst_list.append(idx_array[(i + j) % len(idx_array)])
                                src_list.append(idx_array[(i + j) % len(idx_array)])
                                dst_list.append(idx_array[i])
                                edges_added += 2
                print(f"        Added {edges_added:,} edges")
        
        # Add edges based on description length similarity
        if 'description_len' in feature_names:
            print(f"    Building edges based on description length proximity...")
            desc_idx = feature_names.index('description_len')
            desc_lengths = X[:, desc_idx]
            # Bin by description length
            try:
                length_bins = pd.qcut(desc_lengths, q=30, labels=False, duplicates='drop')
                len_edges = 0
                for bin_val in np.unique(length_bins):
                    bin_indices = np.where(length_bins == bin_val)[0]
                    if len(bin_indices) > 1 and len(bin_indices) < 200:
                        n_connect = min(len(bin_indices), 5)
                        for i in range(len(bin_indices)):
                            for j in range(1, min(n_connect, len(bin_indices) - i)):
                                src_list.append(bin_indices[i])
                                dst_list.append(bin_indices[i + j])
                                src_list.append(bin_indices[i + j])
                                dst_list.append(bin_indices[i])
                                len_edges += 2
                print(f"      Added {len_edges:,} description-length-based edges")
            except:
                pass
        
        if len(src_list) == 0:
            print(f"    WARNING: No edges created. Creating random edges.")
            src_list = np.random.randint(0, n_nodes, n_nodes * 5).tolist()
            dst_list = np.random.randint(0, n_nodes, n_nodes * 5).tolist()
        
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        print(f"    Total edges: {edge_index.shape[1]:,}")
        print(f"    Avg degree: {edge_index.shape[1] / n_nodes:.1f}")
        
        return {'X': X, 'y': y, 'edge_index': edge_index, 'name': 'Fake-Job-Postings'}
    
    except Exception as e:
        import traceback
        print(f"    Error loading Fake Job Postings: {e}")
        traceback.print_exc()
        return None


def load_vehicle_loan_default(vehicle_loan_path=None):
    """
    Load Vehicle Loan Default Prediction dataset.
    
    Expected columns:
    - UNIQUEID: unique loan ID
    - DISBURSED_AMOUNT, ASSET_COST, LTV: loan amount features
    - BRANCH_ID: branch identifier (used for graph)
    - SUPPLIER_ID: supplier identifier (used for graph)
    - MANUFACTURER_ID: manufacturer identifier (used for graph)
    - CURRENT_PINCODE_ID: pincode (used for graph)
    - STATE_ID: state identifier (used for graph)
    - EMPLOYEE_CODE_ID: employee code
    - DATE_OF_BIRTH, EMPLOYMENT_TYPE, DISBURSAL_DATE: demographic/temporal features
    - Various ID flags: MOBILENO_AVL_FLAG, AADHAR_FLAG, PAN_FLAG, etc.
    - PERFORM_CNS_SCORE: credit score
    - Credit history features: PRI_NO_OF_ACCTS, PRI_ACTIVE_ACCTS, etc.
    - AVERAGE_ACCT_AGE, CREDIT_HISTORY_LENGTH, NO_OF_INQUIRIES
    - LOAN_DEFAULT: label (0/1)
    """
    if vehicle_loan_path is None:
        for p in ['./data/vechile_loan_default_prediction/train.csv',
                  './data/vehicle_loan_default/train.csv',
                  './data/vehicle_loan/train.csv',
                  './train.csv', '../train.csv',
                  './data/train.csv']:
            if os.path.exists(p):
                vehicle_loan_path = p
                break
    
    if vehicle_loan_path is None or not os.path.exists(vehicle_loan_path):
        print(f"    Vehicle Loan Default file not found. Searched paths:")
        print(f"      ./data/vechile_loan_default_prediction/train.csv")
        print(f"      ./data/vehicle_loan_default/train.csv")
        print(f"    Please specify path with --vehicle_loan_path")
        return None
    
    try:
        print(f"    Loading from: {vehicle_loan_path}")
        data = pd.read_csv(vehicle_loan_path)
        print(f"    Loaded {len(data):,} rows, {len(data.columns)} columns")
        print(f"    Columns: {list(data.columns)[:15]}...")  # Truncate for readability
        
        # Sample if dataset is too large
        max_rows = 150000
        if len(data) > max_rows:
            print(f"    Sampling {max_rows:,} rows from {len(data):,} total...")
            data = data.sample(n=max_rows, random_state=42).reset_index(drop=True)
        
        # Identify label column
        label_col = None
        for col in ['LOAN_DEFAULT', 'loan_default', 'default', 'is_default', 'label', 'target']:
            if col in data.columns:
                label_col = col
                break
        
        if label_col is None:
            print(f"    ERROR: Could not find label column. Expected one of: LOAN_DEFAULT, loan_default, default, is_default")
            return None
        
        y = data[label_col].values.astype(np.int64)
        print(f"    Label column: {label_col}, Default rate: {y.mean():.2%}")
        
        # Identify ID and exclude columns
        id_cols = ['UNIQUEID', 'uniqueid', 'id', 'ID']
        exclude_cols = [label_col] + id_cols
        
        # Identify categorical columns for graph construction
        graph_cols = []
        for col in ['BRANCH_ID', 'SUPPLIER_ID', 'MANUFACTURER_ID', 'STATE_ID', 
                    'CURRENT_PINCODE_ID', 'EMPLOYEE_CODE_ID', 'EMPLOYMENT_TYPE']:
            if col in data.columns:
                # Only use columns with reasonable cardinality for graph
                n_unique = data[col].nunique()
                if n_unique > 1 and n_unique < len(data) * 0.5:
                    graph_cols.append(col)
        print(f"    Graph columns: {graph_cols}")
        
        # Prepare features
        feature_cols = [c for c in data.columns if c not in exclude_cols]
        
        # Handle date columns
        for date_col in ['DATE_OF_BIRTH', 'DISBURSAL_DATE']:
            if date_col in feature_cols:
                try:
                    dates = pd.to_datetime(data[date_col], dayfirst=True, errors='coerce')
                    if date_col == 'DATE_OF_BIRTH':
                        # Calculate age from DOB
                        reference_date = pd.to_datetime('2018-01-01')
                        data['age'] = (reference_date - dates).dt.days / 365.25
                        feature_cols.append('age')
                    elif date_col == 'DISBURSAL_DATE':
                        data['disbursal_month'] = dates.dt.month
                        data['disbursal_dow'] = dates.dt.dayofweek
                        feature_cols.extend(['disbursal_month', 'disbursal_dow'])
                    feature_cols.remove(date_col)
                except:
                    if date_col in feature_cols:
                        feature_cols.remove(date_col)
        
        # Handle AVERAGE_ACCT_AGE and CREDIT_HISTORY_LENGTH (format: "Xyrs Ymon")
        for age_col in ['AVERAGE_ACCT_AGE', 'CREDIT_HISTORY_LENGTH']:
            if age_col in feature_cols:
                try:
                    def parse_age(val):
                        if pd.isna(val) or val == '':
                            return 0
                        val = str(val)
                        years = 0
                        months = 0
                        if 'yrs' in val:
                            years = int(val.split('yrs')[0])
                        if 'mon' in val:
                            months = int(val.split('yrs')[-1].replace('mon', '').strip())
                        return years * 12 + months
                    data[f'{age_col}_months'] = data[age_col].apply(parse_age)
                    feature_cols.append(f'{age_col}_months')
                    feature_cols.remove(age_col)
                except:
                    if age_col in feature_cols:
                        feature_cols.remove(age_col)
        
        # Handle PERFORM_CNS_SCORE_DESCRIPTION (categorical)
        if 'PERFORM_CNS_SCORE_DESCRIPTION' in feature_cols:
            feature_cols.remove('PERFORM_CNS_SCORE_DESCRIPTION')
        
        # Encode categorical features
        for col in feature_cols:
            if col in data.columns:
                if data[col].dtype == 'object' or str(data[col].dtype) == 'category':
                    data[col] = pd.factorize(data[col])[0]
        
        # Filter to valid feature columns
        feature_cols = [c for c in feature_cols if c in data.columns]
        
        # Select numeric features only
        numeric_cols = []
        for col in feature_cols:
            if col in data.columns:
                try:
                    data[col].astype(np.float32)
                    numeric_cols.append(col)
                except:
                    pass
        
        X = data[numeric_cols].fillna(-999).values.astype(np.float32)
        print(f"    Features: {len(numeric_cols)} columns")
        
        # Build graph from categorical columns
        n_nodes = len(data)
        src_list, dst_list = [], []
        
        for col in graph_cols:
            if col in data.columns:
                n_unique = data[col].nunique()
                print(f"      Building edges from {col} ({n_unique} unique values)...")
                groups = data.groupby(col).indices
                edges_added = 0
                for group_name, idx_list in groups.items():
                    idx_array = np.array(list(idx_list))
                    if len(idx_array) > 1 and len(idx_array) < 500:
                        # Connect nodes in same group (limit connections for large groups)
                        n_connect = min(len(idx_array), 15)
                        for i in range(len(idx_array)):
                            # Connect to next few nodes in group
                            for j in range(1, min(n_connect, len(idx_array) - i)):
                                src_list.append(idx_array[i])
                                dst_list.append(idx_array[(i + j) % len(idx_array)])
                                src_list.append(idx_array[(i + j) % len(idx_array)])
                                dst_list.append(idx_array[i])
                                edges_added += 2
                print(f"        Added {edges_added:,} edges")
        
        # Add edges based on credit score proximity
        if 'PERFORM_CNS_SCORE' in data.columns:
            print(f"    Building edges based on credit score proximity...")
            score_values = data['PERFORM_CNS_SCORE'].values
            # Only for non-zero scores
            valid_scores = score_values > 0
            if valid_scores.sum() > 1000:
                valid_indices = np.where(valid_scores)[0]
                sorted_order = np.argsort(score_values[valid_indices])
                sorted_indices = valid_indices[sorted_order]
                k_neighbors = 3
                score_edges = 0
                for i in range(len(sorted_indices)):
                    for j in range(max(0, i - k_neighbors), min(len(sorted_indices), i + k_neighbors + 1)):
                        if i != j:
                            src_list.append(sorted_indices[i])
                            dst_list.append(sorted_indices[j])
                            score_edges += 1
                print(f"      Added {score_edges:,} score-proximity edges")
        
        if len(src_list) == 0:
            print(f"    WARNING: No edges created. Creating random edges.")
            src_list = np.random.randint(0, n_nodes, n_nodes * 5).tolist()
            dst_list = np.random.randint(0, n_nodes, n_nodes * 5).tolist()
        
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        print(f"    Total edges: {edge_index.shape[1]:,}")
        print(f"    Avg degree: {edge_index.shape[1] / n_nodes:.1f}")
        
        return {'X': X, 'y': y, 'edge_index': edge_index, 'name': 'Vehicle-Loan-Default'}
    
    except Exception as e:
        import traceback
        print(f"    Error loading Vehicle Loan Default: {e}")
        traceback.print_exc()
        return None


def load_ci_badguys(ci_badguys_path=None, ci_badguys_url=None):
    """
    Load CI-BadGuys IP dataset from CINSscore.com.
    
    This dataset contains a list of known malicious IP addresses.
    
    Source: https://cinsscore.com/list/ci-badguys.txt
    
    Features are extracted from the IP address structure:
    - Octets (4 values)
    - First octet class (A/B/C/D/E)
    - Private IP flag
    - Various octet ratios and patterns
    
    Graph is built by connecting IPs with:
    - Same /24 subnet
    - Same /16 subnet
    - Similar first octet (same class)
    
    Since all IPs in the list are malicious, we create a balanced dataset
    by generating synthetic benign IPs for comparison.
    """
    import urllib.request
    
    default_url = "https://cinsscore.com/list/ci-badguys.txt"
    
    if ci_badguys_path is None:
        for p in ['./data/ci_badguys/ci-badguys.txt',
                  './data/ci-badguys.txt',
                  './ci-badguys.txt', '../ci-badguys.txt',
                  './data/cinsscore/ci-badguys.txt']:
            if os.path.exists(p):
                ci_badguys_path = p
                break
    
    # Load data from file or URL
    try:
        if ci_badguys_path and os.path.exists(ci_badguys_path):
            print(f"    Loading from file: {ci_badguys_path}")
            with open(ci_badguys_path, 'r') as f:
                content = f.read()
        else:
            url = ci_badguys_url if ci_badguys_url else default_url
            print(f"    Downloading from: {url}")
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read().decode('utf-8')
            
            # Save to local file for future use
            save_dir = './data/ci_badguys'
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'ci-badguys.txt')
            with open(save_path, 'w') as f:
                f.write(content)
            print(f"    Saved to: {save_path}")
        
        # Parse IP addresses
        lines = content.strip().split('\n')
        malicious_ips = []
        for line in lines:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            # Validate IP format (basic check)
            parts = line.split('.')
            if len(parts) == 4:
                try:
                    octets = [int(p) for p in parts]
                    if all(0 <= o <= 255 for o in octets):
                        malicious_ips.append(line)
                except ValueError:
                    continue
        
        print(f"    Loaded {len(malicious_ips):,} malicious IPs")
        
        # Sample if too large
        max_malicious = 50000
        if len(malicious_ips) > max_malicious:
            print(f"    Sampling {max_malicious:,} malicious IPs from {len(malicious_ips):,} total...")
            np.random.seed(42)
            indices = np.random.choice(len(malicious_ips), max_malicious, replace=False)
            malicious_ips = [malicious_ips[i] for i in indices]
        
        # Generate synthetic benign IPs to create balanced dataset
        # Use common benign IP ranges (avoiding known malicious patterns)
        print(f"    Generating synthetic benign IPs for balanced dataset...")
        np.random.seed(42)
        n_benign = len(malicious_ips)
        benign_ips = []
        
        # Common benign IP ranges (simplified)
        benign_ranges = [
            # Common ISP ranges, cloud providers, CDNs
            (1, 50),    # Class A low
            (64, 95),   # Various ISPs
            (128, 170), # Class B
            (172, 191), # Mixed
            (192, 223), # Class C
        ]
        
        malicious_set = set(malicious_ips)
        while len(benign_ips) < n_benign:
            # Pick a random benign range
            range_start, range_end = benign_ranges[np.random.randint(len(benign_ranges))]
            o1 = np.random.randint(range_start, range_end + 1)
            o2 = np.random.randint(0, 256)
            o3 = np.random.randint(0, 256)
            o4 = np.random.randint(1, 255)  # Avoid .0 and .255
            
            ip = f"{o1}.{o2}.{o3}.{o4}"
            if ip not in malicious_set and ip not in benign_ips:
                benign_ips.append(ip)
        
        print(f"    Generated {len(benign_ips):,} synthetic benign IPs")
        
        # Combine into dataset
        all_ips = malicious_ips + benign_ips
        y = np.array([1] * len(malicious_ips) + [0] * len(benign_ips), dtype=np.int64)
        
        # Shuffle
        np.random.seed(42)
        shuffle_idx = np.random.permutation(len(all_ips))
        all_ips = [all_ips[i] for i in shuffle_idx]
        y = y[shuffle_idx]
        
        print(f"    Total dataset: {len(all_ips):,} IPs, Malicious rate: {y.mean():.2%}")
        
        # Extract features from IPs
        print(f"    Extracting features from IP addresses...")
        feature_arrays = []
        
        # Parse all IPs into octets
        octets_array = np.zeros((len(all_ips), 4), dtype=np.float32)
        for i, ip in enumerate(all_ips):
            parts = ip.split('.')
            for j in range(4):
                octets_array[i, j] = int(parts[j])
        
        # Basic octet features
        feature_arrays.append(octets_array)  # 4 features
        
        # Octet statistics
        feature_arrays.append(octets_array.sum(axis=1, keepdims=True))  # Sum of octets
        feature_arrays.append(octets_array.mean(axis=1, keepdims=True))  # Mean
        feature_arrays.append(octets_array.std(axis=1, keepdims=True))   # Std
        feature_arrays.append(octets_array.max(axis=1, keepdims=True))   # Max
        feature_arrays.append(octets_array.min(axis=1, keepdims=True))   # Min
        
        # First octet class (A=0-127, B=128-191, C=192-223, D=224-239, E=240-255)
        o1 = octets_array[:, 0]
        class_a = (o1 < 128).astype(np.float32).reshape(-1, 1)
        class_b = ((o1 >= 128) & (o1 < 192)).astype(np.float32).reshape(-1, 1)
        class_c = ((o1 >= 192) & (o1 < 224)).astype(np.float32).reshape(-1, 1)
        class_d = ((o1 >= 224) & (o1 < 240)).astype(np.float32).reshape(-1, 1)
        class_e = (o1 >= 240).astype(np.float32).reshape(-1, 1)
        feature_arrays.extend([class_a, class_b, class_c, class_d, class_e])
        
        # Private IP ranges
        # 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
        is_private_10 = (o1 == 10)
        is_private_172 = ((o1 == 172) & (octets_array[:, 1] >= 16) & (octets_array[:, 1] <= 31))
        is_private_192 = ((o1 == 192) & (octets_array[:, 1] == 168))
        is_private = (is_private_10 | is_private_172 | is_private_192).astype(np.float32).reshape(-1, 1)
        feature_arrays.append(is_private)
        
        # Loopback and special ranges
        is_loopback = (o1 == 127).astype(np.float32).reshape(-1, 1)
        is_multicast = ((o1 >= 224) & (o1 <= 239)).astype(np.float32).reshape(-1, 1)
        feature_arrays.extend([is_loopback, is_multicast])
        
        # Numeric IP value (normalized)
        ip_value = (octets_array[:, 0] * 256**3 + octets_array[:, 1] * 256**2 + 
                   octets_array[:, 2] * 256 + octets_array[:, 3])
        ip_value_norm = (ip_value / (256**4)).astype(np.float32).reshape(-1, 1)
        feature_arrays.append(ip_value_norm)
        
        # Subnet identifiers (normalized)
        subnet_24 = (octets_array[:, 0] * 256**2 + octets_array[:, 1] * 256 + octets_array[:, 2])
        subnet_24_norm = (subnet_24 / (256**3)).astype(np.float32).reshape(-1, 1)
        subnet_16 = (octets_array[:, 0] * 256 + octets_array[:, 1])
        subnet_16_norm = (subnet_16 / (256**2)).astype(np.float32).reshape(-1, 1)
        feature_arrays.extend([subnet_24_norm, subnet_16_norm])
        
        # Octet ratios
        o1_ratio = (octets_array[:, 0] / 255).astype(np.float32).reshape(-1, 1)
        o2_ratio = (octets_array[:, 1] / 255).astype(np.float32).reshape(-1, 1)
        o3_ratio = (octets_array[:, 2] / 255).astype(np.float32).reshape(-1, 1)
        o4_ratio = (octets_array[:, 3] / 255).astype(np.float32).reshape(-1, 1)
        feature_arrays.extend([o1_ratio, o2_ratio, o3_ratio, o4_ratio])
        
        # Even/odd patterns
        o1_even = (octets_array[:, 0] % 2 == 0).astype(np.float32).reshape(-1, 1)
        o4_even = (octets_array[:, 3] % 2 == 0).astype(np.float32).reshape(-1, 1)
        feature_arrays.extend([o1_even, o4_even])
        
        X = np.concatenate(feature_arrays, axis=1).astype(np.float32)
        print(f"    Features: {X.shape[1]} columns")
        
        # Build graph based on subnet similarity
        print(f"    Building graph based on subnet similarity...")
        n_nodes = len(all_ips)
        src_list, dst_list = [], []
        
        # Group by /24 subnet
        subnet_24_groups = {}
        for i in range(n_nodes):
            key = (int(octets_array[i, 0]), int(octets_array[i, 1]), int(octets_array[i, 2]))
            if key not in subnet_24_groups:
                subnet_24_groups[key] = []
            subnet_24_groups[key].append(i)
        
        edges_24 = 0
        for subnet, indices in subnet_24_groups.items():
            if len(indices) > 1 and len(indices) < 100:
                n_connect = min(len(indices), 10)
                for i in range(len(indices)):
                    for j in range(1, min(n_connect, len(indices) - i)):
                        src_list.append(indices[i])
                        dst_list.append(indices[i + j])
                        src_list.append(indices[i + j])
                        dst_list.append(indices[i])
                        edges_24 += 2
        print(f"      Added {edges_24:,} /24 subnet edges")
        
        # Group by /16 subnet
        subnet_16_groups = {}
        for i in range(n_nodes):
            key = (int(octets_array[i, 0]), int(octets_array[i, 1]))
            if key not in subnet_16_groups:
                subnet_16_groups[key] = []
            subnet_16_groups[key].append(i)
        
        edges_16 = 0
        for subnet, indices in subnet_16_groups.items():
            if len(indices) > 1 and len(indices) < 200:
                n_connect = min(len(indices), 5)
                for i in range(len(indices)):
                    for j in range(1, min(n_connect, len(indices) - i)):
                        # Avoid duplicate edges from /24
                        if (int(octets_array[indices[i], 2]) != int(octets_array[indices[i + j], 2])):
                            src_list.append(indices[i])
                            dst_list.append(indices[i + j])
                            src_list.append(indices[i + j])
                            dst_list.append(indices[i])
                            edges_16 += 2
        print(f"      Added {edges_16:,} /16 subnet edges")
        
        # Group by first octet (Class)
        class_groups = {}
        for i in range(n_nodes):
            key = int(octets_array[i, 0])
            if key not in class_groups:
                class_groups[key] = []
            class_groups[key].append(i)
        
        edges_class = 0
        for octet, indices in class_groups.items():
            if len(indices) > 1 and len(indices) < 500:
                n_connect = min(len(indices), 3)
                for i in range(len(indices)):
                    for j in range(1, min(n_connect, len(indices) - i)):
                        # Avoid duplicate edges
                        if (int(octets_array[indices[i], 1]) != int(octets_array[indices[i + j], 1])):
                            src_list.append(indices[i])
                            dst_list.append(indices[i + j])
                            src_list.append(indices[i + j])
                            dst_list.append(indices[i])
                            edges_class += 2
        print(f"      Added {edges_class:,} first-octet class edges")
        
        if len(src_list) == 0:
            print(f"    WARNING: No edges created. Creating random edges.")
            src_list = np.random.randint(0, n_nodes, n_nodes * 5).tolist()
            dst_list = np.random.randint(0, n_nodes, n_nodes * 5).tolist()
        
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        print(f"    Total edges: {edge_index.shape[1]:,}")
        print(f"    Avg degree: {edge_index.shape[1] / n_nodes:.1f}")
        
        return {'X': X, 'y': y, 'edge_index': edge_index, 'name': 'CI-BadGuys'}
    
    except Exception as e:
        import traceback
        print(f"    Error loading CI-BadGuys: {e}")
        traceback.print_exc()
        return None


def load_malicious_urls(malicious_urls_path=None):
    """
    Load Malicious URLs dataset.
    
    Expected columns:
    - url: the URL string
    - type: classification type (benign, phishing, malware, defacement)
    
    Features are extracted from the URL structure:
    - URL length, path length, query length
    - Count of special characters (., -, @, /, etc.)
    - Presence of IP address
    - Domain features (TLD, subdomain count)
    
    Graph is built by connecting URLs with similar domains or TLDs.
    """
    if malicious_urls_path is None:
        for p in ['./data/malicious_URLs_dataset/malicious_phish.csv',
                  './data/malicious_urls/malicious_phish.csv',
                  './malicious_phish.csv', '../malicious_phish.csv',
                  './data/malicious_phish.csv']:
            if os.path.exists(p):
                malicious_urls_path = p
                break
    
    if malicious_urls_path is None or not os.path.exists(malicious_urls_path):
        print(f"    Malicious URLs file not found. Searched paths:")
        print(f"      ./data/malicious_URLs_dataset/malicious_phish.csv")
        print(f"      ./data/malicious_urls/malicious_phish.csv")
        print(f"    Please specify path with --malicious_urls_path")
        return None
    
    try:
        print(f"    Loading from: {malicious_urls_path}")
        data = pd.read_csv(malicious_urls_path)
        print(f"    Loaded {len(data):,} rows, {len(data.columns)} columns")
        print(f"    Columns: {list(data.columns)}")
        
        # Sample if dataset is too large
        max_rows = 100000
        if len(data) > max_rows:
            print(f"    Sampling {max_rows:,} rows from {len(data):,} total...")
            data = data.sample(n=max_rows, random_state=42).reset_index(drop=True)
        
        # Identify label column
        label_col = None
        for col in ['type', 'label', 'class', 'is_malicious', 'malicious', 'target']:
            if col in data.columns:
                label_col = col
                break
        
        if label_col is None:
            print(f"    ERROR: Could not find label column. Expected one of: type, label, class, is_malicious")
            return None
        
        # Convert multi-class to binary (malicious=1, benign=0)
        if data[label_col].dtype == 'object':
            malicious_types = ['phishing', 'malware', 'defacement', 'spam', 'Malicious', 'Phishing', 'Malware']
            y = data[label_col].apply(lambda x: 1 if x in malicious_types else 0).values.astype(np.int64)
        else:
            y = data[label_col].values.astype(np.int64)
        
        print(f"    Label column: {label_col}, Malicious rate: {y.mean():.2%}")
        print(f"    Type distribution: {data[label_col].value_counts().to_dict()}")
        
        # Extract features from URLs
        print(f"    Extracting features from URLs...")
        urls = data['url'].astype(str).values
        
        # URL length features
        url_length = np.array([len(u) for u in urls], dtype=np.float32)
        
        # Count special characters
        dot_count = np.array([u.count('.') for u in urls], dtype=np.float32)
        dash_count = np.array([u.count('-') for u in urls], dtype=np.float32)
        slash_count = np.array([u.count('/') for u in urls], dtype=np.float32)
        at_count = np.array([u.count('@') for u in urls], dtype=np.float32)
        question_count = np.array([u.count('?') for u in urls], dtype=np.float32)
        equal_count = np.array([u.count('=') for u in urls], dtype=np.float32)
        ampersand_count = np.array([u.count('&') for u in urls], dtype=np.float32)
        underscore_count = np.array([u.count('_') for u in urls], dtype=np.float32)
        
        # Digit and letter counts
        digit_count = np.array([sum(c.isdigit() for c in u) for u in urls], dtype=np.float32)
        letter_count = np.array([sum(c.isalpha() for c in u) for u in urls], dtype=np.float32)
        
        # Has https, has www
        has_https = np.array([1 if 'https' in u.lower() else 0 for u in urls], dtype=np.float32)
        has_http = np.array([1 if 'http' in u.lower() else 0 for u in urls], dtype=np.float32)
        has_www = np.array([1 if 'www' in u.lower() else 0 for u in urls], dtype=np.float32)
        
        # Has IP address pattern
        import re
        ip_pattern = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
        has_ip = np.array([1 if ip_pattern.search(u) else 0 for u in urls], dtype=np.float32)
        
        # Path and query features
        path_length = np.zeros(len(urls), dtype=np.float32)
        query_length = np.zeros(len(urls), dtype=np.float32)
        for i, u in enumerate(urls):
            if '?' in u:
                parts = u.split('?', 1)
                path_length[i] = len(parts[0])
                query_length[i] = len(parts[1])
            else:
                path_length[i] = len(u)
        
        # Domain extraction for graph building
        domains = []
        tlds = []
        for u in urls:
            # Remove protocol
            u_clean = u.lower().replace('https://', '').replace('http://', '').replace('www.', '')
            # Get domain
            domain = u_clean.split('/')[0].split('?')[0]
            domains.append(domain)
            # Get TLD
            parts = domain.split('.')
            tld = parts[-1] if len(parts) > 1 else 'none'
            tlds.append(tld)
        
        X = np.column_stack([
            url_length, path_length, query_length,
            dot_count, dash_count, slash_count, at_count, question_count,
            equal_count, ampersand_count, underscore_count,
            digit_count, letter_count,
            has_https, has_http, has_www, has_ip,
            digit_count / (url_length + 1),  # digit ratio
            letter_count / (url_length + 1),  # letter ratio
        ]).astype(np.float32)
        
        print(f"    Features: {X.shape[1]} columns")
        
        # Build graph based on TLD and domain similarity
        n_nodes = len(data)
        src_list, dst_list = [], []
        
        # Group by TLD
        print(f"    Building edges based on TLD ({len(set(tlds))} unique)...")
        tld_groups = {}
        for i, tld in enumerate(tlds):
            if tld not in tld_groups:
                tld_groups[tld] = []
            tld_groups[tld].append(i)
        
        tld_edges = 0
        for tld, indices in tld_groups.items():
            if len(indices) > 1 and len(indices) < 500:
                n_connect = min(len(indices), 10)
                for i in range(len(indices)):
                    for j in range(1, min(n_connect, len(indices) - i)):
                        src_list.append(indices[i])
                        dst_list.append(indices[i + j])
                        src_list.append(indices[i + j])
                        dst_list.append(indices[i])
                        tld_edges += 2
        print(f"      Added {tld_edges:,} TLD-based edges")
        
        # Group by URL length bins
        print(f"    Building edges based on URL length proximity...")
        length_bins = pd.qcut(url_length, q=50, labels=False, duplicates='drop')
        length_edges = 0
        for bin_val in np.unique(length_bins):
            bin_indices = np.where(length_bins == bin_val)[0]
            if len(bin_indices) > 1 and len(bin_indices) < 200:
                n_connect = min(len(bin_indices), 5)
                for i in range(len(bin_indices)):
                    for j in range(1, min(n_connect, len(bin_indices) - i)):
                        src_list.append(bin_indices[i])
                        dst_list.append(bin_indices[i + j])
                        src_list.append(bin_indices[i + j])
                        dst_list.append(bin_indices[i])
                        length_edges += 2
        print(f"      Added {length_edges:,} length-based edges")
        
        if len(src_list) == 0:
            print(f"    WARNING: No edges created. Creating random edges.")
            src_list = np.random.randint(0, n_nodes, n_nodes * 5).tolist()
            dst_list = np.random.randint(0, n_nodes, n_nodes * 5).tolist()
        
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        print(f"    Total edges: {edge_index.shape[1]:,}")
        print(f"    Avg degree: {edge_index.shape[1] / n_nodes:.1f}")
        
        return {'X': X, 'y': y, 'edge_index': edge_index, 'name': 'Malicious-URLs'}
    
    except Exception as e:
        import traceback
        print(f"    Error loading Malicious URLs: {e}")
        traceback.print_exc()
        return None


def load_ecommerce(ecommerce_path=None):
    """
    Load Ecommerce fraud dataset.
    
    Expected columns:
    - user_id: user identifier (used for graph)
    - signup_time: signup datetime
    - purchase_time: purchase datetime
    - purchase_value: transaction amount
    - device_id: device identifier (used for graph)
    - source: traffic source (used for graph)
    - browser: browser type (used for graph)
    - sex: gender
    - age: age
    - ip_address: IP address
    - class: label (0/1)
    """
    if ecommerce_path is None:
        for p in ['./data/ecommerce_fraud/Fraud_Data.csv', './data/ecommerce_fraud/fraud_data.csv',
                  './data/ecommerce/Fraud_Data.csv', './Fraud_Data.csv', '../Fraud_Data.csv',
                  './data/Fraud_Data.csv', '../data/Fraud_Data.csv']:
            if os.path.exists(p):
                ecommerce_path = p
                break
    
    if ecommerce_path is None or not os.path.exists(ecommerce_path):
        print(f"    Ecommerce Fraud file not found. Searched paths:")
        print(f"      ./data/ecommerce_fraud/Fraud_Data.csv")
        print(f"      ./data/ecommerce/Fraud_Data.csv")
        print(f"      ./Fraud_Data.csv")
        print(f"    Please specify path with --ecommerce_path")
        return None
    
    try:
        print(f"    Loading from: {ecommerce_path}")
        data = pd.read_csv(ecommerce_path)
        print(f"    Loaded {len(data):,} rows, {len(data.columns)} columns")
        print(f"    Columns: {list(data.columns)}")
        
        # Identify label column
        label_col = None
        for col in ['class', 'Class', 'is_fraud', 'isFraud', 'fraud', 'label', 'target']:
            if col in data.columns:
                label_col = col
                break
        
        if label_col is None:
            print(f"    ERROR: Could not find label column. Expected one of: class, is_fraud, fraud, label, target")
            return None
        
        y = data[label_col].values.astype(np.int64)
        print(f"    Label column: {label_col}, Fraud rate: {y.mean():.2%}")
        
        # Identify ID and exclude columns
        id_cols = ['transaction_id', 'id', 'ID', 'TransactionID']
        exclude_cols = [label_col] + id_cols
        
        # Identify categorical columns for graph construction
        graph_cols = []
        for col in ['user_id', 'device_id', 'source', 'browser', 'ip_address']:
            if col in data.columns:
                graph_cols.append(col)
        print(f"    Graph columns: {graph_cols}")
        
        # Prepare features
        feature_cols = [c for c in data.columns if c not in exclude_cols]
        
        # Handle datetime columns
        for time_col in ['signup_time', 'purchase_time']:
            if time_col in feature_cols:
                try:
                    dates = pd.to_datetime(data[time_col])
                    data[f'{time_col}_hour'] = dates.dt.hour
                    data[f'{time_col}_dow'] = dates.dt.dayofweek
                    data[f'{time_col}_day'] = dates.dt.day
                    data[f'{time_col}_month'] = dates.dt.month
                    feature_cols.remove(time_col)
                    feature_cols.extend([f'{time_col}_hour', f'{time_col}_dow', f'{time_col}_day', f'{time_col}_month'])
                except:
                    feature_cols.remove(time_col)
        
        # Compute time difference between signup and purchase
        if 'signup_time' in data.columns and 'purchase_time' in data.columns:
            try:
                signup = pd.to_datetime(data['signup_time'])
                purchase = pd.to_datetime(data['purchase_time'])
                data['time_diff_seconds'] = (purchase - signup).dt.total_seconds()
                feature_cols.append('time_diff_seconds')
            except:
                pass
        
        # Encode categorical features
        for col in feature_cols:
            if col in data.columns:
                if data[col].dtype == 'object' or str(data[col].dtype) == 'category':
                    data[col] = pd.factorize(data[col])[0]
        
        # Filter to valid feature columns that exist
        feature_cols = [c for c in feature_cols if c in data.columns]
        X = data[feature_cols].fillna(-999).values.astype(np.float32)
        print(f"    Features: {len(feature_cols)} columns")
        
        # Build graph from categorical columns
        n_nodes = len(data)
        src_list, dst_list = [], []
        
        for col in graph_cols:
            if col in data.columns:
                print(f"      Building edges from {col} ({data[col].nunique()} unique values)...")
                groups = data.groupby(col).indices
                edges_added = 0
                for group_name, idx_list in groups.items():
                    idx_array = np.array(list(idx_list))
                    if len(idx_array) > 1 and len(idx_array) < 1000:
                        # Connect nodes in same group (limit connections for large groups)
                        n_connect = min(len(idx_array), 20)
                        for i in range(len(idx_array)):
                            # Connect to next few nodes in group
                            for j in range(1, min(n_connect, len(idx_array) - i)):
                                src_list.append(idx_array[i])
                                dst_list.append(idx_array[(i + j) % len(idx_array)])
                                src_list.append(idx_array[(i + j) % len(idx_array)])
                                dst_list.append(idx_array[i])
                                edges_added += 2
                print(f"        Added {edges_added:,} edges")
        
        if len(src_list) == 0:
            print(f"    WARNING: No edges created from graph columns. Creating random edges.")
            src_list = np.random.randint(0, n_nodes, n_nodes * 5).tolist()
            dst_list = np.random.randint(0, n_nodes, n_nodes * 5).tolist()
        
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        print(f"    Total edges: {edge_index.shape[1]:,}")
        print(f"    Avg degree: {edge_index.shape[1] / n_nodes:.1f}")
        
        return {'X': X, 'y': y, 'edge_index': edge_index, 'name': 'Ecommerce'}
    
    except Exception as e:
        import traceback
        print(f"    Error loading Ecommerce: {e}")
        traceback.print_exc()
        return None


def load_customs(customs_path=None, customs_url=None):
    """
    Load Customs fraud dataset (Synthetic Import Declarations).
    
    Source: https://raw.githubusercontent.com/Roytsai27/Dual-Attentive-Tree-aware-Embedding/refs/heads/master/data/synthetic-imports-declarations.csv
    
    Expected columns:
    - sgd.id: transaction ID
    - sgd.date: date
    - importer.id: importer identifier (used for graph)
    - declarant.id: declarant identifier (used for graph)
    - country: origin country (used for graph)
    - office.id: customs office
    - tariff.code: HS/tariff code (used for graph)
    - quantity: quantity
    - gross.weight: weight
    - fob.value: FOB value
    - cif.value: CIF value
    - total.taxes: taxes
    - illicit: label (0/1)
    - revenue: revenue
    """
    import urllib.request
    
    default_url = "https://raw.githubusercontent.com/Roytsai27/Dual-Attentive-Tree-aware-Embedding/refs/heads/master/data/synthetic-imports-declarations.csv"
    
    if customs_path is None:
        for p in ['./data/customs/customs.csv', './data/customs/synthetic-imports-declarations.csv',
                  './customs.csv', '../customs.csv', 
                  './data/customs.csv', '../data/customs.csv',
                  './data/synthetic-imports-declarations.csv']:
            if os.path.exists(p):
                customs_path = p
                break
    
    # Download if not found locally
    if customs_path is None or not os.path.exists(customs_path):
        url = customs_url if customs_url else default_url
        print(f"    Customs file not found locally. Downloading from:")
        print(f"      {url}")
        
        try:
            save_dir = './data/customs'
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'synthetic-imports-declarations.csv')
            
            urllib.request.urlretrieve(url, save_path)
            print(f"    Downloaded and saved to: {save_path}")
            customs_path = save_path
        except Exception as e:
            print(f"    ERROR: Failed to download customs dataset: {e}")
            print(f"    Please download manually from: {url}")
            print(f"    And place it in ./data/customs/")
            return None
    
    try:
        print(f"    Loading from: {customs_path}")
        data = pd.read_csv(customs_path)
        print(f"    Loaded {len(data):,} rows, {len(data.columns)} columns")
        print(f"    Columns: {list(data.columns)}")
        
        # Identify label column
        label_col = None
        for col in ['illicit', 'fraud', 'label', 'is_fraud', 'target']:
            if col in data.columns:
                label_col = col
                break
        
        if label_col is None:
            print(f"    ERROR: Could not find label column. Expected one of: illicit, fraud, label, is_fraud, target")
            return None
        
        y = data[label_col].values.astype(np.int64)
        print(f"    Label column: {label_col}, Fraud rate: {y.mean():.2%}")
        
        # Identify ID and exclude columns
        id_cols = ['sgd.id', 'id', 'ID', 'TransactionID', 'transaction_id']
        exclude_cols = [label_col] + id_cols
        
        # Identify categorical columns for graph construction
        graph_cols = []
        for col in ['importer.id', 'declarant.id', 'country', 'office.id', 'tariff.code',
                    'importer_id', 'declarant_id', 'hs_code', 'HS6']:
            if col in data.columns:
                graph_cols.append(col)
        print(f"    Graph columns: {graph_cols}")
        
        # Prepare features
        feature_cols = [c for c in data.columns if c not in exclude_cols]
        
        # Handle date column
        if 'sgd.date' in feature_cols:
            try:
                dates = pd.to_datetime(data['sgd.date'])
                data['month'] = dates.dt.month
                data['day_of_week'] = dates.dt.dayofweek
                data['day_of_month'] = dates.dt.day
                feature_cols.remove('sgd.date')
                feature_cols.extend(['month', 'day_of_week', 'day_of_month'])
            except:
                feature_cols.remove('sgd.date')
        
        # Encode categorical features
        for col in feature_cols:
            if data[col].dtype == 'object' or str(data[col].dtype) == 'category':
                data[col] = pd.factorize(data[col])[0]
        
        X = data[feature_cols].fillna(-999).values.astype(np.float32)
        print(f"    Features: {len(feature_cols)} columns")
        
        # Build graph from categorical columns
        n_nodes = len(data)
        src_list, dst_list = [], []
        
        for col in graph_cols:
            if col in data.columns:
                print(f"      Building edges from {col} ({data[col].nunique()} unique values)...")
                groups = data.groupby(col).indices
                edges_added = 0
                for group_name, idx_list in groups.items():
                    idx_array = np.array(list(idx_list))
                    if len(idx_array) > 1 and len(idx_array) < 1000:
                        # Connect nodes in same group (limit connections for large groups)
                        n_connect = min(len(idx_array), 20)
                        for i in range(len(idx_array)):
                            # Connect to next few nodes in group
                            for j in range(1, min(n_connect, len(idx_array) - i)):
                                src_list.append(idx_array[i])
                                dst_list.append(idx_array[(i + j) % len(idx_array)])
                                src_list.append(idx_array[(i + j) % len(idx_array)])
                                dst_list.append(idx_array[i])
                                edges_added += 2
                print(f"        Added {edges_added:,} edges")
        
        if len(src_list) == 0:
            print(f"    WARNING: No edges created from graph columns. Creating random edges.")
            src_list = np.random.randint(0, n_nodes, n_nodes * 5).tolist()
            dst_list = np.random.randint(0, n_nodes, n_nodes * 5).tolist()
        
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        print(f"    Total edges: {edge_index.shape[1]:,}")
        print(f"    Avg degree: {edge_index.shape[1] / n_nodes:.1f}")
        
        return {'X': X, 'y': y, 'edge_index': edge_index, 'name': 'Customs'}
    
    except Exception as e:
        import traceback
        print(f"    Error loading Customs: {e}")
        traceback.print_exc()
        return None


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_dataset(data):
    """Compute all metrics for a dataset"""
    X = data['X']
    y = data['y']
    edge_index = data['edge_index']
    labeled_mask = data.get('labeled_mask', None)
    
    if labeled_mask is None:
        labeled_mask = (y == 0) | (y == 1)
    
    n_nodes = len(y)
    n_edges = edge_index.shape[1] if isinstance(edge_index, torch.Tensor) else len(edge_index[0])
    
    results = {}
    
    # Basic stats
    results['Nodes'] = n_nodes
    results['Edges'] = n_edges
    results['Features'] = X.shape[1]
    results['Labeled'] = labeled_mask.sum() if isinstance(labeled_mask, np.ndarray) else labeled_mask.numpy().sum()
    
    # Class imbalance
    y_labeled = y[labeled_mask] if isinstance(labeled_mask, np.ndarray) else y[labeled_mask.numpy()]
    results['Fraud Rate'] = (y_labeled == 1).mean()
    results['Imbalance Ratio'] = (y_labeled == 0).sum() / max(1, (y_labeled == 1).sum())
    
    # Homophily metrics
    print("    Computing homophily...")
    results['Homophily'] = compute_homophily(edge_index, y, labeled_mask)
    results['Adj. Homophily'] = compute_adjusted_homophily(edge_index, y, labeled_mask)
    results['Fraud Homophily'] = compute_class_homophily(edge_index, y, target_class=1, labeled_mask=labeled_mask)
    
    # Feature predictiveness
    print("    Computing feature predictiveness...")
    results['Feature Pred (LR)'] = compute_feature_predictiveness(X, y, labeled_mask)
    results['Feature Pred (RF)'] = compute_feature_predictiveness_rf(X, y, labeled_mask)
    
    # Graph utility
    print("    Computing graph utility...")
    results['Graph Utility'] = compute_graph_utility(X, edge_index, y, labeled_mask)
    
    # Density
    results['Density'] = compute_density(edge_index, n_nodes)
    
    # Degree stats
    print("    Computing degree statistics...")
    degree_stats = compute_degree_stats(edge_index, n_nodes)
    results['Avg Degree'] = degree_stats['mean']
    results['Max Degree'] = degree_stats['max']
    results['Isolated Nodes'] = degree_stats['isolated']
    
    # Clustering
    print("    Computing clustering coefficient...")
    results['Clustering Coef'] = compute_clustering_coefficient(edge_index, n_nodes)
    
    # Label propagation
    print("    Computing LP accuracy...")
    results['LP Accuracy'] = compute_label_propagation_accuracy(edge_index, y, labeled_mask)
    
    # Feature-label correlation
    print("    Computing feature-label correlation...")
    results['Feat-Label Corr'] = compute_feature_label_correlation(X, y, labeled_mask)
    
    # Neighbor entropy
    print("    Computing neighbor entropy...")
    results['Neighbor Entropy'] = compute_neighbor_label_entropy(edge_index, y, labeled_mask)
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze dataset characteristics')
    parser.add_argument('--ieee_path', type=str, default=None, 
                        help='Path to IEEE-CIS data directory')
    parser.add_argument('--customs_path', type=str, default=None,
                        help='Path to customs CSV file')
    parser.add_argument('--credit_card_path', type=str, default=None,
                        help='Path to credit card fraud CSV file')
    parser.add_argument('--ecommerce_path', type=str, default=None,
                        help='Path to ecommerce fraud CSV file')
    parser.add_argument('--cc_transactions_path', type=str, default=None,
                        help='Path to credit card transactions fraud CSV file')
    parser.add_argument('--twitter_bots_path', type=str, default=None,
                        help='Path to Twitter bots dataset CSV file')
    parser.add_argument('--malicious_urls_path', type=str, default=None,
                        help='Path to malicious URLs dataset CSV file')
    parser.add_argument('--vehicle_loan_path', type=str, default=None,
                        help='Path to vehicle loan default prediction CSV file')
    parser.add_argument('--fake_job_path', type=str, default=None,
                        help='Path to fake job postings CSV file')
    parser.add_argument('--ci_badguys_path', type=str, default=None,
                        help='Path to CI-BadGuys IP list file')
    parser.add_argument('--ci_badguys_url', type=str, default=None,
                        help='URL to download CI-BadGuys IP list (default: https://cinsscore.com/list/ci-badguys.txt)')
    parser.add_argument('--generate_customs', action='store_true',
                        help='Generate synthetic customs data if not found')
    parser.add_argument('--datasets', type=str, default='all',
                        help='Comma-separated list of datasets to analyze (or "all")')
    args = parser.parse_args()
    
    print("="*120)
    print("DATASET CHARACTERISTICS ANALYSIS")
    print("="*120)
    
    # Generate synthetic customs if requested
    if args.generate_customs:
        print("\nGenerating synthetic customs data...")
        try:
            from generate_customs_data import generate_customs_data
            generate_customs_data(output_path='./data/customs/customs.csv')
        except Exception as e:
            print(f"Could not generate customs data: {e}")
    
    # Data loaders with custom paths
    loaders = {
        'Elliptic': load_elliptic,
        'Amazon': load_amazon,
        'Yelp': load_yelp,
        'Bitcoin-OTC': lambda: load_bitcoin('otc'),
        'Bitcoin-Alpha': lambda: load_bitcoin('alpha'),
        'IEEE-CIS': lambda: load_ieee_cis(args.ieee_path),
        'Customs': lambda: load_customs(args.customs_path),
        'Credit-Card': lambda: load_credit_card(args.credit_card_path),
        'Ecommerce': lambda: load_ecommerce(args.ecommerce_path),
        'CC-Transactions': lambda: load_cc_transactions(args.cc_transactions_path),
        'Twitter-Bots': lambda: load_twitter_bots(args.twitter_bots_path),
        'Malicious-URLs': lambda: load_malicious_urls(args.malicious_urls_path),
        'Vehicle-Loan-Default': lambda: load_vehicle_loan_default(args.vehicle_loan_path),
        'Fake-Job-Postings': lambda: load_fake_job_postings(args.fake_job_path),
        'CI-BadGuys': lambda: load_ci_badguys(args.ci_badguys_path, args.ci_badguys_url),
    }
    
    # Filter datasets if specified
    if args.datasets.lower() != 'all':
        requested = [d.strip() for d in args.datasets.split(',')]
        loaders = {k: v for k, v in loaders.items() if k.lower() in [r.lower() for r in requested]}
    
    all_results = []
    
    for name, loader in loaders.items():
        print(f"\n{'='*80}")
        print(f"Analyzing: {name}")
        print('='*80)
        
        data = loader()
        if data is None:
            print(f"  Could not load {name}")
            continue
        
        results = analyze_dataset(data)
        results['Dataset'] = name
        all_results.append(results)
        
        print(f"\n  Results for {name}:")
        for key, value in results.items():
            if key == 'Dataset':
                continue
            if isinstance(value, float):
                print(f"    {key:<20}: {value:.4f}")
            else:
                print(f"    {key:<20}: {value}")
    
    # Create summary table
    if len(all_results) > 0:
        df = pd.DataFrame(all_results)
        
        # Reorder columns
        cols = ['Dataset', 'Nodes', 'Edges', 'Features', 'Fraud Rate', 'Imbalance Ratio',
                'Homophily', 'Adj. Homophily', 'Fraud Homophily',
                'Feature Pred (LR)', 'Feature Pred (RF)', 'Graph Utility',
                'Density', 'Avg Degree', 'Clustering Coef',
                'LP Accuracy', 'Feat-Label Corr', 'Neighbor Entropy']
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
        
        print("\n" + "="*120)
        print("SUMMARY TABLE")
        print("="*120)
        
        # Print key metrics
        print("\n" + "-"*100)
        print(f"{'Dataset':<15} {'Homophily':>10} {'Feat Pred':>10} {'Graph Util':>12} {'Density':>12} {'Fraud Rate':>12}")
        print("-"*100)
        for _, row in df.iterrows():
            print(f"{row['Dataset']:<15} {row['Homophily']:>10.3f} {row['Feature Pred (RF)']:>10.3f} "
                  f"{row['Graph Utility']:>12.4f} {row['Density']:>12.2e} {row['Fraud Rate']:>12.2%}")
        
        # Print interpretation
        print("\n" + "="*120)
        print("INTERPRETATION")
        print("="*120)
        
        print("""
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              WHAT THESE METRICS MEAN                                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                      │
│  HOMOPHILY (0-1):                                                                                   │
│    - High (>0.7): Neighbors have same label → GNNs should help                                      │
│    - Medium (0.5-0.7): Mixed → Graph may or may not help                                            │
│    - Low (<0.5): Neighbors have different labels → GNNs may hurt (heterophily)                      │
│                                                                                                      │
│  ADJUSTED HOMOPHILY (-1 to 1):                                                                       │
│    - Positive: Graph structure is informative beyond random                                         │
│    - Zero: No structural signal                                                                     │
│    - Negative: Anti-homophily (neighbors tend to have opposite labels)                              │
│                                                                                                      │
│  FRAUD HOMOPHILY (0-1):                                                                              │
│    - High: Fraud nodes connect to other fraud nodes → Easy to detect via graph                     │
│    - Low: Fraud nodes camouflaged among legitimate nodes → Hard to detect                          │
│                                                                                                      │
│  FEATURE PREDICTIVENESS (0.5-1):                                                                     │
│    - High (>0.8): Features alone predict well → XGBoost should excel                               │
│    - Medium (0.6-0.8): Features moderately predictive                                               │
│    - Low (<0.6): Features not very informative → Need graph or other signals                       │
│                                                                                                      │
│  GRAPH UTILITY (-0.1 to 0.1+):                                                                       │
│    - Positive: Graph features add AUC beyond features → GE-XGB should help                         │
│    - Zero: Graph doesn't help                                                                       │
│    - Negative: Graph features hurt (noise or heterophily)                                           │
│                                                                                                      │
│  DENSITY:                                                                                            │
│    - Very sparse (<1e-4): Few edges → Limited graph signal                                         │
│    - Moderate (1e-4 to 1e-2): Reasonable connectivity                                               │
│    - Dense (>1e-2): Many edges → Rich graph signal                                                  │
│                                                                                                      │
│  LP ACCURACY (0.5-1):                                                                                │
│    - High (>0.7): Simple label propagation works → Graph very informative                          │
│    - Low (~0.5): LP no better than random → Features more important                                │
│                                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
        """)
        
        # Method recommendations
        print("\n" + "="*120)
        print("METHOD RECOMMENDATIONS BY DATASET")
        print("="*120)
        
        for _, row in df.iterrows():
            dataset = row['Dataset']
            homophily = row['Homophily']
            feat_pred = row['Feature Pred (RF)']
            graph_util = row['Graph Utility']
            
            print(f"\n{dataset}:")
            print(f"  Homophily: {homophily:.3f}, Feature Pred: {feat_pred:.3f}, Graph Utility: {graph_util:.4f}")
            
            # Recommendation logic
            if feat_pred > 0.85:
                print("  → HIGH feature predictiveness: XGBoost should dominate")
            elif feat_pred < 0.65:
                print("  → LOW feature predictiveness: Need graph or other signals")
            
            if homophily > 0.7:
                print("  → HIGH homophily: GNNs should help significantly")
            elif homophily < 0.5:
                print("  → LOW homophily: GNNs may hurt (heterophily problem)")
            
            if graph_util > 0.02:
                print("  → POSITIVE graph utility: GE-XGB should outperform XGBoost")
            elif graph_util < -0.01:
                print("  → NEGATIVE graph utility: Graph features may hurt")
            
            # Final recommendation
            if feat_pred > 0.8 and homophily < 0.6:
                print("  ★ RECOMMENDATION: XGBoost (features strong, graph weak)")
            elif homophily > 0.7 and graph_util > 0.02:
                print("  ★ RECOMMENDATION: GE-XGB or GNN (graph very informative)")
            elif feat_pred > 0.7 and graph_util > 0:
                print("  ★ RECOMMENDATION: GE-XGB (combines both signals)")
            else:
                print("  ★ RECOMMENDATION: Try both XGBoost and GE-XGB")
        
        # Save results
        df.to_csv('dataset_characteristics.csv', index=False)
        print(f"\n\nResults saved to: dataset_characteristics.csv")


if __name__ == '__main__':
    main()
