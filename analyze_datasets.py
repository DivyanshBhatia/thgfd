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
        for p in ['./data/ieee-cis', './ieee-cis-fraud-detection', '/kaggle/input/ieee-cis-fraud-detection']:
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


def load_customs(customs_path=None):
    """
    Load Customs fraud dataset.
    
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
    if customs_path is None:
        for p in ['./data/customs/customs.csv', './customs.csv', '../customs.csv', 
                  './data/customs.csv', '../data/customs.csv']:
            if os.path.exists(p):
                customs_path = p
                break
    
    if customs_path is None or not os.path.exists(customs_path):
        print(f"    Customs file not found. Searched paths:")
        print(f"      ./data/customs/customs.csv")
        print(f"      ./customs.csv")
        print(f"    Please specify path with --customs_path")
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
