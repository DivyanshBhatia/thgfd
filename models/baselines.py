"""
Baseline Models for Comparison
Includes: XGBoost, MLP, GCN, GraphSAGE, GAT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data
import numpy as np
from typing import Optional, Tuple
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


# =============================================================================
# Tree-based Baselines
# =============================================================================

class XGBoostBaseline:
    """XGBoost baseline for tabular fraud detection"""
    
    def __init__(self, **kwargs):
        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                scale_pos_weight=kwargs.get('scale_pos_weight', 10),  # Handle imbalance
                use_label_encoder=False,
                eval_metric='auc',
                random_state=42
            )
        except ImportError:
            print("XGBoost not installed. Run: pip install xgboost")
            self.model = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if self.model is None:
            return
        
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
    
    def predict_proba(self, X):
        if self.model is None:
            return np.zeros((len(X), 2))
        return self.model.predict_proba(X)
    
    def predict(self, X):
        if self.model is None:
            return np.zeros(len(X))
        return self.model.predict(X)


class LightGBMBaseline:
    """LightGBM baseline"""
    
    def __init__(self, **kwargs):
        try:
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                class_weight='balanced',
                random_state=42,
                verbose=-1
            )
        except ImportError:
            print("LightGBM not installed. Run: pip install lightgbm")
            self.model = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if self.model is None:
            return
        self.model.fit(X_train, y_train)
    
    def predict_proba(self, X):
        if self.model is None:
            return np.zeros((len(X), 2))
        return self.model.predict_proba(X)


# =============================================================================
# MLP Baseline
# =============================================================================

class MLPBaseline(nn.Module):
    """Simple MLP baseline"""
    
    def __init__(self, 
                 num_features: int,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        
        layers = []
        in_dim = num_features
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


# =============================================================================
# GNN Baselines
# =============================================================================

class GCNBaseline(nn.Module):
    """Graph Convolutional Network baseline"""
    
    def __init__(self,
                 num_features: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return self.classifier(x).squeeze(-1)


class GraphSAGEBaseline(nn.Module):
    """GraphSAGE baseline"""
    
    def __init__(self,
                 num_features: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(num_features, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return self.classifier(x).squeeze(-1)


class GATBaseline(nn.Module):
    """Graph Attention Network baseline"""
    
    def __init__(self,
                 num_features: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(num_features, hidden_dim // num_heads, heads=num_heads))
        
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads))
        
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return self.classifier(x).squeeze(-1)


# =============================================================================
# GraphFC-like Baseline (Semi-supervised with self-training)
# =============================================================================

class GraphFCBaseline(nn.Module):
    """
    Simplified GraphFC baseline.
    Uses GraphSAGE with self-supervised pretraining.
    """
    
    def __init__(self,
                 num_features: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        
        # Encoder
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(num_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout
    
    def encode(self, x, edge_index):
        """Get node embeddings"""
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def forward(self, x, edge_index):
        h = self.encode(x, edge_index)
        return self.classifier(h).squeeze(-1)
    
    def contrastive_loss(self, x, edge_index, temperature=0.5):
        """
        Self-supervised contrastive loss.
        Positive pairs: connected nodes
        Negative pairs: random nodes
        """
        h = self.encode(x, edge_index)
        z = self.projector(h)
        z = F.normalize(z, dim=-1)
        
        # Positive pairs from edges
        src, dst = edge_index
        pos_sim = (z[src] * z[dst]).sum(dim=-1) / temperature
        
        # Negative pairs (random sampling)
        num_neg = min(1000, len(src))
        neg_idx = torch.randint(0, z.size(0), (num_neg,), device=z.device)
        neg_sim = torch.mm(z[src[:num_neg]], z[neg_idx].t()) / temperature
        
        # InfoNCE loss
        pos_exp = torch.exp(pos_sim[:num_neg])
        neg_exp = torch.exp(neg_sim).sum(dim=-1)
        loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8)).mean()
        
        return loss


# =============================================================================
# Utility functions for training baselines
# =============================================================================

def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train and evaluate XGBoost baseline"""
    model = XGBoostBaseline()
    model.fit(X_train, y_train, X_val, y_val)
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metrics
    metrics = {
        'auc': roc_auc_score(y_test, y_pred_proba),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }
    
    return model, metrics


def train_gnn_baseline(model, data, train_mask, val_mask, test_mask,
                       epochs=200, lr=0.01, weight_decay=1e-4, device='cpu'):
    """Train GNN baseline model"""
    model = model.to(device)
    data = data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Handle class imbalance
    pos_weight = (train_mask.sum() - data.y[train_mask].sum()) / (data.y[train_mask].sum() + 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_val_auc = 0
    best_model_state = None
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask].float())
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_proba = torch.sigmoid(out[val_mask]).cpu().numpy()
            val_labels = data.y[val_mask].cpu().numpy()
            
            if len(np.unique(val_labels)) > 1:
                val_auc = roc_auc_score(val_labels, val_proba)
            else:
                val_auc = 0.5
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Val AUC={val_auc:.4f}")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Test evaluation
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        test_proba = torch.sigmoid(out[test_mask]).cpu().numpy()
        test_labels = data.y[test_mask].cpu().numpy()
        test_pred = (test_proba > 0.5).astype(int)
        
        metrics = {
            'auc': roc_auc_score(test_labels, test_proba),
            'f1': f1_score(test_labels, test_pred),
            'precision': precision_score(test_labels, test_pred, zero_division=0),
            'recall': recall_score(test_labels, test_pred)
        }
    
    return model, metrics
