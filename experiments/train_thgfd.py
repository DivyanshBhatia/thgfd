"""
TH-GFD Training Pipeline
Complete training with:
1. Self-supervised pre-training
2. Semi-supervised fine-tuning
3. Pseudo-labeling
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from typing import Dict, Optional, Tuple
from tqdm import tqdm


class ContrastiveLoss(nn.Module):
    """Multi-view contrastive loss for pre-training"""
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent contrastive loss between two views.
        
        Args:
            z1, z2: Node embeddings from two views (N, D)
        """
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        N = z1.size(0)
        
        # Similarity matrix
        sim = torch.mm(z1, z2.t()) / self.temperature  # (N, N)
        
        # Positive pairs on diagonal
        pos = torch.diag(sim)
        
        # InfoNCE loss
        loss = -pos + torch.logsumexp(sim, dim=1)
        
        return loss.mean()


class THGFDTrainer:
    """
    Complete training pipeline for TH-GFD.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 hidden_dim: int = 64,
                 temperature: float = 0.5,
                 lr: float = 0.005,
                 weight_decay: float = 1e-4):
        
        self.model = model.to(device)
        self.device = device
        self.hidden_dim = hidden_dim
        
        # Losses
        self.contrastive_loss = ContrastiveLoss(temperature)
        
        # Optimizer
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Tracking
        self.pretrain_losses = []
        self.train_losses = []
        self.val_aucs = []
    
    def temporal_augmentation(self, data: HeteroData) -> HeteroData:
        """
        Create temporal augmentation by sampling edges from different time windows.
        """
        data_aug = data.clone()
        
        # For each edge type with temporal info
        for edge_type in data.edge_types:
            if hasattr(data[edge_type], 'edge_time'):
                edge_time = data[edge_type].edge_time
                edge_index = data[edge_type].edge_index
                
                # Sample edges from recent time window
                max_time = edge_time.max()
                time_window = (max_time - edge_time.min()) * 0.5  # 50% time window
                
                recent_mask = edge_time > (max_time - time_window)
                
                if recent_mask.sum() > 10:
                    # Randomly drop some older edges
                    keep_mask = recent_mask | (torch.rand_like(edge_time) > 0.3)
                    data_aug[edge_type].edge_index = edge_index[:, keep_mask]
                    data_aug[edge_type].edge_time = edge_time[keep_mask]
        
        return data_aug
    
    def relation_augmentation(self, data: HeteroData) -> HeteroData:
        """
        Create relation augmentation by masking one relation type.
        """
        data_aug = data.clone()
        
        # Randomly select a relation type to mask
        edge_types = list(data.edge_types)
        if len(edge_types) > 1:
            mask_type = edge_types[np.random.randint(len(edge_types))]
            
            # Keep only 20% of edges for this type
            edge_index = data[mask_type].edge_index
            n_edges = edge_index.size(1)
            keep_idx = torch.randperm(n_edges)[:int(n_edges * 0.2)]
            
            data_aug[mask_type].edge_index = edge_index[:, keep_idx]
            if hasattr(data[mask_type], 'edge_time'):
                data_aug[mask_type].edge_time = data[mask_type].edge_time[keep_idx]
        
        return data_aug
    
    def pretrain_step(self, data: HeteroData) -> float:
        """Single pre-training step with contrastive learning"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Get embeddings from original view
        z1 = self.model.get_embeddings(data)
        
        # Create augmented view (temporal)
        data_aug1 = self.temporal_augmentation(data)
        z2 = self.model.get_embeddings(data_aug1)
        
        # Temporal consistency loss
        loss_temp = self.contrastive_loss(z1, z2)
        
        # Create another augmented view (relation)
        data_aug2 = self.relation_augmentation(data)
        z3 = self.model.get_embeddings(data_aug2)
        
        # Cross-relation consistency loss
        loss_rel = self.contrastive_loss(z1, z3)
        
        # Neighbor preservation loss (simplified)
        loss_neigh = self._neighbor_loss(z1, data)
        
        # Combined loss
        loss = loss_temp + 0.5 * loss_rel + 0.5 * loss_neigh
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _neighbor_loss(self, z: torch.Tensor, data: HeteroData) -> torch.Tensor:
        """Neighbor preservation loss"""
        loss = 0.0
        n_edges = 0
        
        for edge_type in data.edge_types:
            if edge_type[0] == 'transaction' or edge_type[2] == 'transaction':
                edge_index = data[edge_type].edge_index
                
                if edge_type[0] == 'transaction':
                    src_emb = z[edge_index[0]]
                    # Need to handle entity embeddings
                    continue
                
                n_edges += edge_index.size(1)
        
        if n_edges == 0:
            return torch.tensor(0.0, device=z.device)
        
        # Simplified: use edge homophily within transaction graph
        return torch.tensor(0.0, device=z.device)
    
    def pretrain(self, data: HeteroData, epochs: int = 100, 
                 verbose: bool = True) -> list:
        """
        Self-supervised pre-training.
        
        Args:
            data: Heterogeneous graph data
            epochs: Number of pre-training epochs
        """
        data = data.to(self.device)
        self.pretrain_losses = []
        
        pbar = tqdm(range(epochs), desc="Pre-training") if verbose else range(epochs)
        
        for epoch in pbar:
            loss = self.pretrain_step(data)
            self.pretrain_losses.append(loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        return self.pretrain_losses
    
    def finetune_step(self, data: HeteroData, 
                      train_mask: torch.Tensor,
                      label_mask: torch.Tensor,
                      pseudo_threshold: float = 0.9,
                      alpha: float = 10.0,
                      mu: float = 0.1) -> Tuple[float, float]:
        """
        Single fine-tuning step with pseudo-labeling.
        
        Args:
            data: Graph data
            train_mask: Training set mask
            label_mask: Labeled samples mask
            pseudo_threshold: Confidence threshold for pseudo-labels
            alpha: Weight for revenue loss
            mu: Weight for pseudo-label loss
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        fraud_logits, revenue_pred = self.model(data)
        
        # Get labels
        y = data['transaction'].y.float()
        
        # Classification loss on labeled data
        labeled_mask = train_mask & label_mask
        if labeled_mask.sum() > 0:
            pos_weight = (labeled_mask.sum() - y[labeled_mask].sum()) / (y[labeled_mask].sum() + 1)
            loss_cls = F.binary_cross_entropy_with_logits(
                fraud_logits[labeled_mask], 
                y[labeled_mask],
                pos_weight=pos_weight.clamp(max=20)
            )
        else:
            loss_cls = torch.tensor(0.0, device=self.device)
        
        # Revenue loss (if available)
        loss_rev = torch.tensor(0.0, device=self.device)
        if hasattr(data['transaction'], 'revenue') and revenue_pred is not None:
            revenue = data['transaction'].revenue
            # Only compute on fraudulent labeled samples
            fraud_mask = labeled_mask & (y == 1)
            if fraud_mask.sum() > 0:
                loss_rev = F.mse_loss(revenue_pred[fraud_mask], revenue[fraud_mask])
        
        # Pseudo-labeling loss
        loss_pseudo = torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            probs = torch.sigmoid(fraud_logits)
            # High confidence predictions on unlabeled training data
            unlabeled_train = train_mask & ~label_mask
            confident_pos = unlabeled_train & (probs > pseudo_threshold)
            confident_neg = unlabeled_train & (probs < (1 - pseudo_threshold))
        
        if confident_pos.sum() > 0 or confident_neg.sum() > 0:
            pseudo_mask = confident_pos | confident_neg
            pseudo_labels = (probs[pseudo_mask] > 0.5).float()
            loss_pseudo = F.binary_cross_entropy_with_logits(
                fraud_logits[pseudo_mask],
                pseudo_labels
            )
        
        # Combined loss
        loss = loss_cls + alpha * loss_rev + mu * loss_pseudo
        
        loss.backward()
        self.optimizer.step()
        
        return loss_cls.item(), loss_rev.item()
    
    def finetune(self, data: HeteroData,
                 train_mask: torch.Tensor,
                 val_mask: torch.Tensor,
                 label_mask: torch.Tensor,
                 epochs: int = 200,
                 patience: int = 20,
                 pseudo_threshold_start: float = 0.95,
                 pseudo_threshold_end: float = 0.7,
                 verbose: bool = True) -> dict:
        """
        Semi-supervised fine-tuning with curriculum pseudo-labeling.
        """
        data = data.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        label_mask = label_mask.to(self.device)
        
        best_val_auc = 0
        best_model_state = None
        patience_counter = 0
        
        self.train_losses = []
        self.val_aucs = []
        
        pbar = tqdm(range(epochs), desc="Fine-tuning") if verbose else range(epochs)
        
        for epoch in pbar:
            # Curriculum: gradually decrease pseudo-label threshold
            progress = epoch / epochs
            pseudo_threshold = pseudo_threshold_start - progress * (pseudo_threshold_start - pseudo_threshold_end)
            
            # Training step
            loss_cls, loss_rev = self.finetune_step(
                data, train_mask, label_mask, pseudo_threshold
            )
            self.train_losses.append(loss_cls)
            
            # Validation
            val_auc = self.evaluate(data, val_mask)
            self.val_aucs.append(val_auc)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch}")
                break
            
            if verbose and (epoch + 1) % 20 == 0:
                pbar.set_postfix({
                    'loss': f'{loss_cls:.4f}',
                    'val_auc': f'{val_auc:.4f}',
                    'best': f'{best_val_auc:.4f}'
                })
        
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            self.model.to(self.device)
        
        return {
            'best_val_auc': best_val_auc,
            'train_losses': self.train_losses,
            'val_aucs': self.val_aucs
        }
    
    @torch.no_grad()
    def evaluate(self, data: HeteroData, mask: torch.Tensor) -> float:
        """Evaluate model on masked subset"""
        self.model.eval()
        
        fraud_logits, _ = self.model(data)
        probs = torch.sigmoid(fraud_logits[mask]).cpu().numpy()
        labels = data['transaction'].y[mask].cpu().numpy()
        
        if len(np.unique(labels)) > 1:
            auc = roc_auc_score(labels, probs)
        else:
            auc = 0.5
        
        return auc
    
    @torch.no_grad()
    def predict(self, data: HeteroData) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions for all nodes"""
        self.model.eval()
        data = data.to(self.device)
        
        fraud_logits, revenue_pred = self.model(data)
        
        fraud_probs = torch.sigmoid(fraud_logits).cpu().numpy()
        revenue = revenue_pred.cpu().numpy() if revenue_pred is not None else None
        
        return fraud_probs, revenue


def run_thgfd_experiment(data: HeteroData,
                         train_idx: np.ndarray,
                         val_idx: np.ndarray,
                         test_idx: np.ndarray,
                         label_mask: np.ndarray,
                         config: dict,
                         device: torch.device) -> dict:
    """
    Run complete TH-GFD experiment.
    
    Args:
        data: HeteroData graph
        train_idx, val_idx, test_idx: Split indices
        label_mask: Boolean mask for labeled samples
        config: Configuration dictionary
        device: Torch device
    
    Returns:
        Dictionary with results and predictions
    """
    from models.thgfd import THGFD
    
    # Create masks
    n = data['transaction'].num_nodes
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    label_mask_tensor = torch.tensor(label_mask, dtype=torch.bool)
    
    # Get model parameters from data
    num_features = data['transaction'].x.size(-1)
    entity_vocab_sizes = {}
    for node_type in data.node_types:
        if node_type != 'transaction':
            entity_vocab_sizes[node_type] = data[node_type].num_nodes
    
    edge_types = list(data.edge_types)
    
    # Create model
    model = THGFD(
        num_features=num_features,
        hidden_dim=config.get('hidden_dim', 64),
        num_layers=config.get('num_layers', 2),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('dropout', 0.3),
        entity_vocab_sizes=entity_vocab_sizes,
        edge_types=edge_types,
        use_temporal=config.get('use_temporal', True),
        use_revenue_head=True
    )
    
    # Create trainer
    trainer = THGFDTrainer(
        model=model,
        device=device,
        hidden_dim=config.get('hidden_dim', 64),
        lr=config.get('lr', 0.005),
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # Pre-training
    if config.get('pretrain', True):
        print("Starting pre-training...")
        pretrain_losses = trainer.pretrain(
            data, 
            epochs=config.get('pretrain_epochs', 100),
            verbose=True
        )
    
    # Fine-tuning
    print("Starting fine-tuning...")
    finetune_results = trainer.finetune(
        data,
        train_mask,
        val_mask,
        label_mask_tensor,
        epochs=config.get('finetune_epochs', 200),
        patience=config.get('patience', 20),
        verbose=True
    )
    
    # Evaluation
    print("Evaluating on test set...")
    fraud_probs, revenue_pred = trainer.predict(data)
    
    test_probs = fraud_probs[test_mask.numpy()]
    test_labels = data['transaction'].y[test_mask].numpy()
    
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    
    test_pred = (test_probs > 0.5).astype(int)
    
    results = {
        'AUC': roc_auc_score(test_labels, test_probs),
        'F1': f1_score(test_labels, test_pred),
        'Precision': precision_score(test_labels, test_pred, zero_division=0),
        'Recall': recall_score(test_labels, test_pred),
        'best_val_auc': finetune_results['best_val_auc']
    }
    
    # Precision@k and Recall@k
    for k in [1, 5]:
        n_select = int(len(test_probs) * k / 100)
        top_k_idx = np.argsort(test_probs)[-n_select:]
        
        results[f'Precision@{k}%'] = test_labels[top_k_idx].mean()
        results[f'Recall@{k}%'] = test_labels[top_k_idx].sum() / max(test_labels.sum(), 1)
    
    print(f"\nTest Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    
    return {
        'metrics': results,
        'predictions': fraud_probs,
        'revenue_predictions': revenue_pred,
        'model': model,
        'trainer': trainer
    }
