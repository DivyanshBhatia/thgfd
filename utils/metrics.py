"""
Evaluation metrics for fraud detection
Following DATE and GraphFC papers
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, precision_recall_curve, confusion_matrix
)
from typing import Dict, Optional


def compute_precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: float) -> float:
    """
    Compute Precision@k%
    
    Args:
        y_true: True labels (0 or 1)
        y_scores: Predicted probabilities
        k: Percentage to consider (e.g., 1 for top 1%, 5 for top 5%)
    
    Returns:
        Precision among top k% predictions
    """
    n = len(y_true)
    n_select = max(int(n * k / 100), 1)
    
    # Get indices of top k% scores
    top_k_idx = np.argsort(y_scores)[-n_select:]
    
    # Precision is the fraction of true positives among selected
    return y_true[top_k_idx].mean()


def compute_recall_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: float) -> float:
    """
    Compute Recall@k%
    
    Args:
        y_true: True labels (0 or 1)
        y_scores: Predicted probabilities
        k: Percentage to consider
    
    Returns:
        Fraction of all positives found in top k%
    """
    n = len(y_true)
    n_select = max(int(n * k / 100), 1)
    
    # Get indices of top k% scores
    top_k_idx = np.argsort(y_scores)[-n_select:]
    
    # Recall is the fraction of all positives that are in top k%
    total_positives = y_true.sum()
    if total_positives == 0:
        return 0.0
    
    return y_true[top_k_idx].sum() / total_positives


def compute_revenue_at_k(y_true: np.ndarray, y_scores: np.ndarray, 
                         revenue: np.ndarray, k: float) -> float:
    """
    Compute Revenue@k% (fraction of total revenue captured in top k%)
    
    Args:
        y_true: True labels (0 or 1) 
        y_scores: Predicted probabilities
        revenue: Revenue values for each transaction
        k: Percentage to consider
    
    Returns:
        Fraction of total revenue captured in top k%
    """
    n = len(y_true)
    n_select = max(int(n * k / 100), 1)
    
    # Get indices of top k% scores
    top_k_idx = np.argsort(y_scores)[-n_select:]
    
    # Revenue captured
    total_revenue = revenue.sum()
    if total_revenue == 0:
        return 0.0
    
    return revenue[top_k_idx].sum() / total_revenue


def compute_all_metrics(y_true: np.ndarray, 
                        y_scores: np.ndarray,
                        revenue: Optional[np.ndarray] = None,
                        threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute all fraud detection metrics.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
        revenue: Revenue values (optional)
        threshold: Classification threshold
    
    Returns:
        Dictionary of metrics
    """
    y_pred = (y_scores > threshold).astype(int)
    
    metrics = {}
    
    # Basic metrics
    if len(np.unique(y_true)) > 1:
        metrics['AUC'] = roc_auc_score(y_true, y_scores)
        metrics['AP'] = average_precision_score(y_true, y_scores)
    else:
        metrics['AUC'] = 0.5
        metrics['AP'] = y_true.mean()
    
    metrics['F1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
    
    # Top-k metrics (customs-specific)
    for k in [1, 5, 10]:
        metrics[f'Precision@{k}%'] = compute_precision_at_k(y_true, y_scores, k)
        metrics[f'Recall@{k}%'] = compute_recall_at_k(y_true, y_scores, k)
        
        if revenue is not None:
            metrics[f'Revenue@{k}%'] = compute_revenue_at_k(y_true, y_scores, revenue, k)
    
    # Confusion matrix derived metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['TP'] = tp
    metrics['FP'] = fp
    metrics['TN'] = tn
    metrics['FN'] = fn
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """Pretty print metrics"""
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print('='*50)
    
    # Group metrics
    basic = ['AUC', 'AP', 'F1', 'Precision', 'Recall', 'Specificity']
    topk = [k for k in metrics.keys() if '@' in k]
    
    print("\nBasic Metrics:")
    for m in basic:
        if m in metrics:
            print(f"  {m:<15}: {metrics[m]:.4f}")
    
    print("\nTop-k Metrics:")
    for m in sorted(topk):
        print(f"  {m:<15}: {metrics[m]:.4f}")
    
    print('='*50)


def compare_methods(results: Dict[str, Dict[str, float]], 
                    metrics_to_show: list = None) -> str:
    """
    Create comparison table across methods.
    
    Args:
        results: {method_name: {metric: value}}
        metrics_to_show: List of metric names to include
    
    Returns:
        Formatted table string
    """
    if metrics_to_show is None:
        metrics_to_show = ['AUC', 'F1', 'Precision@1%', 'Recall@5%', 'Revenue@5%']
    
    # Filter to available metrics
    available_metrics = set()
    for method_results in results.values():
        available_metrics.update(method_results.keys())
    metrics_to_show = [m for m in metrics_to_show if m in available_metrics]
    
    # Build table
    header = f"{'Method':<15}" + ''.join(f"{m:>12}" for m in metrics_to_show)
    separator = '-' * len(header)
    
    rows = [header, separator]
    
    for method, method_results in results.items():
        row = f"{method:<15}"
        for metric in metrics_to_show:
            val = method_results.get(metric, 0)
            row += f"{val:>12.4f}"
        rows.append(row)
    
    return '\n'.join(rows)


class MetricsTracker:
    """Track metrics during training"""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.val_f1s = []
        self.best_val_auc = 0
        self.best_epoch = 0
        
    def update(self, epoch: int, train_loss: float, 
               val_loss: float = None, val_metrics: dict = None):
        """Update tracking with new epoch results"""
        self.train_losses.append(train_loss)
        
        if val_loss is not None:
            self.val_losses.append(val_loss)
        
        if val_metrics:
            self.val_aucs.append(val_metrics.get('AUC', 0))
            self.val_f1s.append(val_metrics.get('F1', 0))
            
            if val_metrics.get('AUC', 0) > self.best_val_auc:
                self.best_val_auc = val_metrics['AUC']
                self.best_epoch = epoch
                return True  # New best
        
        return False
    
    def get_summary(self) -> dict:
        """Get training summary"""
        return {
            'best_val_auc': self.best_val_auc,
            'best_epoch': self.best_epoch,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_auc': self.val_aucs[-1] if self.val_aucs else None,
            'n_epochs': len(self.train_losses)
        }
