"""
TH-GFD: Temporal-Heterogeneous Graph Neural Network for Fraud Detection
Main model implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, Linear
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional, Tuple
import math


class TemporalDecay(nn.Module):
    """
    Learnable temporal decay function: φ(Δt) = exp(-λ * Δt)
    Different decay rates for different relation types.
    """
    
    def __init__(self, num_relations: int, init_decay: float = 0.1):
        super().__init__()
        # Learnable decay rates (one per relation type)
        self.decay_rates = nn.Parameter(torch.ones(num_relations) * init_decay)
        
    def forward(self, delta_t: torch.Tensor, relation_idx: int = 0) -> torch.Tensor:
        """
        Compute temporal decay weights.
        
        Args:
            delta_t: Time differences (batch_size,)
            relation_idx: Index of relation type
            
        Returns:
            Decay weights in [0, 1]
        """
        lambda_r = F.softplus(self.decay_rates[relation_idx])  # Ensure positive
        return torch.exp(-lambda_r * delta_t)


class TemporalHeteroConv(nn.Module):
    """
    Temporal-Heterogeneous Graph Convolution Layer.
    Combines relation-specific transformations with time-decayed attention.
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 edge_types: List[Tuple[str, str, str]],
                 num_heads: int = 4,
                 dropout: float = 0.3,
                 use_temporal: bool = True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_types = edge_types
        self.num_heads = num_heads
        self.use_temporal = use_temporal
        
        # Relation-specific transformations
        self.relation_transforms = nn.ModuleDict()
        self.relation_attention = nn.ModuleDict()
        
        for edge_type in edge_types:
            edge_key = '__'.join(edge_type)
            
            # Linear transformation per relation
            self.relation_transforms[edge_key] = nn.Linear(in_channels, out_channels)
            
            # Attention mechanism per relation
            self.relation_attention[edge_key] = nn.Sequential(
                nn.Linear(out_channels * 2, num_heads),
                nn.LeakyReLU(0.2)
            )
        
        # Temporal decay
        if use_temporal:
            self.temporal_decay = TemporalDecay(len(edge_types))
        
        # Output projection
        self.output_proj = nn.Linear(out_channels * num_heads, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_channels)
        
    def forward(self, 
                x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple, torch.Tensor],
                edge_time_dict: Optional[Dict[Tuple, torch.Tensor]] = None,
                current_time: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with temporal-heterogeneous message passing.
        """
        out_dict = {node_type: [] for node_type in x_dict.keys()}
        
        for rel_idx, edge_type in enumerate(self.edge_types):
            if edge_type not in edge_index_dict:
                continue
                
            src_type, rel_type, dst_type = edge_type
            edge_key = '__'.join(edge_type)
            
            edge_index = edge_index_dict[edge_type]
            src_nodes, dst_nodes = edge_index[0], edge_index[1]
            
            # Get node features
            x_src = x_dict.get(src_type)
            x_dst = x_dict.get(dst_type)
            
            if x_src is None or x_dst is None:
                continue
            
            # Relation-specific transformation
            h_src = self.relation_transforms[edge_key](x_src)
            h_dst = self.relation_transforms[edge_key](x_dst)
            
            # Compute attention scores
            h_src_edge = h_src[src_nodes]
            h_dst_edge = h_dst[dst_nodes]
            
            attn_input = torch.cat([h_src_edge, h_dst_edge], dim=-1)
            attn_scores = self.relation_attention[edge_key](attn_input)  # (E, num_heads)
            
            # Apply temporal decay if available
            if self.use_temporal and edge_time_dict and edge_type in edge_time_dict:
                edge_times = edge_time_dict[edge_type]
                if current_time is not None:
                    delta_t = current_time - edge_times
                else:
                    delta_t = edge_times.max() - edge_times
                
                temporal_weights = self.temporal_decay(delta_t, rel_idx)
                attn_scores = attn_scores * temporal_weights.unsqueeze(-1)
            
            # Softmax over source nodes for each destination
            attn_weights = self._sparse_softmax(attn_scores, dst_nodes, x_dst.size(0))
            
            # Aggregate messages
            messages = h_src_edge.unsqueeze(-1) * attn_weights.unsqueeze(1)  # (E, out_dim, heads)
            messages = messages.view(-1, self.out_channels * self.num_heads)
            
            # Scatter add to destination nodes
            aggregated = torch.zeros(x_dst.size(0), self.out_channels * self.num_heads,
                                    device=x_dst.device)
            aggregated.scatter_add_(0, dst_nodes.unsqueeze(-1).expand_as(messages), messages)
            
            out_dict[dst_type].append(aggregated)
        
        # Combine messages from all relation types
        result = {}
        for node_type, messages in out_dict.items():
            if messages:
                combined = torch.stack(messages, dim=0).mean(dim=0)
                combined = self.output_proj(combined)
                combined = self.dropout(combined)
                combined = self.layer_norm(combined + x_dict[node_type])  # Residual
                result[node_type] = F.relu(combined)
            else:
                result[node_type] = x_dict[node_type]
        
        return result
    
    def _sparse_softmax(self, scores: torch.Tensor, 
                        indices: torch.Tensor, 
                        num_nodes: int) -> torch.Tensor:
        """Compute softmax over sparse indices"""
        scores_max = torch.zeros(num_nodes, scores.size(-1), device=scores.device)
        scores_max.scatter_reduce_(0, indices.unsqueeze(-1).expand_as(scores), 
                                   scores, reduce='amax', include_self=False)
        scores = scores - scores_max[indices]
        scores_exp = torch.exp(scores)
        
        scores_sum = torch.zeros(num_nodes, scores.size(-1), device=scores.device)
        scores_sum.scatter_add_(0, indices.unsqueeze(-1).expand_as(scores_exp), scores_exp)
        
        return scores_exp / (scores_sum[indices] + 1e-8)


class THGFD(nn.Module):
    """
    TH-GFD: Temporal-Heterogeneous Graph Neural Network for Fraud Detection
    
    Architecture:
        1. Entity embeddings for virtual nodes
        2. Feature projection for transaction nodes
        3. Temporal-heterogeneous message passing layers
        4. Classification and revenue prediction heads
    """
    
    def __init__(self,
                 num_features: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.3,
                 entity_vocab_sizes: Dict[str, int] = None,
                 edge_types: List[Tuple[str, str, str]] = None,
                 use_temporal: bool = True,
                 use_revenue_head: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_temporal = use_temporal
        self.use_revenue_head = use_revenue_head
        self.edge_types = edge_types or []
        
        # Transaction feature projection
        self.tx_encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Entity embeddings
        self.entity_embeddings = nn.ModuleDict()
        if entity_vocab_sizes:
            for entity_type, vocab_size in entity_vocab_sizes.items():
                self.entity_embeddings[entity_type] = nn.Embedding(vocab_size, hidden_dim)
        
        # Temporal-heterogeneous convolution layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(
                TemporalHeteroConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    edge_types=edge_types,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_temporal=use_temporal
                )
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Revenue prediction head (optional)
        if use_revenue_head:
            self.revenue_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
    
    def encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Encode all nodes into hidden representations"""
        x_dict = {}
        
        # Encode transaction nodes
        if 'transaction' in data.node_types:
            x_dict['transaction'] = self.tx_encoder(data['transaction'].x)
        
        # Encode entity nodes
        for entity_type in self.entity_embeddings.keys():
            if entity_type in data.node_types:
                # Use embedding lookup (indices from 0 to vocab_size-1)
                indices = torch.arange(data[entity_type].num_nodes, device=data['transaction'].x.device)
                x_dict[entity_type] = self.entity_embeddings[entity_type](indices)
        
        return x_dict
    
    def forward(self, data: HeteroData) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Returns:
            fraud_logits: Fraud probability logits for transactions
            revenue_pred: Predicted revenue (if use_revenue_head=True)
        """
        # Encode nodes
        x_dict = self.encode(data)
        
        # Get edge indices and times
        edge_index_dict = {et: data[et].edge_index for et in self.edge_types if et in data.edge_types}
        edge_time_dict = {}
        for et in self.edge_types:
            if et in data.edge_types and hasattr(data[et], 'edge_time'):
                edge_time_dict[et] = data[et].edge_time
        
        # Current time (max time in data)
        current_time = None
        if hasattr(data['transaction'], 'time'):
            current_time = data['transaction'].time.max().item()
        
        # Message passing
        for conv in self.conv_layers:
            x_dict = conv(x_dict, edge_index_dict, edge_time_dict, current_time)
        
        # Get transaction embeddings
        tx_embeddings = x_dict['transaction']
        
        # Classification
        fraud_logits = self.classifier(tx_embeddings).squeeze(-1)
        
        # Revenue prediction
        revenue_pred = None
        if self.use_revenue_head:
            revenue_pred = self.revenue_head(tx_embeddings).squeeze(-1)
        
        return fraud_logits, revenue_pred
    
    def get_embeddings(self, data: HeteroData) -> torch.Tensor:
        """Get transaction embeddings after message passing"""
        x_dict = self.encode(data)
        
        edge_index_dict = {et: data[et].edge_index for et in self.edge_types if et in data.edge_types}
        edge_time_dict = {}
        for et in self.edge_types:
            if et in data.edge_types and hasattr(data[et], 'edge_time'):
                edge_time_dict[et] = data[et].edge_time
        
        current_time = None
        if hasattr(data['transaction'], 'time'):
            current_time = data['transaction'].time.max().item()
        
        for conv in self.conv_layers:
            x_dict = conv(x_dict, edge_index_dict, edge_time_dict, current_time)
        
        return x_dict['transaction']


class SimplifiedTHGFD(nn.Module):
    """
    Simplified TH-GFD using PyG's built-in HeteroConv.
    Easier to train and debug.
    """
    
    def __init__(self,
                 num_features: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 edge_types: List[Tuple[str, str, str]] = None):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        # HeteroConv layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                conv_dict[edge_type] = SAGEConv(hidden_dim, hidden_dim)
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Revenue head
        self.revenue_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x_dict, edge_index_dict):
        # Project transaction features
        if 'transaction' in x_dict:
            x_dict = dict(x_dict)  # Make mutable copy
            x_dict['transaction'] = self.input_proj(x_dict['transaction'])
        
        # Initialize entity node features if needed
        for node_type in x_dict.keys():
            if node_type != 'transaction' and x_dict[node_type].size(-1) != self.hidden_dim:
                x_dict[node_type] = torch.zeros(
                    x_dict[node_type].size(0), self.hidden_dim,
                    device=x_dict['transaction'].device
                )
        
        # Message passing
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.relu(self.dropout(v)) for k, v in x_dict.items()}
        
        # Get transaction embeddings
        tx_emb = x_dict['transaction']
        
        # Predictions
        fraud_logits = self.classifier(tx_emb).squeeze(-1)
        revenue_pred = self.revenue_head(tx_emb).squeeze(-1)
        
        return fraud_logits, revenue_pred


def create_model(data: HeteroData, config) -> nn.Module:
    """Factory function to create TH-GFD model from data and config"""
    
    # Get feature dimension
    num_features = data['transaction'].x.size(-1)
    
    # Get entity vocab sizes
    entity_vocab_sizes = {}
    for node_type in data.node_types:
        if node_type != 'transaction':
            entity_vocab_sizes[node_type] = data[node_type].num_nodes
    
    # Get edge types
    edge_types = list(data.edge_types)
    
    model = THGFD(
        num_features=num_features,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        entity_vocab_sizes=entity_vocab_sizes,
        edge_types=edge_types,
        use_temporal=config.model.use_temporal,
        use_revenue_head=True
    )
    
    return model
