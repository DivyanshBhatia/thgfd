"""
Graph Construction for TH-GFD
Converts tabular data to temporal-heterogeneous graphs
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData, Data
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class TemporalHeteroGraphBuilder:
    """
    Build temporal heterogeneous graph from tabular transaction data.
    
    Node types:
        - transaction: Each row becomes a transaction node
        - importer, declarant, office, country: Virtual entity nodes
    
    Edge types:
        - (transaction, belongs_to, importer)
        - (transaction, filed_by, declarant)  
        - (transaction, processed_at, office)
        - (transaction, from, country)
    """
    
    def __init__(self, entity_columns: List[str], numerical_columns: List[str]):
        self.entity_columns = entity_columns
        self.numerical_columns = numerical_columns
        self.entity_mappings = {}  # Maps entity values to node indices
        
    def build(self, df: pd.DataFrame, 
              label_col: str = 'illicit',
              revenue_col: str = 'revenue',
              date_col: str = 'sgd.date') -> HeteroData:
        """
        Build HeteroData graph from DataFrame.
        
        Args:
            df: Transaction DataFrame
            label_col: Column name for fraud labels
            revenue_col: Column name for revenue (optional)
            date_col: Column name for timestamps
            
        Returns:
            HeteroData object for PyG
        """
        data = HeteroData()
        n_transactions = len(df)
        
        # 1. Create transaction node features
        feature_cols = [c for c in self.numerical_columns if c in df.columns]
        if feature_cols:
            tx_features = df[feature_cols].fillna(0).values.astype(np.float32)
        else:
            tx_features = np.zeros((n_transactions, 1), dtype=np.float32)
        
        data['transaction'].x = torch.tensor(tx_features, dtype=torch.float32)
        
        # 2. Create labels and masks
        if label_col in df.columns:
            labels = df[label_col].values.astype(np.int64)
            data['transaction'].y = torch.tensor(labels, dtype=torch.long)
        
        if revenue_col in df.columns:
            revenue = df[revenue_col].fillna(0).values.astype(np.float32)
            data['transaction'].revenue = torch.tensor(revenue, dtype=torch.float32)
        
        # 3. Create temporal information
        if date_col in df.columns:
            dates = pd.to_datetime(df[date_col], format='%y-%m-%d', errors='coerce')
            # Convert to days since start
            min_date = dates.min()
            time_steps = (dates - min_date).dt.days.fillna(0).values.astype(np.float32)
            data['transaction'].time = torch.tensor(time_steps, dtype=torch.float32)
        
        # 4. Create entity nodes and edges
        for entity_col in self.entity_columns:
            if entity_col not in df.columns:
                continue
                
            # Get unique entities
            entities = df[entity_col].astype(str).fillna('UNKNOWN')
            unique_entities = entities.unique()
            
            # Create mapping
            entity_to_idx = {e: i for i, e in enumerate(unique_entities)}
            self.entity_mappings[entity_col] = entity_to_idx
            
            # Entity node features (one-hot or learnable embeddings later)
            n_entities = len(unique_entities)
            entity_type = entity_col.replace('.', '_').replace(' ', '_')
            data[entity_type].x = torch.zeros(n_entities, 1)  # Placeholder
            data[entity_type].num_nodes = n_entities
            
            # Create edges: transaction -> entity
            tx_indices = torch.arange(n_transactions)
            entity_indices = torch.tensor([entity_to_idx[e] for e in entities])
            
            edge_type = ('transaction', f'to_{entity_type}', entity_type)
            data[edge_type].edge_index = torch.stack([tx_indices, entity_indices])
            
            # Reverse edges: entity -> transaction
            rev_edge_type = (entity_type, f'has_{entity_type}', 'transaction')
            data[rev_edge_type].edge_index = torch.stack([entity_indices, tx_indices])
            
            # Add temporal info to edges if available
            if date_col in df.columns:
                data[edge_type].edge_time = data['transaction'].time
                data[rev_edge_type].edge_time = data['transaction'].time
        
        # Store metadata
        data.num_transactions = n_transactions
        data.entity_columns = self.entity_columns
        
        return data


class HomogeneousGraphBuilder:
    """
    Build homogeneous graph (for baselines like GCN, GraphSAGE).
    Connects transactions that share entities.
    """
    
    def __init__(self, entity_columns: List[str], numerical_columns: List[str]):
        self.entity_columns = entity_columns
        self.numerical_columns = numerical_columns
        
    def build(self, df: pd.DataFrame,
              label_col: str = 'illicit',
              max_neighbors: int = 50) -> Data:
        """
        Build homogeneous graph where transactions sharing entities are connected.
        """
        n = len(df)
        
        # Features
        feature_cols = [c for c in self.numerical_columns if c in df.columns]
        if feature_cols:
            features = df[feature_cols].fillna(0).values.astype(np.float32)
        else:
            features = np.zeros((n, 1), dtype=np.float32)
        
        # Labels
        labels = df[label_col].values.astype(np.int64) if label_col in df.columns else None
        
        # Build edges based on shared entities
        edges = []
        
        for entity_col in self.entity_columns:
            if entity_col not in df.columns:
                continue
            
            # Group transactions by entity
            entity_groups = df.groupby(entity_col).indices
            
            for entity, tx_indices in entity_groups.items():
                if len(tx_indices) < 2:
                    continue
                
                # Sample if too many (to avoid memory issues)
                if len(tx_indices) > max_neighbors:
                    tx_indices = np.random.choice(tx_indices, max_neighbors, replace=False)
                
                # Connect all pairs (or use sampling for large groups)
                for i, idx1 in enumerate(tx_indices):
                    for idx2 in tx_indices[i+1:]:
                        edges.append([idx1, idx2])
                        edges.append([idx2, idx1])  # Undirected
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        data = Data(
            x=torch.tensor(features, dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor(labels, dtype=torch.long) if labels is not None else None
        )
        
        return data


def build_graph_for_dataset(df: pd.DataFrame, 
                           config,
                           graph_type: str = 'hetero') -> HeteroData:
    """
    Convenience function to build graph based on config.
    
    Args:
        df: Transaction DataFrame
        config: Configuration object
        graph_type: 'hetero' or 'homo'
    """
    if graph_type == 'hetero':
        builder = TemporalHeteroGraphBuilder(
            entity_columns=config.graph.entity_columns,
            numerical_columns=config.graph.numerical_columns
        )
    else:
        builder = HomogeneousGraphBuilder(
            entity_columns=config.graph.entity_columns,
            numerical_columns=config.graph.numerical_columns
        )
    
    return builder.build(df)


# Test function
def test_graph_construction():
    """Test graph construction with sample data"""
    # Create sample data
    np.random.seed(42)
    n = 100
    
    df = pd.DataFrame({
        'sgd.id': range(n),
        'sgd.date': pd.date_range('2013-01-01', periods=n, freq='D').strftime('%y-%m-%d'),
        'importer.id': [f'IMP{i % 10}' for i in range(n)],
        'declarant.id': [f'DEC{i % 5}' for i in range(n)],
        'office.id': [f'OFF{i % 3}' for i in range(n)],
        'country': [f'CTY{i % 8}' for i in range(n)],
        'quantity': np.random.randint(1, 100, n),
        'gross.weight': np.random.uniform(100, 1000, n),
        'fob.value': np.random.uniform(1000, 10000, n),
        'cif.value': np.random.uniform(1000, 12000, n),
        'total.taxes': np.random.uniform(100, 1000, n),
        'illicit': np.random.binomial(1, 0.1, n),
        'revenue': np.random.uniform(0, 500, n)
    })
    
    # Build heterogeneous graph
    builder = TemporalHeteroGraphBuilder(
        entity_columns=['importer.id', 'declarant.id', 'office.id', 'country'],
        numerical_columns=['quantity', 'gross.weight', 'fob.value', 'cif.value', 'total.taxes']
    )
    
    hetero_data = builder.build(df)
    
    print("=== Heterogeneous Graph ===")
    print(f"Node types: {hetero_data.node_types}")
    print(f"Edge types: {hetero_data.edge_types}")
    print(f"Transaction features shape: {hetero_data['transaction'].x.shape}")
    print(f"Transaction labels shape: {hetero_data['transaction'].y.shape}")
    print(f"Transaction time shape: {hetero_data['transaction'].time.shape}")
    
    for node_type in hetero_data.node_types:
        if node_type != 'transaction':
            print(f"{node_type} nodes: {hetero_data[node_type].num_nodes}")
    
    return hetero_data


if __name__ == "__main__":
    test_graph_construction()
