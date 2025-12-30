"""
Configuration for TH-GFD experiments
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """Data configuration"""
    dataset: str = "customs"  # customs, amazon, elliptic
    data_path: str = "./data/raw/"
    label_ratio: float = 0.05  # 5% labels available
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    temporal_split: bool = True  # Use temporal split for train/val/test
    random_seed: int = 42


@dataclass
class GraphConfig:
    """Graph construction configuration"""
    # Entity types to use as virtual nodes
    entity_columns: List[str] = field(default_factory=lambda: [
        'importer.id', 'declarant.id', 'office.id', 'country'
    ])
    # Numerical features
    numerical_columns: List[str] = field(default_factory=lambda: [
        'quantity', 'gross.weight', 'fob.value', 'cif.value', 'total.taxes'
    ])
    # Use GBDT for feature extraction (like GraphFC)
    use_gbdt_features: bool = True
    n_trees: int = 100
    max_depth: int = 4


@dataclass 
class ModelConfig:
    """Model architecture configuration"""
    # Embedding dimensions
    hidden_dim: int = 64
    embedding_dim: int = 64
    
    # GNN architecture
    num_layers: int = 2
    num_heads: int = 4  # For attention
    dropout: float = 0.3
    
    # Temporal parameters
    use_temporal: bool = True
    temporal_decay_init: float = 0.1  # Initial lambda for temporal decay
    
    # Heterogeneous parameters
    use_heterogeneous: bool = True
    num_relation_types: int = 4  # Number of edge/relation types


@dataclass
class PretrainConfig:
    """Self-supervised pre-training configuration"""
    enabled: bool = True
    epochs: int = 100
    lr: float = 0.005
    batch_size: int = 512
    
    # Contrastive learning
    temperature: float = 0.5
    beta: float = 0.5  # Weight for cross-relation loss
    gamma: float = 0.5  # Weight for neighbor preservation loss
    
    # Negative sampling
    num_negative_samples: int = 5


@dataclass
class TrainConfig:
    """Fine-tuning configuration"""
    epochs: int = 200
    lr: float = 0.005
    weight_decay: float = 1e-4
    batch_size: int = 512
    
    # Dual-task learning
    alpha: float = 10.0  # Weight for revenue prediction loss
    
    # Pseudo-labeling
    use_pseudo_labels: bool = True
    pseudo_threshold_start: float = 0.95
    pseudo_threshold_end: float = 0.7
    mu: float = 0.1  # Weight for pseudo-label loss
    
    # Early stopping
    patience: int = 20
    
    # Neighborhood sampling
    neighbor_sizes: List[int] = field(default_factory=lambda: [25, 10])


@dataclass
class Config:
    """Main configuration combining all sub-configs"""
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    # General
    device: str = "cuda"  # cuda or cpu
    seed: int = 42
    exp_name: str = "thgfd_experiment"
    save_dir: str = "./checkpoints/"


def get_config(dataset: str = "customs", label_ratio: float = 0.05) -> Config:
    """Get configuration for specific dataset and label ratio"""
    config = Config()
    config.data.dataset = dataset
    config.data.label_ratio = label_ratio
    
    # Dataset-specific adjustments
    if dataset == "customs":
        config.graph.entity_columns = ['importer.id', 'declarant.id', 'office.id', 'country']
        config.model.num_relation_types = 4
        config.model.use_temporal = True
        
    elif dataset == "amazon":
        config.graph.entity_columns = []  # Amazon already has graph structure
        config.model.num_relation_types = 3  # U-P-U, U-S-U, U-V-U
        config.model.use_temporal = False  # No temporal info
        
    elif dataset == "elliptic":
        config.graph.entity_columns = ['wallet_id']  # If using wallet info
        config.model.num_relation_types = 2
        config.model.use_temporal = True
    
    return config


# Preset configurations for experiments
CONFIGS = {
    "customs_5pct": get_config("customs", 0.05),
    "customs_1pct": get_config("customs", 0.01),
    "customs_10pct": get_config("customs", 0.10),
    "amazon_5pct": get_config("amazon", 0.05),
    "elliptic_5pct": get_config("elliptic", 0.05),
}
