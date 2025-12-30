# TH-GFD: Temporal-Heterogeneous Graph Neural Networks for Fraud Detection

This repository contains the implementation of TH-GFD for semi-supervised fraud detection on temporal heterogeneous graphs.

## 📁 Project Structure

```
thgfd_code/
├── configs/
│   └── config.py          # All hyperparameters and configurations
├── data/
│   ├── data_loader.py     # Data loading utilities
│   └── graph_builder.py   # Graph construction from tabular data
├── models/
│   ├── thgfd.py           # TH-GFD model implementation
│   └── baselines.py       # Baseline models (XGBoost, GCN, GraphSAGE, GAT)
├── experiments/
│   └── train_thgfd.py     # Complete training pipeline with pretraining
├── quick_start.py         # Quick test script (run this first!)
├── run_experiment.py      # Full experiment runner
└── requirements.txt       # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torch-geometric torch-scatter torch-sparse
pip install numpy pandas scikit-learn xgboost tqdm
```

### 2. Run Quick Test

```bash
cd thgfd_code
python quick_start.py
```

This will:
- Create synthetic customs data
- Build a temporal-heterogeneous graph
- Train and evaluate XGBoost, GraphSAGE, and TH-GFD
- Print comparison results

### 3. Run Full Experiments

```bash
# Default: 5% label ratio on customs data
python run_experiment.py

# Different label ratios
python run_experiment.py --label_ratio 0.01
python run_experiment.py --label_ratio 0.10

# Use your own data
python run_experiment.py --data_path /path/to/your/customs.csv
```

## 📊 Expected Data Format

Your CSV should have these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `sgd.date` | Transaction date (YY-MM-DD) | `13-01-15` |
| `importer.id` | Importer identifier | `IMP000123` |
| `declarant.id` | Customs broker ID | `DEC0045` |
| `country` | Country of origin | `CNTRY023` |
| `office.id` | Processing office | `OFFICE03` |
| `quantity` | Quantity of goods | `150` |
| `gross.weight` | Weight in kg | `2500.5` |
| `fob.value` | FOB value | `45000.00` |
| `cif.value` | CIF value | `52000.00` |
| `total.taxes` | Tax amount | `5200.00` |
| `illicit` | Fraud label (0/1) | `1` |
| `revenue` | Recovered revenue | `3500.00` |

## 🔬 Model Architecture

### Temporal-Heterogeneous Message Passing

```
                    ┌─────────────────┐
                    │   Transaction   │
                    │     Features    │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         ┌────────┐    ┌──────────┐   ┌─────────┐
         │Importer│    │Declarant │   │ Country │
         └────┬───┘    └────┬─────┘   └────┬────┘
              │             │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │ Temporal-Aware  │
                    │ Attention Aggr. │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         ┌────────┐    ┌──────────┐   ┌─────────┐
         │ Fraud  │    │ Revenue  │   │Embedding│
         │ Logits │    │   Pred   │   │  Output │
         └────────┘    └──────────┘   └─────────┘
```

### Training Pipeline

1. **Pre-training** (Self-supervised)
   - Temporal consistency: Same node at different times → similar embeddings
   - Cross-relation consistency: Same node across relation views → similar embeddings
   - Neighborhood preservation: Connected nodes → similar embeddings

2. **Fine-tuning** (Semi-supervised)
   - Classification loss on labeled data
   - Revenue prediction loss (multi-task)
   - Pseudo-labeling with curriculum learning

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| AUC | Area under ROC curve |
| F1 | Harmonic mean of precision/recall |
| Precision@k% | Precision in top k% predictions |
| Recall@k% | Fraction of frauds in top k% |
| Revenue@k% | Fraction of revenue recovered at k% |

## 🎛️ Key Hyperparameters

```python
# Model
hidden_dim = 64          # Embedding dimension
num_layers = 2           # GNN layers
num_heads = 4            # Attention heads
dropout = 0.3            # Dropout rate

# Pre-training
pretrain_epochs = 100
temperature = 0.5        # Contrastive temperature
beta = 0.5               # Cross-relation loss weight
gamma = 0.5              # Neighbor loss weight

# Fine-tuning
finetune_epochs = 200
lr = 0.005
alpha = 10.0             # Revenue loss weight
mu = 0.1                 # Pseudo-label loss weight
pseudo_threshold = 0.95 → 0.7  # Curriculum schedule
```

## 📝 Adding New Datasets

1. Implement a data loader in `data/data_loader.py`:

```python
class MyDataLoader:
    def load(self):
        # Load your data
        return df
    
    def preprocess(self, df):
        # Feature engineering
        return df, metadata
```

2. Update graph construction in `data/graph_builder.py` if needed

3. Add configuration in `configs/config.py`

## 🧪 Running Ablation Studies

```bash
# Without temporal module
python run_experiment.py --no_temporal

# Without pre-training
python run_experiment.py --no_pretrain

# Different label ratios for label scarcity analysis
for ratio in 0.01 0.02 0.05 0.10 0.20 0.50 1.00; do
    python run_experiment.py --label_ratio $ratio
done
```

## 📚 Citation

If you use this code, please cite:

```bibtex
@inproceedings{thgfd2026,
  title={TH-GFD: Temporal-Heterogeneous Graph Neural Networks for Semi-Supervised Fraud Detection},
  author={Anonymous},
  booktitle={IJCAI},
  year={2026}
}
```

## 🐛 Troubleshooting

**Q: CUDA out of memory**
- Reduce `batch_size` in config
- Use `--device cpu` for CPU training
- Reduce `hidden_dim` or `num_layers`

**Q: Graph construction is slow**
- Use `max_neighbors` parameter to limit edges
- Sample entities for very large categorical columns

**Q: Poor performance**
- Ensure sufficient label ratio (try 5% first)
- Check class imbalance (use `pos_weight`)
- Try longer pre-training
