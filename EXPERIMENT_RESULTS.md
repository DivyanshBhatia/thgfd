# GNN Experiment Results

## Experimental Validation Summary

**Date:** April 1, 2026

This document contains the results from running comprehensive GNN experiments on fraud detection datasets to validate the Effective Graph Utility framework.

---

## 1. Suggested AUC Recalibration Formula

A linear regression analysis on the experimental data suggests the following formula for predicting baseline AUC:

```
Predicted_AUC = 0.611 × Feature_Pred
              + -0.038 × Fraud_Homo
              + 0.076 × Fraud_Rate
              + 0.398
```

**Model Fit:** R² = 0.6095

### Interpretation:
- **Feature_Pred** (coefficient: 0.611): Strongest predictor of AUC performance
- **Fraud_Homo** (coefficient: -0.038): Minor negative effect on baseline AUC
- **Fraud_Rate** (coefficient: 0.076): Slight positive effect on AUC

---

## 2. Decision Rule Validation

**Decision Rule Accuracy: 75.0%**

The decision rule for choosing between GNN and XGBoost based on Effective Graph Utility achieves 75% accuracy on the test datasets.

### Decision Rule:
- If `Effective_GU > 0.01` → Recommend **GNN**
- Otherwise → Recommend **XGBoost**

Where: `Effective_GU = Graph_Utility × (1 - XGB_AUC)`

---

## 3. Correlation Analysis

| Correlation | Pearson r | p-value | Significant? |
|-------------|-----------|---------|--------------|
| Effective_GU ↔ GNN_Improvement | 0.6434 | 0.0240 | ✅ Yes (p < 0.05) |

### Key Finding:
The correlation between Effective Graph Utility and GNN improvement is **statistically significant** (p = 0.0240), validating that:
- Higher Effective_GU → Greater potential for GNN improvement over XGBoost
- The Effective_GU metric can be used as a reliable predictor for model selection

---

## 4. Datasets Tested

| Dataset | Graph Utility | XGB AUC | Expected Winner |
|---------|---------------|---------|-----------------|
| Ecommerce | 0.06 | 0.7808 | GNN |
| Vehicle Loan | 0.05 | 0.6626 | GNN |
| IP Blocklist | 0.2895 | 0.9116 | GNN |
| Customs | 0.11 | 0.9997 | XGBoost |
| Yelp | 0.10 | 0.9348 | Either |
| Twitter Bots | 0.09 | 0.9370 | Either |
| Fake Job | 0.10 | 0.9738 | Either |
| Amazon | 0.12 | 0.9769 | XGBoost |
| Elliptic | 0.15 | 0.9962 | XGBoost |
| Credit Card | 0.05 | 0.9592 | XGBoost |
| CC Transactions | 0.07 | 0.9885 | XGBoost |
| Malicious URLs | 0.04 | 0.9885 | XGBoost |

---

## 5. Conclusions

1. **The Effective Graph Utility framework is validated** with statistically significant correlation (r=0.6434, p<0.05)

2. **Decision rule achieves 75% accuracy** for model selection between GNN and XGBoost

3. **Feature predictiveness is the dominant factor** in baseline AUC (coefficient 0.611)

4. **Fraud homophily has minimal direct impact** on baseline AUC but influences GNN improvement potential

---

## 6. Next Steps

- [ ] Run experiments on additional datasets
- [ ] Fine-tune GNN hyperparameters for low-performing cases
- [ ] Investigate failure cases where high Graph Utility didn't translate to improvement
- [ ] Test hybrid model approaches

---

## Reproducibility

To reproduce these results:

```bash
cd paper_works/thgfd
pip install -r requirements.txt
python complete_gnn_experiments.py --datasets all --cv_folds 3
