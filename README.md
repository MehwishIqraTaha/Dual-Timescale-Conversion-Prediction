# Dual-Timescale Behaviour Modelling for Customer Conversion Prediction

This repository contains the reproducibility materials accompanying the study:

**Dual-Timescale Behaviour Modelling for Customer Conversion Prediction Using Hybrid Deep Learning Models in Loyalty and Subscription-Based Systems**

The study proposes a hybrid Dual-Timescale deep learning framework that combines short-term sequential customer behaviour with long-term cumulative behavioural history for predicting subsequent observed purchase behaviour.

## Overview

Customer behaviour develops across multiple temporal scales. Recent transactions may capture immediate behavioural patterns, while cumulative purchase history represents longer-term engagement.

The proposed framework integrates both sources of information through:

- a Short-Term LSTM encoder for sequential transactional behaviour;
- a Long-Term fully connected neural network (FCNN) encoder for cumulative behavioural features; and
- a fusion layer that combines both representations for binary prediction.

The proposed Dual-Timescale model is evaluated against traditional machine-learning baselines and single-timescale neural architectures.

## Prediction Target

In this study, the binary target represents **subsequent observed purchase behaviour**:

- **y = 1:** the current order is followed by another observed order from the same customer;
- **y = 0:** the current order is the customer's final observed order in the dataset.

Accordingly, the task is most precisely interpreted as **repeat-order or purchase-continuation prediction**, rather than first-time customer acquisition.

The term *conversion* is retained in the accompanying study to describe continued transactional engagement and its potential relevance to retention-oriented loyalty and subscription settings.

## Dataset

Experiments use the **Instacart Online Grocery Shopping Dataset**, originally released for the Instacart Market Basket Analysis challenge.

The dataset contains more than three million grocery orders from over 200,000 customers and provides repeated customer-level transaction histories suitable for temporal behavioural modelling.

The raw dataset is **not redistributed in this repository**.

To reproduce the experiments, obtain the Instacart dataset separately and provide the following files:

```text
orders.csv
order_products__prior.csv
order_products__train.csv
products.csv
aisles.csv
departments.csv
```

Place the files inside a local `data/` directory or configure the data path as described in the reproducibility notebook.

The Instacart dataset does not contain explicit loyalty-program membership or subscription-status variables. It is therefore used as a behavioural testbed for repeated purchasing and continued engagement rather than as direct evidence of formal loyalty or subscription participation.

## Model Architecture

The Dual-Timescale architecture contains two parallel branches.

### Short-Term Branch

The short-term branch uses an LSTM with 64 hidden units to model sequential order-level behaviour.

Short-term features include:

- order day of week;
- order hour of day;
- days since prior order;
- basket size; and
- reorder ratio.

### Long-Term Branch

The long-term branch uses a fully connected neural network with 64 hidden units, ReLU activation, and dropout of 0.20.

Long-term features include:

- cumulative order count;
- cumulative average days between orders;
- cumulative standard deviation of inter-order intervals;
- cumulative average basket size;
- cumulative standard deviation of basket size; and
- cumulative reorder ratio.

These features are calculated cumulatively from the customer's available order history rather than from a fixed-length long-term window.

### Fusion

The 64-dimensional short-term and 64-dimensional long-term representations are concatenated and passed through a 64-unit fusion layer before binary prediction.

## Experimental Configuration

The primary experimental configuration uses:

| Setting | Value |
|---|---|
| Random seed | 42 |
| Primary maximum sequence length (T) | 50 |
| LSTM hidden units | 64 |
| Long-term hidden units | 64 |
| Fusion hidden units | 64 |
| Dropout | 0.20 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Primary holdout epochs | 10 |
| Cross-validation folds | 10 |
| Cross-validation epochs per fold | 5 |
| Decision threshold | 0.50 |

The primary train/test split is performed at the **user level** to prevent the same customer's records from appearing in both training and holdout partitions.

Training data are rebalanced to a 75/25 class distribution using random undersampling, while the holdout test set retains its natural class distribution.

The cross-validation analysis uses **10-fold GroupKFold** with `user_id` as the grouping variable.

## Models Evaluated

Traditional machine-learning baselines:

- Logistic Regression
- Random Forest
- Gradient Boosting

Neural architectures:

- Short-only LSTM
- Long-only FCNN
- Dual-Timescale model

## Primary Results

On the natural-distribution holdout test set, the proposed Dual-Timescale model achieved:

| Metric | Dual-Timescale |
|---|---:|
| ROC-AUC | 0.8750 |
| PR-AUC | 0.9890 |
| F1-score | 0.9784 |
| MCC | 0.5855 |

The strongest traditional baseline, Gradient Boosting, achieved a holdout ROC-AUC of **0.8413**.

Under 10-fold GroupKFold validation:

- Dual-Timescale ROC-AUC: **0.8710 ± 0.0034**
- Gradient Boosting ROC-AUC: **0.8414 ± 0.0016**
- Mean ROC-AUC improvement: **0.0296**
- Paired t-test: **p < 0.001**
- Bootstrap 95% CI for the ROC-AUC improvement: **[0.0279, 0.0314]**

See [`results/README.md`](results/README.md) for a more detailed summary of the reported results and supplementary analyses.

## Ablation and Sensitivity Analysis

The repository also includes analyses examining the contribution of the two temporal branches.

The primary holdout ROC-AUC values were:

- Dual-Timescale: **0.8750**
- Short-only LSTM: **0.8515**
- Long-only FCNN: **0.8230**

A retrospective sequence-length sensitivity analysis evaluates `T = 5, 10, 20, 30, 50`.

This sensitivity analysis is supplementary. The primary holdout configuration remains **T = 50** and was not retuned using the holdout test set.

## Explainability

SHAP analysis is applied to the **Gradient Boosting baseline** using the engineered long-term behavioural features.

The SHAP analysis provides feature-level interpretability for these cumulative variables. It does **not** directly explain the internal temporal representations learned by the LSTM or the complete Dual-Timescale fusion architecture.

## Repository Structure

```text
Dual-Timescale-Conversion-Prediction/
│
├── README.md
├── requirements.txt
├── DualTimescale_Reproducibility.ipynb
│
├── data/
│   └── README.md
│
└── results/
    └── README.md
```

The raw Instacart data are intentionally excluded from the repository.

## Reproducibility

The repository provides the experimental notebook, environment requirements, implementation details, evaluation procedures, and reference outputs associated with the revised manuscript.

The reproducibility notebook covers:

1. data loading and preprocessing;
2. target construction;
3. short- and long-term feature engineering;
4. exploratory analysis;
5. user-level train/test splitting;
6. traditional machine-learning baselines;
7. Short-only, Long-only, and Dual-Timescale neural models;
8. holdout evaluation;
9. 10-fold GroupKFold validation;
10. statistical significance testing;
11. sequence-length sensitivity analysis;
12. class-imbalance robustness analysis; and
13. Gradient Boosting feature-importance and SHAP analysis.

### Reproducibility Status

The notebook has been executed through the complete experimental workflow and contains reference outputs corresponding to the results reported in the revised manuscript.

Numerical differences may occur across hardware, operating systems, library builds, and stochastic deep-learning operations. Execution times are hardware-dependent.

## Environment

The reported experiments were conducted using:

- Python 3.10.19
- NumPy 2.2.5
- pandas 2.3.3
- Matplotlib 3.10.8
- SciPy 1.15.3
- scikit-learn 1.7.1
- PyTorch 2.10.0
- SHAP 0.49.1

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Running the Notebook

1. Clone or download this repository.
2. Install the dependencies listed in `requirements.txt`.
3. Obtain the Instacart dataset separately.
4. Place the required CSV files in the local `data/` directory, or configure `INSTACART_DATA_DIR`.
5. Open `DualTimescale_Reproducibility.ipynb`.
6. Run the notebook sequentially from the beginning.

Generated outputs are written to the configured output directory.

## Code and Data Availability

The implementation and reproducibility materials are publicly available through this repository.

The Instacart dataset is not redistributed and must be obtained separately from its original source.

## Citation

If you use this repository, please cite the accompanying research article.

Full publication details will be added after publication.

## Authors

**Mehwish Iqra Taha**, Muneer Ahmad, Seyed Ebrahim Hosseini, Shahbaz Pervez, Mohsin Iftikhar, and Peer Azmat Shah.

## Research Use

This repository is provided to support research transparency and reproducibility associated with the accompanying study.
