# Experimental Results

This directory documents the principal experimental results associated with the study:

**Dual-Timescale Behaviour Modelling for Customer Conversion Prediction Using Hybrid Deep Learning Models in Loyalty and Subscription-Based Systems**

The values below correspond to the final revised manuscript and the reference execution of the reproducibility notebook.

## 1. Primary Holdout Results

The primary evaluation uses a user-level holdout test set retaining the natural class distribution.

### Neural Models

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Dual-Timescale | **0.8750** | **0.9890** | 0.9586 | 0.9992 | **0.9784** | **0.5855** |
| Short-only LSTM | 0.8515 | 0.9854 | 0.9752 | 0.8500 | 0.9083 | 0.3381 |
| Long-only FCNN | 0.8230 | 0.9845 | 0.9595 | 0.9203 | 0.9395 | 0.2863 |

The Dual-Timescale model achieved the strongest overall holdout discrimination and the highest ROC-AUC, PR-AUC, F1-score, and MCC among the evaluated models.

### Traditional Machine-Learning Baselines

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Gradient Boosting | **0.8413** | **0.9872** | 0.9633 | 0.9338 | **0.9483** | 0.3148 |
| Random Forest | 0.8403 | 0.9870 | 0.9642 | 0.9313 | 0.9474 | **0.3205** |
| Logistic Regression | 0.7788 | 0.9802 | 0.9738 | 0.7336 | 0.8368 | 0.2222 |

Gradient Boosting was the strongest traditional baseline by ROC-AUC.

## 2. Architectural Ablation

The primary neural-model comparison provides an architectural ablation of the proposed framework.

| Architecture | ROC-AUC |
|---|---:|
| Dual-Timescale | **0.8750** |
| Short-only LSTM | 0.8515 |
| Long-only FCNN | 0.8230 |

Removing the long-term branch reduced ROC-AUC by approximately **0.0235**, while removing the short-term branch reduced ROC-AUC by approximately **0.0520**.

These results indicate that both temporal components contribute useful predictive information, with the short-term sequential component making the larger contribution to discrimination in the primary experiment.

## 3. 10-Fold GroupKFold Validation

Cross-validation uses `user_id` as the grouping variable to prevent customer-level leakage between training and validation partitions.

### Dual-Timescale Model

| Metric | Mean ± SD |
|---|---:|
| ROC-AUC | **0.8710 ± 0.0034** |
| PR-AUC | **0.9885 ± 0.0004** |
| F1-score | **0.9778 ± 0.0009** |
| MCC | **0.5754 ± 0.0161** |

### Gradient Boosting

| Metric | Mean ± SD |
|---|---:|
| ROC-AUC | **0.8414 ± 0.0016** |

## 4. Statistical Comparison

The fold-wise ROC-AUC values of the Dual-Timescale model were compared with those of Gradient Boosting.

| Statistic | Result |
|---|---:|
| Mean Dual-Timescale ROC-AUC | 0.8710 |
| Mean Gradient Boosting ROC-AUC | 0.8414 |
| Mean ROC-AUC improvement | **0.0296** |
| Paired t-statistic | **32.17** |
| Paired t-test | **p < 0.001** |
| Bootstrap 95% CI | **[0.0279, 0.0314]** |

The bootstrap confidence interval was estimated using 2,000 iterations.

The positive confidence interval and paired statistical test support a consistent ROC-AUC improvement of the Dual-Timescale model over Gradient Boosting across the GroupKFold validation folds.

## 5. Sequence-Length Sensitivity Analysis

A retrospective sensitivity analysis evaluated maximum retained sequence lengths of:

`T = 5, 10, 20, 30, 50`

| T | Short ROC-AUC | Dual ROC-AUC | Short PR-AUC | Dual PR-AUC |
|---:|---:|---:|---:|---:|
| 5 | 0.8879 | **0.9257** | 0.9305 | **0.9607** |
| 10 | 0.8519 | **0.8875** | 0.9299 | **0.9524** |
| 20 | 0.8378 | **0.8689** | 0.9287 | **0.9479** |
| 30 | 0.8371 | **0.8779** | 0.9288 | **0.9523** |
| 50 | 0.8367 | **0.8748** | 0.9282 | **0.9511** |

The Dual-Timescale model achieved higher ROC-AUC and PR-AUC than the Short-only model at every evaluated sequence length.

The Dual-Timescale ROC-AUC advantage ranged from approximately **0.0311 to 0.0407** across the tested values of T.

### Important Interpretation

This analysis is **retrospective and supplementary**.

Although `T = 5` produced the highest ROC-AUC within this internal sensitivity analysis, the primary experiment remains fixed at **T = 50**. The primary holdout configuration was not retuned using these supplementary results.

Therefore, the sensitivity-analysis values should not be interpreted as replacements for the primary holdout result of **ROC-AUC = 0.8750**.

## 6. Class-Imbalance Robustness Analysis

A supplementary robustness analysis used order/time-step class-weighted masked binary cross-entropy rather than sequence-level resampling.

| Model | ROC-AUC | PR-AUC | F1-score | MCC |
|---|---:|---:|---:|---:|
| Dual-Timescale | **0.8724** | **0.9887** | **0.9303** | **0.3759** |
| Short-only LSTM | 0.8513 | 0.9853 | 0.6046 | 0.1771 |
| Long-only FCNN | 0.8232 | 0.9845 | 0.8069 | 0.2490 |

The Dual-Timescale ROC-AUC remained close to the primary holdout result:

- Primary Dual ROC-AUC: **0.8750**
- Class-weighted Dual ROC-AUC: **0.8724**
- Difference: approximately **−0.0026**

This supplementary analysis supports the stability of ranking/discrimination while also showing that threshold-dependent metrics can be more sensitive to the imbalance-handling strategy.

## 7. Explainability Results

SHAP analysis was applied to the **Gradient Boosting baseline** using the engineered long-term behavioural features.

Mean absolute SHAP values ranked the features as follows:

| Rank | Feature | Mean \|SHAP\| |
|---:|---|---:|
| 1 | `cum_order_count` | 0.9728 |
| 2 | `cum_avg_days` | 0.7721 |
| 3 | `cum_std_basket` | 0.2018 |
| 4 | `cum_reorder_ratio` | 0.2007 |
| 5 | `cum_avg_basket` | 0.1942 |
| 6 | `cum_std_days` | 0.1483 |

Mean absolute SHAP values quantify the **magnitude** of feature contributions and should not by themselves be interpreted as indicating positive or negative direction.

The SHAP analysis provides feature-level interpretability for the engineered long-term variables. It does not directly explain the internal sequential representations learned by the LSTM or the complete Dual-Timescale fusion architecture.

## 8. Interpretation

Across the primary holdout evaluation, architectural ablation, GroupKFold validation, statistical testing, and supplementary sensitivity analyses, the results indicate that combining short-term sequential behaviour with long-term cumulative behavioural history provides complementary predictive information.

The Short-only LSTM also exceeded the traditional baselines in holdout ROC-AUC, while the Long-only FCNN performed below the tree-based baselines in ROC-AUC. The strongest performance was obtained when both temporal representations were combined in the Dual-Timescale architecture.

## Reproducibility Note

These values correspond to the reference execution associated with the revised manuscript.

Deep-learning experiments can exhibit small numerical differences across hardware, operating systems, package builds, and stochastic operations. Execution times are also hardware-dependent.

For implementation details and instructions, return to the [main repository README](../README.md).
