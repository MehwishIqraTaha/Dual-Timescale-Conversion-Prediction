# Dual-Timescale Behaviour Modelling for Customer Conversion Prediction Using Hybrid Deep Learning Models in Loyalty and Subscription-Based Systems

## Overview

This repository contains the reproducibility materials associated with
the research paper **"Dual-Timescale Behaviour Modelling for Customer
Conversion Prediction Using Hybrid Deep Learning Models in Loyalty and
Subscription-Based Systems."**

The study evaluates a hybrid deep learning framework combining
**short-term behavioural dynamics** modelled with an LSTM and
**long-term cumulative engagement history** modelled with a fully
connected neural network (FCNN). These representations are fused in the
proposed **Dual-Timescale model** for customer continuation prediction.

## Prediction Target

The prediction task is defined as **subsequent observed purchase
behaviour**:

-   `y = 1` if another observed order follows the current order.
-   `y = 0` if the current order is the customer's final observed order
    in the dataset.

The task is therefore best interpreted as **repeat-order /
purchase-continuation prediction**, rather than conventional first-time
customer acquisition.


## Dataset

The experiments use the **Instacart Online Grocery Shopping Dataset**,
containing more than three million orders from more than 200,000 users.

The Instacart data do not contain explicit loyalty-program or
subscription-status variables. Repeated transactional engagement is used
as a behavioural proxy for persistence and retention. Generalisation to
datasets containing explicit loyalty or subscription information remains
an area for future validation.

Repeated customer-level transactions make the dataset suitable for
constructing both sequential short-term behaviour and cumulative
long-term behavioural history.

### Data availability

The raw Instacart dataset is **not redistributed in this repository**.
Users should obtain the original Instacart Market Basket Analysis data
from its authorised public source and place the required files in the
local data location specified by the notebook.

## Behavioural Features

### Short-term sequential features

-   `order_dow`
-   `order_hour_of_day`
-   `days_since_prior_order`
-   `basket_size`
-   `reorder_ratio`

The primary sequence length is capped at the most recent **T = 50
orders**.

### Long-term cumulative features

-   `cum_order_count`
-   `cum_avg_days`
-   `cum_std_days`
-   `cum_avg_basket`
-   `cum_std_basket`
-   `cum_reorder_ratio`

These variables summarise accumulated behavioural history available up
to each prediction point.

## Models

### Traditional machine-learning baselines

-   Logistic Regression
-   Random Forest
-   Gradient Boosting

### Deep-learning models

-   **Short-only LSTM** --- sequential short-term behaviour.
-   **Long-only FCNN** --- cumulative long-term behavioural features.
-   **Dual-Timescale model** --- fusion of the LSTM short-term
    representation and FCNN long-term representation.

The primary Dual-Timescale architecture uses an LSTM hidden dimension of
64, a 64-unit long-term dense representation, ReLU activation, dropout
of 0.2, and a 64-unit fusion layer.

## Training and Evaluation

The primary protocol uses a **user-level holdout split**, preventing
orders belonging to the same user from appearing across training and
test partitions.

Class imbalance is handled through **training-only random
undersampling**, producing a 75% positive / 25% negative training
distribution. The test set retains its natural class distribution.

Primary neural-network settings:

-   Random seed: `42`
-   Optimizer: Adam
-   Learning rate: `0.001`
-   Batch size: `64`
-   Primary sequence cap: `T = 50`
-   Primary holdout training epochs: `10`
-   Fixed classification threshold: `0.50`

Evaluation metrics include ROC-AUC, PR-AUC, Precision, Recall, F1-score,
and Matthews Correlation Coefficient (MCC).

## Cross-Validation and Statistical Analysis

Robustness is evaluated using **10-fold GroupKFold cross-validation**,
grouped by `user_id`. Training-fold data are balanced using the same
training-only undersampling procedure, while validation folds retain
their natural class distribution.

The Dual-Timescale model is statistically compared with the strongest
traditional baseline using fold-wise ROC-AUC values, a paired t-test,
and a bootstrap 95% confidence interval.

The notebook includes checkpoint-safe handling for the computationally
intensive baseline cross-validation stage so completed model-fold
evaluations can be preserved if execution is interrupted.

## Additional Analyses

The notebook includes:

-   Architectural ablation comparing Short-only, Long-only, and
    Dual-Timescale models.
-   Retrospective sequence-length sensitivity analysis for
    `T = 5, 10, 20, 30, 50`.
-   Class-weighted robustness analysis.
-   Correlation analysis of engineered long-term features.
-   Feature importance analysis for the Gradient Boosting baseline.
-   SHAP analysis for the Gradient Boosting baseline.

The SHAP analysis explains the engineered long-term behavioural features
used by the Gradient Boosting baseline. It does **not** directly explain
the internal temporal representations learned by the LSTM or the
complete Dual-Timescale fusion architecture.

## Repository Structure

``` text
dual-timescale-behaviour-modelling/
│
├── README.md
├── DualTimescale_Reproducibility.ipynb
├── requirements.txt
│
├── data/
│   └── README.md
│
└── results/
    └── README.md
```


## Software Environment

The experiments were developed using **Python 3.10.19**. Main libraries
include NumPy, pandas, scikit-learn, PyTorch, SciPy, SHAP, and
Matplotlib.

Exact package requirements are provided in `requirements.txt`.

## Running the Reproducibility Notebook

1.  Obtain the Instacart source data.
2.  Clone or download this repository.
3.  Create a Python environment and install the packages in
    `requirements.txt`.
4.  Place the required Instacart source files in the data location
    specified in the notebook.
5.  Open `DualTimescale_Reproducibility.ipynb`.
6.  Review the configuration/path cell.
7.  Run the notebook sequentially from the beginning.
8.  Allow the computationally intensive cross-validation and
    deep-learning sections to complete.
9.  Review the generated tables, figures, and result files in the
    configured output directories.

### Runtime note

Some stages, particularly 10-fold cross-validation and repeated
neural-network training, can require substantial execution time. Runtime
depends on hardware and available computational resources.

## Reproducibility Notes

NumPy and PyTorch random seeds are set to `42` for the primary
experiments. The notebook uses user-level splitting and grouped
cross-validation to reduce leakage between customer histories.

Sequence-length sensitivity is a retrospective validation analysis. The
primary `T = 50` holdout experiment is retained as the predefined main
experiment and is not retuned using the holdout set.

## Results

The repository is intended to reproduce the experimental workflow and
analyses reported in the associated manuscript.

## Citation

If you use this code or methodology in academic work, please cite the
associated paper. Full journal citation details will be added after
publication.

## Code Availability

The reproducibility notebook, environment information, and supporting
instructions are provided in this repository to support transparent
evaluation and replication of the study.


