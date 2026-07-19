# Term Deposit Marketing — Two-Layer Calling Model

Predicting which customers are most likely to say **yes** to a term deposit offer, using a two-stage machine learning funnel that mirrors how a real call center should operate: screen cheaply first with information available before any contact, then refine with richer signal once some contact history exists.

## The problem

Calling every customer in a marketing list wastes time on people who were never going to convert. Calling too few misses real subscribers. This project builds a system — not just a single model — that filters a 40,000-customer pool down to a high-confidence call list in two deliberate stages, while staying honest about the trade-offs at every step.

## Repository structure

```
├── Term_Deposit_EDA.ipynb          # Exploratory data analysis on the raw dataset
├── First_Layer_Modelling.ipynb     # Layer 1: Logistic Regression + CatBoost stacked ensemble
├── Second_Layer_Modelling.ipynb    # Layer 2: XGBoost refinement + customer segmentation
├── catboost_info/                  # CatBoost training logs/artifacts
└── .gitignore
```

## Pipeline overview

### 1. Exploratory Data Analysis (`Term_Deposit_EDA.ipynb`)
Initial exploration of the raw term deposit marketing dataset (40,000 customers) — feature distributions, class imbalance (roughly 9% baseline conversion rate), and identification of which features are legitimately available *before* any customer contact versus which only exist *after* a call has taken place (e.g., call duration).

### 2. Layer 1 — Cold Screening (`First_Layer_Modelling.ipynb`)
A stacked ensemble of **Logistic Regression** and **CatBoost**, blended by a meta-learner trained on out-of-fold predictions (5-fold cross-validation) to prevent leakage between the base models and the stack.

**Features used (7):** day, month, contact channel, age, marital status, balance, housing status — deliberately restricted to information available before any contact is made.

**Results:**
- 5-fold CV AUC — LR: 0.678, CatBoost: 0.712 (individually)
- Stack test AUC: 0.729
- At the selected threshold: **96.1% recall** on true "yes" customers, **9.0% precision**
- Correctly filtered out 9,110 of 9,223 true "no" customers, missing only 113 real subscribers
- Narrowed the pool from 40,000 → 30,777 candidates (76.9%)

### 3. Layer 2 — Informed Refinement (`Second_Layer_Modelling.ipynb`)
An **XGBoost** classifier trained on the Layer 1 output, now with access to features that weren't available (or weren't fair to use) at the first screening stage: job, education, loan status, campaign contact count, and call duration from prior contact history.

**Results:**
- 5-fold CV AUC: 0.9405 ± 0.0034 (test AUC: 0.9414)
- At the auto-selected threshold (0.28): **96.2% recall**, **30.4% precision** on the test set
- Applied to the full Layer 1 pool: narrowed 30,777 candidates → **8,589** final call list, with recall holding at **97.8%** and precision improving to **31.7%** — more than 3x concentration of real subscribers versus the raw baseline
- Top predictive feature: `duration`, followed by campaign-timing signals — confirming the value of the two-layer design, since these signals structurally weren't available to Layer 1

### 4. Customer Segmentation (within `Second_Layer_Modelling.ipynb`)
UMAP dimensionality reduction + Ward-linkage hierarchical clustering applied to the final 8,589-customer call list, scanning k=2–10 and selecting by silhouette score (0.699 at k=10 — noted as sitting at the edge of the tested range, worth widening in future iterations).

**Result:** 10 customer segments with conversion rates ranging from **26.9% to 42.9%**, enabling call prioritization within an already-strong list — e.g., divorced management professionals contacted in March (42.9% conversion) versus the largest segment, a blue-collar May-contacted cohort (26.9% conversion, 1,728 customers).

## Key design principle: no leakage, structurally enforced

Call duration and campaign contact count are excluded from Layer 1 not as an afterthought, but because the architecture makes it impossible to use information that wouldn't exist yet in a live deployment — you can't know how long a call will last before you make it. Layer 2 can use these features because, by the time it runs, some contact history already exists.

## Honest limitations

- The original target thresholds (80% yes-recall AND 50% no-recall simultaneously) were found to be mutually unachievable across the full threshold scan — the selected threshold is the best available compromise, not a target that was fully met.
- Layer 1 precision (9.0%) is low by design — a consequence of the dataset's inherent class imbalance (~9% baseline conversion), not a modeling flaw.
- The clustering's optimal k sat at the edge of the tested range (k=10 was the maximum scanned) — treat the 10-segment result as a strong starting point, not a finalized answer.

## Dependencies

```
pandas
numpy
scikit-learn
catboost
xgboost
duckdb
umap-learn
matplotlib
```

## Results summary

| Stage | Pool Size | Conversion Rate | Recall (Yes) |
|---|---|---|---|
| Raw dataset | 40,000 | ~9.0% | — |
| Layer 1 output | 30,777 | 9.0% | 96.1% |
| Layer 2 output | 8,589 | 31.7% | 97.8% |

A companion Power BI dashboard built on top of this pipeline translates the confusion matrices into plain-language cards, visualizes the class-balance progression and demographic impact of each filtering stage, and quantifies the operational payoff in hours and work-weeks saved.