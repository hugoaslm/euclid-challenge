# Starting kit & baselines

A starting kit is provided to help participants begin quickly.

It contains:

- A notebook with exploratory data analysis
- A baseline model
- Utilities for generating submission files

---

# Exploratory Data Analysis

The notebook introduces the dataset and explores:

- dataset size and structure
- missing value patterns
- class imbalance
- relationships between photometric features and the target

One important visualization is the **color-magnitude diagram**.

In galaxy evolution studies, colors trace stellar populations:

- **Blue galaxies** tend to be star-forming
- **Red galaxies** tend to be quenched

Plotting color versus magnitude often reveals two galaxy populations known as the **blue cloud** and the **red sequence**.

This visualization helps confirm that the classification task is physically meaningful.

---

# Baseline model

The starting kit provides a simple baseline model based on **scikit-learn**.

The baseline pipeline:

1. Handles missing values using median imputation
2. Standardizes features
3. Trains a logistic regression classifier

The model outputs **probabilities** required by the evaluation metrics.

---

# Suggested approaches

Participants may improve upon the baseline using:

- gradient boosting models (LightGBM, XGBoost, CatBoost)
- feature engineering (colors, magnitude differences)
- handling missing values more carefully
- probability calibration

Tree-based models often perform well on **tabular datasets** such as this one.

---

# Submission format

Participants must submit a CSV file containing:

| column | description |
|------|------|
| object_id | galaxy identifier |
| y_pred | predicted probability that the galaxy is quenched |

Example:
object_id,y_pred
12345,0.21
12346,0.83
12347,0.14

---

# Tips for participants

- The dataset is **imbalanced**
- Some features contain **missing values**
- The test sets contain galaxies at **higher redshift**

Models must therefore generalize to slightly different data distributions.