# Evaluation metrics

The competition uses three metrics.

Participants are ranked using the **primary metric**.

---

# Primary Metric  
## Redshift-Stratified Weighted Log Loss

The primary metric evaluates probabilistic predictions using **log loss**.

To ensure fairness across cosmic time, the loss is computed **independently per redshift bin** and then averaged.
s
$$
L = -\frac{1}{N}\sum_i [y_i\log(p_i) + (1-y_i)\log(1-p_i)]
$$

where:

- $y_i$ is the true label
- $p_i$ is the predicted probability

The final score is the **macro-average across redshift bins**.

Lower values are better.

---

# Secondary Metric  
## Area Under Precision-Recall Curve (AUPRC)

Because the dataset is **class-imbalanced**, we evaluate the **precision-recall curve**.

AUPRC measures how well the model retrieves quenched galaxies while maintaining high precision.

Higher values are better.

---

# Tertiary Metric  
## Recall at 85% Precision

This metric measures:

> How many quenched galaxies can be recovered while keeping false positives low.

The recall is evaluated at **85% precision**.

Higher values are better.

---

# Why These Metrics?

The metric combination reflects astrophysical goals:

- **Log Loss** → well-calibrated probabilities
- **AUPRC** → performance under class imbalance
- **Recall@85% precision** → reliable identification of quenched galaxies

Together they evaluate both **statistical accuracy** and **scientific usefulness**.