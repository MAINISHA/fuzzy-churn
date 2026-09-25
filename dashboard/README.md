# European Bank — Customer Churn Risk Dashboard

An interactive Streamlit dashboard for exploring customer churn risk, built
on top of five candidate models trained on `European_Bank.csv`, with the
best-performing model (by ROC-AUC) automatically used for all live
predictions.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Make sure `European_Bank.csv` stays in the **same folder** as `app.py` — the
app loads it directly and trains all five models on first run (cached
afterwards, so it only trains once per session).

## What's inside

**Sidebar — Customer Profile**
A single customer profile (demographics, account details, engagement &
products) that drives every profile-based module below it. The sidebar also
shows which model is currently best and its ROC-AUC.

**🎯 Dependent & Independent Variables**
A collapsible panel (below the header metrics) that names the target
variable (`Exited`) and lists all predictor variables, split into raw
features and engineered features.

**📋 Executive Summary**
Portfolio-wide churn KPIs computed live from the full dataset (overall churn
rate, balance held by churned customers, highest-risk market, churn rate for
the 3+ product segment), plus key findings and recommended priorities — the
dashboard-native counterpart to the full Word report.

**🏆 Model Comparison**
Trains Logistic Regression, Decision Tree, Random Forest, Gradient Boosting,
and XGBoost on the same feature set and test split, and compares them on
Accuracy, Precision, Recall, F1-Score, and ROC-AUC. The best model (highest
ROC-AUC) is highlighted and automatically used everywhere else in the app.

**🧮 Risk Calculator**
Live churn probability for the current profile from the best model, shown as
a gauge plus a color-coded risk band (Low / Moderate / High / Critical) with
a short recommended action.

**📊 Probability Distribution**
Histogram of the best model's predicted churn probabilities across the
held-out test set, split by actual outcome (churned vs. retained), with the
current customer's position marked and percentile-ranked.

**🔑 Feature Importance**
Global feature importances from the best model — mean decrease in impurity
for tree-based models, or absolute standardized coefficients for a linear
model like Logistic Regression — adjustable to show more or fewer features.

**🎛️ What-If Simulator**
- Single-feature sensitivity sweep: pick a feature (products, activity
  status, age, tenure, balance, credit score, salary) and see a line chart of
  how churn probability changes across its full range (best model), holding
  everything else fixed at the sidebar profile.
- Best-case / worst-case callouts for that sweep.
- A multi-feature adjustment panel to nudge products, activity status,
  balance, and tenure together and see the combined probability shift versus
  the current profile.

## Notes

- All five models are retrained from scratch each time the app starts, using
  the same preprocessing and feature engineering as the original analysis
  notebook (Balance-to-Salary ratio, Product density, Engagement×Product
  interaction, Age×Tenure interaction, etc.). Training all five together
  still runs quickly on this dataset's 10,000 rows.
- The best model is selected by test-set ROC-AUC; the Model Comparison tab
  shows Recall alongside it since that's often the more business-relevant
  metric for catching churners.
- To point the app at a different dataset, replace `European_Bank.csv` with a
  file that has the same column names, or edit `DATA_PATH` in `app.py`.
