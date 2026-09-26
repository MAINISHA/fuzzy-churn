"""
European Bank — Customer Churn Risk Dashboard
Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Risk Dashboard",
    page_icon="📉",
    layout="wide",
)

from pathlib import Path

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "European_Bank.csv"

NUMERIC_FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary",
    "BalanceSalaryRatio", "ProductDensity", "EngagementProductInteraction",
    "AgeTenureInteraction", "CreditScoreAgeRatio",
]

# --------------------------------------------------------------------------
# Data / feature engineering / model training (cached)
# --------------------------------------------------------------------------
@st.cache_data
def load_raw_data(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Check that data/European_Bank.csv "
            "is committed to GitHub."
        )

    df = pd.read_csv(path)

    required_columns = [
        "CreditScore", "Geography", "Gender", "Age",
        "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember",
        "EstimatedSalary", "Exited"
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    return df


def engineer_features(row_df):
    """Apply the same feature engineering used in training to any dataframe
    of raw customer rows (CreditScore, Geography, Gender, Age, Tenure,
    Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)."""
    d = row_df.copy()
    eps = 1
    d["BalanceSalaryRatio"] = d["Balance"] / (d["EstimatedSalary"] + eps)
    d["ProductDensity"] = d["NumOfProducts"] / (d["Tenure"] + eps)
    d["EngagementProductInteraction"] = d["IsActiveMember"] * d["NumOfProducts"]
    d["AgeTenureInteraction"] = d["Age"] * d["Tenure"]
    d["HasBalance"] = (d["Balance"] > 0).astype(int)
    d["CreditScoreAgeRatio"] = d["CreditScore"] / (d["Age"] + eps)
    return d


@st.cache_resource
def train_models(path):
    df = load_raw_data(path)

    drop_cols = [c for c in ["CustomerId", "Surname"] if c in df.columns]
    constant_cols = [c for c in df.columns if df[c].nunique() == 1]
    drop_cols += [c for c in constant_cols if c not in drop_cols]
    data = df.drop(columns=drop_cols)

    data = engineer_features(data)
    data = pd.get_dummies(data, columns=["Geography", "Gender"], drop_first=True)

    bool_cols = data.select_dtypes(include="bool").columns
    data[bool_cols] = data[bool_cols].astype(int)

    X = data.drop(columns=["Exited"])
    y = data["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_s = X_train.copy()
    X_test_s = X_test.copy()
    X_train_s[NUMERIC_FEATURES] = scaler.fit_transform(X_train[NUMERIC_FEATURES])
    X_test_s[NUMERIC_FEATURES] = scaler.transform(X_test[NUMERIC_FEATURES])

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model_specs = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    trained_models = {}
    metrics_rows = []
    proba_by_model = {}

    for name, model in model_specs.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1]

        trained_models[name] = model
        proba_by_model[name] = y_proba
        metrics_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        })

    metrics_df = pd.DataFrame(metrics_rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    best_model_name = metrics_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    return {
        "models": trained_models,
        "metrics_df": metrics_df,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "scaler": scaler,
        "feature_columns": list(X.columns),
        "X_test": X_test_s,
        "y_test": y_test,
        "test_proba": proba_by_model[best_model_name],
        "proba_by_model": proba_by_model,
        "test_auc": metrics_df.iloc[0]["ROC-AUC"],
        "geography_levels": [c for c in X.columns if c.startswith("Geography_")],
        "gender_levels": [c for c in X.columns if c.startswith("Gender_")],
    }


def build_feature_row(inputs, artifacts):
    """Turn a dict of raw customer inputs into a fully engineered, scaled,
    one-hot-encoded row matching the training feature columns."""
    raw = pd.DataFrame([{
        "CreditScore": inputs["CreditScore"],
        "Age": inputs["Age"],
        "Tenure": inputs["Tenure"],
        "Balance": inputs["Balance"],
        "NumOfProducts": inputs["NumOfProducts"],
        "HasCrCard": inputs["HasCrCard"],
        "IsActiveMember": inputs["IsActiveMember"],
        "EstimatedSalary": inputs["EstimatedSalary"],
    }])
    eng = engineer_features(raw)

    for col in artifacts["geography_levels"]:
        level = col.replace("Geography_", "")
        eng[col] = 1 if inputs["Geography"] == level else 0
    for col in artifacts["gender_levels"]:
        level = col.replace("Gender_", "")
        eng[col] = 1 if inputs["Gender"] == level else 0

    eng = eng.reindex(columns=artifacts["feature_columns"], fill_value=0)
    eng[NUMERIC_FEATURES] = artifacts["scaler"].transform(eng[NUMERIC_FEATURES])
    return eng


def predict_proba(inputs, artifacts, model=None):
    row = build_feature_row(inputs, artifacts)
    use_model = model if model is not None else artifacts["best_model"]
    return float(use_model.predict_proba(row)[:, 1][0])


def get_feature_importance(model, feature_columns):
    """Return a Series of importances regardless of model type. Tree-based
    models expose feature_importances_; linear models expose coef_."""
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_columns), "impurity"
    elif hasattr(model, "coef_"):
        return pd.Series(np.abs(model.coef_[0]), index=feature_columns), "coefficient"
    return None, None


def risk_band(p):
    if p < 0.25:
        return "Low", "#2E7D32"
    elif p < 0.50:
        return "Moderate", "#C9A24B"
    elif p < 0.75:
        return "High", "#E07B2A"
    else:
        return "Critical", "#B3402F"


# --------------------------------------------------------------------------
# Load / train
# --------------------------------------------------------------------------
artifacts = train_models(DATA_PATH)
raw_df = load_raw_data(DATA_PATH)

# --------------------------------------------------------------------------
# Sidebar — shared customer profile (used by all modules)
# --------------------------------------------------------------------------
st.sidebar.title("👤 Customer Profile")
st.sidebar.caption("This profile feeds every module in the dashboard.")

if "profile" not in st.session_state:
    st.session_state.profile = {
        "Geography": "France", "Gender": "Female", "CreditScore": 650,
        "Age": 40, "Tenure": 5, "Balance": 75000.0, "NumOfProducts": 1,
        "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 100000.0,
    }

p = st.session_state.profile

with st.sidebar.expander("Demographics", expanded=True):
    p["Geography"] = st.selectbox("Geography", ["France", "Germany", "Spain"],
                                   index=["France", "Germany", "Spain"].index(p["Geography"]))
    p["Gender"] = st.selectbox("Gender", ["Female", "Male"],
                                index=["Female", "Male"].index(p["Gender"]))
    p["Age"] = st.slider("Age", 18, 92, p["Age"])

with st.sidebar.expander("Account Details", expanded=True):
    p["CreditScore"] = st.slider("Credit Score", 350, 850, p["CreditScore"])
    p["Tenure"] = st.slider("Tenure (years with bank)", 0, 10, p["Tenure"])
    p["Balance"] = st.number_input("Account Balance ($)", 0.0, 300000.0, p["Balance"], step=1000.0)
    p["EstimatedSalary"] = st.number_input("Estimated Salary ($)", 0.0, 300000.0, p["EstimatedSalary"], step=1000.0)

with st.sidebar.expander("Engagement & Products", expanded=True):
    p["NumOfProducts"] = st.slider("Number of Products", 1, 4, p["NumOfProducts"])
    p["HasCrCard"] = 1 if st.checkbox("Has Credit Card", value=bool(p["HasCrCard"])) else 0
    p["IsActiveMember"] = 1 if st.checkbox("Active Member", value=bool(p["IsActiveMember"])) else 0

st.sidebar.divider()
st.sidebar.metric("Best Model", artifacts["best_model_name"])
st.sidebar.metric("Best Model ROC-AUC", f"{artifacts['test_auc']:.3f}")
st.sidebar.caption(
    f"Live predictions use {artifacts['best_model_name']}, the top performer of five "
    "models trained on European_Bank.csv (10,000 customers). See the Model Comparison "
    "tab for the full breakdown."
)

current_proba = predict_proba(p, artifacts)
band, band_color = risk_band(current_proba)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("📉 European Bank — Customer Churn Risk Dashboard")
st.caption("Adjust the customer profile in the sidebar and explore the summary and modules below.")

hcol1, hcol2, hcol3, hcol4 = st.columns(4)
hcol1.metric("Churn Probability", f"{current_proba*100:.1f}%")
hcol2.metric("Risk Band", band)
hcol3.metric("Portfolio Avg. Churn", f"{raw_df['Exited'].mean()*100:.1f}%")
hcol4.metric("Delta vs. Portfolio Avg.", f"{(current_proba - raw_df['Exited'].mean())*100:+.1f} pp")

with st.expander("🎯 Dependent & Independent Variables", expanded=False):
    vcol1, vcol2 = st.columns([1, 2])

    with vcol1:
        st.markdown("**Dependent Variable (Target)**")
        st.dataframe(
            pd.DataFrame({
                "Variable": ["Exited"],
                "Meaning": ["Customer churned (1) vs. retained (0)"],
            }),
            hide_index=True, use_container_width=True,
        )

    with vcol2:
        st.markdown("**Independent Variables (Predictors)**")
        raw_features = ["CreditScore", "Geography", "Gender", "Age", "Tenure",
                         "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
                         "EstimatedSalary"]
        engineered_features = ["BalanceSalaryRatio", "ProductDensity",
                                "EngagementProductInteraction", "AgeTenureInteraction",
                                "HasBalance", "CreditScoreAgeRatio"]
        var_df = pd.DataFrame({
            "Variable": raw_features + engineered_features,
            "Type": ["Raw"] * len(raw_features) + ["Engineered"] * len(engineered_features),
        })
        st.dataframe(var_df, hide_index=True, use_container_width=True, height=250)

    st.caption(
        "Geography and Gender are one-hot encoded before modeling; CustomerId, Surname, "
        "and Year are dropped as non-informative. All numeric predictors are standardized."
    )

st.divider()

tab0, tab_models, tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Executive Summary",
    "🏆 Model Comparison",
    "🧮 Risk Calculator",
    "📊 Probability Distribution",
    "🔑 Feature Importance",
    "🎛️ What-If Simulator",
])

# ==========================================================================
# TAB 0 — Executive Summary
# ==========================================================================
with tab0:
    st.subheader("Executive Summary")
    st.write(
        "Portfolio-wide churn findings from the full customer base "
        f"({len(raw_df):,} customers)."
    )

    overall_churn = raw_df["Exited"].mean()
    total_balance = raw_df["Balance"].sum()
    churned_balance = raw_df.loc[raw_df["Exited"] == 1, "Balance"].sum()
    balance_at_risk_pct = churned_balance / total_balance if total_balance else 0

    geo_churn = raw_df.groupby("Geography")["Exited"].mean().sort_values(ascending=False)
    top_geo, top_geo_rate = geo_churn.index[0], geo_churn.iloc[0]

    prod_churn = raw_df.groupby("NumOfProducts")["Exited"].mean()
    high_prod_customers = raw_df[raw_df["NumOfProducts"] >= 3]
    high_prod_rate = high_prod_customers["Exited"].mean() if len(high_prod_customers) else 0

    active_churn = raw_df.groupby("IsActiveMember")["Exited"].mean()
    inactive_rate = active_churn.get(0, 0)
    active_rate = active_churn.get(1, 0)

    ecol1, ecol2, ecol3, ecol4 = st.columns(4)
    ecol1.metric("Overall Churn Rate", f"{overall_churn*100:.1f}%")
    ecol2.metric("Balance Held by Churned Customers",
                 f"${churned_balance/1e6:,.1f}M", f"{balance_at_risk_pct*100:.1f}% of total")
    ecol3.metric(f"{top_geo} Churn Rate (highest market)", f"{top_geo_rate*100:.1f}%")
    ecol4.metric("Churn Rate — Customers with 3+ Products", f"{high_prod_rate*100:.1f}%")

    st.markdown("##### Key Findings")
    st.markdown(
        f"- **Geographic concentration:** {top_geo} has the highest churn rate at "
        f"{top_geo_rate*100:.1f}%, versus {geo_churn.iloc[-1]*100:.1f}% in "
        f"{geo_churn.index[-1]}.\n"
        f"- **Product over-exposure:** churn among customers holding 3 or more products "
        f"({len(high_prod_customers):,} customers) reaches {high_prod_rate*100:.1f}%, "
        f"far above the {prod_churn.get(2,0)*100:.1f}% churn rate for the 2-product segment — "
        f"the healthiest cohort in the book.\n"
        f"- **Engagement matters:** active members churn at {active_rate*100:.1f}% versus "
        f"{inactive_rate*100:.1f}% for inactive members.\n"
        f"- **Revenue exposure:** churned customers collectively hold "
        f"${churned_balance/1e6:,.1f}M in balance ({balance_at_risk_pct*100:.1f}% of the "
        f"portfolio's ${total_balance/1e6:,.1f}M total), disproportionate to their "
        f"{overall_churn*100:.1f}% share of the customer count."
    )

    st.markdown("##### Recommended Priorities")
    st.markdown(
        "1. Audit the 3+ product segment for over-cross-selling or fee stacking.\n"
        f"2. Stand up a {top_geo}-specific retention review given its combined high "
        "churn and high average balances.\n"
        "3. Launch re-engagement programs targeting inactive members.\n"
        "4. Use the Risk Calculator and What-If Simulator tabs below to test individual "
        "customer scenarios and prioritize outreach."
    )

    st.caption(
        "Figures are computed live from European_Bank.csv. For the full narrative report "
        "with charts and detailed recommendations, see the companion Word document."
    )

# ==========================================================================
# TAB (Model Comparison) — Best Model Selection
# ==========================================================================
with tab_models:
    st.subheader("Model Comparison")
    st.write(
        "All five models are trained on the same preprocessed, engineered feature set "
        "and evaluated on the same held-out test set. The best model by ROC-AUC is used "
        "for every live prediction elsewhere in this dashboard (Risk Calculator, "
        "Probability Distribution, Feature Importance, What-If Simulator)."
    )

    metrics_df = artifacts["metrics_df"]
    best_name = artifacts["best_model_name"]
    best_row = metrics_df.iloc[0]

    bcol1, bcol2, bcol3 = st.columns(3)
    bcol1.metric("🏆 Best Model", best_name)
    bcol2.metric("ROC-AUC", f"{best_row['ROC-AUC']:.3f}")
    bcol3.metric("Recall", f"{best_row['Recall']:.3f}",
                 help="Share of actual churners this model correctly catches — often the "
                      "more business-relevant metric alongside ROC-AUC.")

    def highlight_best(row):
        color = "background-color: #FBF3DF; font-weight: 600;" if row["Model"] == best_name else ""
        return [color] * len(row)

    st.dataframe(
        metrics_df.style.apply(highlight_best, axis=1).format({
            "Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
            "F1-Score": "{:.3f}", "ROC-AUC": "{:.3f}",
        }),
        hide_index=True, use_container_width=True,
    )

    metrics_long = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig_models = px.bar(
        metrics_long, x="Model", y="Score", color="Metric", barmode="group",
        color_discrete_sequence=["#7B2CBF", "#00B4D8", "#06D6A0", "#FFD166", "#EF476F"],
    )
    fig_models.update_layout(
        height=440, template="plotly_white", plot_bgcolor="#FBFBFD",
        yaxis_title="Score", legend_title="Metric",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_models, use_container_width=True)

    st.caption(
        "Accuracy, Precision, Recall, and F1-Score use a 0.5 classification threshold; "
        "ROC-AUC is threshold-independent. Ranking is by ROC-AUC."
    )

# ==========================================================================
# TAB 1 — Churn Risk Calculator
# ==========================================================================
with tab1:
    st.subheader("Customer Churn Risk Calculator")
    st.write("Live prediction for the customer profile currently set in the sidebar.")

    c1, c2 = st.columns([1, 1.3])

    with c1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=current_proba * 100,
            number={"suffix": "%"},
            title={"text": f"Churn Probability — {band} Risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": band_color},
                "steps": [
                    {"range": [0, 25], "color": "#E8F3E8"},
                    {"range": [25, 50], "color": "#FBF3DF"},
                    {"range": [50, 75], "color": "#FCE7D6"},
                    {"range": [75, 100], "color": "#F7DEDA"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
            },
        ))
        fig_gauge.update_layout(height=340, margin=dict(t=60, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with c2:
        st.markdown("##### Profile Summary")
        summary = pd.DataFrame({
            "Attribute": ["Geography", "Gender", "Age", "Credit Score", "Tenure",
                          "Balance", "Products Held", "Active Member", "Has Credit Card",
                          "Estimated Salary"],
            "Value": [p["Geography"], p["Gender"], p["Age"], p["CreditScore"], f"{p['Tenure']} yrs",
                      f"${p['Balance']:,.0f}", p["NumOfProducts"],
                      "Yes" if p["IsActiveMember"] else "No",
                      "Yes" if p["HasCrCard"] else "No",
                      f"${p['EstimatedSalary']:,.0f}"],
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)

        if band == "Critical":
            st.error("⚠️ Critical risk — recommend immediate retention outreach.")
        elif band == "High":
            st.warning("⚠️ High risk — flag for proactive relationship management.")
        elif band == "Moderate":
            st.info("ℹ️ Moderate risk — monitor and consider engagement nudges.")
        else:
            st.success("✅ Low risk — no action needed at this time.")

# ==========================================================================
# TAB 2 — Probability Distribution Visualization
# ==========================================================================
with tab2:
    st.subheader("Churn Probability Distribution (Held-Out Test Set)")
    st.write(
        "Distribution of predicted churn probabilities across "
        f"{len(artifacts['test_proba']):,} held-out customers, split by actual outcome. "
        "The dashed line marks where the current sidebar profile falls."
    )

    dist_df = pd.DataFrame({
        "Predicted Probability": artifacts["test_proba"],
        "Actual Outcome": np.where(artifacts["y_test"].values == 1, "Churned", "Retained"),
    })

    fig_dist = px.histogram(
        dist_df, x="Predicted Probability", color="Actual Outcome",
        nbins=40, opacity=0.75, barmode="overlay",
        color_discrete_map={"Retained": "#00B4D8", "Churned": "#FF4D6D"},
        marginal="rug",
    )
    fig_dist.add_vline(x=current_proba, line_dash="dash", line_width=3, line_color="#FFB703",
                        annotation_text="Current customer", annotation_font_color="#FFB703",
                        annotation_position="top")
    fig_dist.update_layout(
        height=460, bargap=0.02, template="plotly_white",
        xaxis_title="Predicted Churn Probability", yaxis_title="Number of Customers",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#FBFBFD",
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    pct_rank = (artifacts["test_proba"] < current_proba).mean() * 100
    c1.metric("Percentile Rank", f"{pct_rank:.0f}th",
              help="Share of test-set customers with a LOWER predicted churn probability.")
    c2.metric("Test Set Churn Rate", f"{artifacts['y_test'].mean()*100:.1f}%")
    c3.metric("Model ROC-AUC", f"{artifacts['test_auc']:.3f}")

# ==========================================================================
# TAB 3 — Feature Importance Dashboard
# ==========================================================================
with tab3:
    st.subheader("Feature Importance Dashboard")
    st.write(f"Global feature importances from the best-performing model: **{artifacts['best_model_name']}**.")

    importance_series, importance_kind = get_feature_importance(
        artifacts["best_model"], artifacts["feature_columns"]
    )

    if importance_series is None:
        st.info(
            f"{artifacts['best_model_name']} doesn't expose a standard importance/coefficient "
            "attribute, so a ranking can't be shown for it here."
        )
    else:
        importances = importance_series.sort_values(ascending=True)

        top_n = st.slider("Number of features to show", 5, len(importances), 12)
        top_importances = importances.tail(top_n)

        fig_imp = px.bar(
            top_importances, x=top_importances.values, y=top_importances.index,
            orientation="h", labels={"x": "Importance", "y": "Feature"},
            color=top_importances.values,
            color_continuous_scale=[[0, "#00B4D8"], [0.5, "#7B2CBF"], [1, "#EF476F"]],
        )
        fig_imp.update_layout(height=max(360, top_n * 32), coloraxis_showscale=False,
                               margin=dict(l=10, r=10, t=20, b=10), template="plotly_white",
                               plot_bgcolor="#FBFBFD")
        st.plotly_chart(fig_imp, use_container_width=True)

        if importance_kind == "impurity":
            st.caption(
                f"Importances are computed via mean decrease in impurity across "
                f"{artifacts['best_model_name']}'s trees, and reflect global model behavior "
                "rather than any single customer."
            )
        else:
            st.caption(
                f"Bars show the absolute value of {artifacts['best_model_name']}'s standardized "
                "coefficients — larger magnitude means a stronger association with churn, "
                "independent of direction."
            )

# ==========================================================================
# TAB 4 — What-If Scenario Simulator
# ==========================================================================
with tab4:
    st.subheader("What-If Scenario Simulator")
    st.write(
        "Explore how churn probability responds to changes in engagement and product "
        "variables, holding the rest of the sidebar profile fixed."
    )

    sim_feature = st.selectbox(
        "Feature to vary",
        ["NumOfProducts", "IsActiveMember", "Age", "Tenure", "Balance",
         "CreditScore", "EstimatedSalary"],
        format_func=lambda x: {
            "NumOfProducts": "Number of Products", "IsActiveMember": "Active Member Status",
            "Age": "Age", "Tenure": "Tenure", "Balance": "Account Balance",
            "CreditScore": "Credit Score", "EstimatedSalary": "Estimated Salary",
        }[x],
    )

    ranges = {
        "NumOfProducts": np.arange(1, 5),
        "IsActiveMember": np.array([0, 1]),
        "Age": np.arange(18, 93, 2),
        "Tenure": np.arange(0, 11),
        "Balance": np.linspace(0, 250000, 40),
        "CreditScore": np.arange(350, 851, 20),
        "EstimatedSalary": np.linspace(0, 250000, 40),
    }
    sweep_values = ranges[sim_feature]

    sweep_probas = []
    for v in sweep_values:
        scenario = p.copy()
        scenario[sim_feature] = v
        sweep_probas.append(predict_proba(scenario, artifacts))

    sweep_df = pd.DataFrame({sim_feature: sweep_values, "Churn Probability": sweep_probas})

    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Scatter(
        x=sweep_df[sim_feature], y=sweep_df["Churn Probability"],
        mode="lines+markers",
        line=dict(color="#7B2CBF", width=3),
        marker=dict(
            size=11, color=sweep_df["Churn Probability"],
            colorscale=[[0, "#06D6A0"], [0.5, "#FFD166"], [1, "#EF476F"]],
            cmin=0, cmax=1, showscale=True,
            colorbar=dict(title="Risk", tickformat=".0%"),
            line=dict(width=1, color="white"),
        ),
        fill="tozeroy", fillcolor="rgba(123, 44, 191, 0.12)",
        hovertemplate=f"{sim_feature}: " + "%{x}<br>Churn Probability: %{y:.1%}<extra></extra>",
    ))
    fig_sweep.add_vline(x=p[sim_feature], line_dash="dash", line_width=3, line_color="#118AB2",
                         annotation_text="Current value", annotation_font_color="#118AB2",
                         annotation_position="top")
    fig_sweep.update_layout(
        height=440, template="plotly_white", plot_bgcolor="#FBFBFD",
        yaxis_tickformat=".0%", yaxis_title="Churn Probability", xaxis_title=sim_feature,
        showlegend=False,
    )
    st.plotly_chart(fig_sweep, use_container_width=True)

    st.markdown("##### Quick Comparison")
    colA, colB, colC = st.columns(3)
    baseline = current_proba

    best_idx = int(np.argmin(sweep_probas))
    worst_idx = int(np.argmax(sweep_probas))

    colA.metric("Current Scenario", f"{baseline*100:.1f}%")
    colB.metric(
        f"Best Case ({sim_feature}={sweep_values[best_idx]})",
        f"{sweep_probas[best_idx]*100:.1f}%",
        delta=f"{(sweep_probas[best_idx]-baseline)*100:+.1f} pp",
        delta_color="inverse",
    )
    colC.metric(
        f"Worst Case ({sim_feature}={sweep_values[worst_idx]})",
        f"{sweep_probas[worst_idx]*100:.1f}%",
        delta=f"{(sweep_probas[worst_idx]-baseline)*100:+.1f} pp",
        delta_color="inverse",
    )

    st.divider()
    st.markdown("##### Multi-Feature Adjustment")
    st.write("Nudge engagement and product variables directly to see the combined effect.")

    mcol1, mcol2 = st.columns(2)
    with mcol1:
        sim_products = st.slider("Simulated Number of Products", 1, 4, p["NumOfProducts"], key="sim_products")
        sim_active = st.checkbox("Simulated Active Member", value=bool(p["IsActiveMember"]), key="sim_active")
    with mcol2:
        sim_balance = st.number_input("Simulated Balance ($)", 0.0, 300000.0, p["Balance"], step=1000.0, key="sim_balance")
        sim_tenure = st.slider("Simulated Tenure", 0, 10, p["Tenure"], key="sim_tenure")

    sim_scenario = p.copy()
    sim_scenario.update({
        "NumOfProducts": sim_products,
        "IsActiveMember": 1 if sim_active else 0,
        "Balance": sim_balance,
        "Tenure": sim_tenure,
    })
    sim_proba = predict_proba(sim_scenario, artifacts)

    delta_pp = (sim_proba - baseline) * 100
    st.metric("Simulated Churn Probability", f"{sim_proba*100:.1f}%", delta=f"{delta_pp:+.1f} pp vs. current profile",
              delta_color="inverse")
