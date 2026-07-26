"""
Step 7: Train the XGBoost classifier -- probability a price target gets hit.

CRITICAL: chronological split, not random k-fold
--------------------------------------------------
If we randomly shuffled rows into train/test, the model could train on a
call from 2020 and be tested on a call from 2015 -- meaning it "learned
the future" relative to some test rows. That's a subtle but serious form
of leakage in any time-series problem. Instead we sort by date and cut
train/test at a single point in time, exactly like a real deployment:
the model only ever sees calls that happened before the ones it's judged
on.

We use an 80/20 split BY TIME (not by row count coincidentally lining up
with 80%) -- e.g. if data spans 2005-2025, we might train on 2005-2021
and test on 2021-2025. The exact cutoff is whatever call ends up at the
80th percentile of dates.
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

df = pd.read_csv("model_ready_data.csv", parse_dates=["call_date"])
df = df.sort_values("call_date").reset_index(drop=True)

FEATURE_COLS = [
    "analyst_hit_rate", "analyst_n_prior_resolved",
    "broker_hit_rate", "broker_n_prior_resolved",
    "target_spread_clipped", "ticker_volatility_90d_clipped",
    "rating_action_code",
]
TARGET_COL = "hit"

# --- Chronological split --------------------------------------------------
cutoff_date = df["call_date"].quantile(0.8)  # the date at the 80th percentile
train_df = df[df["call_date"] <= cutoff_date]
test_df = df[df["call_date"] > cutoff_date]

print(f"Cutoff date: {cutoff_date.date()}")
print(f"Train: {len(train_df):,} rows ({train_df['call_date'].min().date()} to {train_df['call_date'].max().date()})")
print(f"Test:  {len(test_df):,} rows ({test_df['call_date'].min().date()} to {test_df['call_date'].max().date()})")
print()

X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

# --- Train ------------------------------------------------------------
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train)

# --- Evaluate -----------------------------------------------------------
train_probs = model.predict_proba(X_train)[:, 1]
test_probs = model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= 0.5).astype(int)

train_auc = roc_auc_score(y_train, train_probs)
test_auc = roc_auc_score(y_test, test_probs)

print(f"Train AUC: {train_auc:.3f}")
print(f"Test AUC:  {test_auc:.3f}")
if train_auc - test_auc > 0.1:
    print("  (Warning: large train/test AUC gap suggests overfitting.)")
print()

precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_preds, average="binary")
print(f"Test Precision: {precision:.3f}")
print(f"Test Recall:    {recall:.3f}")
print(f"Test F1:        {f1:.3f}")
print()

cm = confusion_matrix(y_test, test_preds)
print("Confusion matrix (rows=actual, cols=predicted, order=[Not Hit, Hit]):")
print(cm)
print()

# --- Feature importance --------------------------------------------------
importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print("Feature importance:")
print(importance)

# --- Save the trained model for later use by the API ----------------------
model.save_model("xgb_target_hit_model.json")
print("\nSaved model to xgb_target_hit_model.json")
