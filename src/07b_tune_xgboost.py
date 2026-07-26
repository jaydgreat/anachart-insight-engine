"""
Step 7b: Tuned XGBoost -- reduce the train/test AUC gap seen in the first
pass (0.808 train vs 0.708 test).

Changes from the first pass, and why each helps:
1. max_depth 4 -> 3          : shallower trees = less capacity to
                                memorize training-specific noise
2. reg_alpha / reg_lambda    : L1/L2 penalties on leaf weights, discourage
                                the model from fitting small quirks
3. min_child_weight raised   : forces each split to be supported by more
                                data, avoiding splits based on a handful
                                of rows
4. early_stopping_rounds     : instead of always training all 200 trees,
                                stop once performance on a VALIDATION slice
                                stops improving. This validation slice is
                                carved from the END of the training period
                                (still chronologically before the test
                                set) -- so it's leakage-safe, not touching
                                test data, and reflects "if I'd stopped
                                training at the right point."
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

# --- Same 80/20 chronological train/test split as before -----------------
cutoff_date = df["call_date"].quantile(0.8)
train_df = df[df["call_date"] <= cutoff_date]
test_df = df[df["call_date"] > cutoff_date]

# --- NEW: carve a validation slice from the LAST 15% of the training
# period, purely for early stopping (never touches test data) -----------
val_cutoff = train_df["call_date"].quantile(0.85)
fit_df = train_df[train_df["call_date"] <= val_cutoff]
val_df = train_df[train_df["call_date"] > val_cutoff]

print(f"Fit:   {len(fit_df):,} rows (up to {fit_df['call_date'].max().date()})")
print(f"Val:   {len(val_df):,} rows ({val_df['call_date'].min().date()} to {val_df['call_date'].max().date()})")
print(f"Test:  {len(test_df):,} rows ({test_df['call_date'].min().date()} to {test_df['call_date'].max().date()})")
print()

X_fit, y_fit = fit_df[FEATURE_COLS], fit_df[TARGET_COL]
X_val, y_val = val_df[FEATURE_COLS], val_df[TARGET_COL]
X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

model = XGBClassifier(
    n_estimators=200,          # ceiling -- early stopping will likely cut this short
    max_depth=3,                # was 4
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,         # was default (1) -- require more data per split
    reg_alpha=0.5,               # NEW: L1 regularization
    reg_lambda=2.0,              # NEW: L2 regularization
    eval_metric="auc",
    early_stopping_rounds=20,
    random_state=42,
)
model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

print(f"Best iteration (trees actually used): {model.best_iteration}")
print()

train_probs = model.predict_proba(X_fit)[:, 1]
test_probs = model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= 0.5).astype(int)

train_auc = roc_auc_score(y_fit, train_probs)
test_auc = roc_auc_score(y_test, test_probs)

print(f"Train AUC: {train_auc:.3f}")
print(f"Test AUC:  {test_auc:.3f}")
print(f"Gap:       {train_auc - test_auc:.3f}  (was 0.100 in the first pass)")
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

importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print("Feature importance:")
print(importance)

model.save_model("xgb_target_hit_model_tuned.json")
print("\nSaved tuned model to xgb_target_hit_model_tuned.json")
