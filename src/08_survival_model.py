"""
Step 8: Time-to-hit model, using a DISCRETE-TIME HAZARD MODEL.

Why not just regress "days_to_hit" directly with the Achieved calls only?
Because that silently throws away every Broken/Expired call -- the model
would never learn "targets with a 300% spread basically never hit
quickly," since those failure cases are excluded entirely. We need a
technique that uses ALL calls, including the ones that never hit.

THE TECHNIQUE: discrete-time hazard model (a standard, well-established
survival analysis approach -- same math family used in medical
"time to recurrence" studies)
--------------------------------------------------------------------------
1. Chop the 365-day window into 30-day bins (~12 bins).
2. Turn each CALL into multiple PERSON-PERIOD rows: one row for every bin
   it "survived through" without hitting, plus a final row marking
   whether it hit (event=1) or the window simply ended (event=0,
   "censored") in that final bin.
3. Train a classifier on this expanded dataset to predict:
   "probability the hit happens in THIS bin, given it hasn't happened yet"
4. Chain these period-by-period probabilities together (multiply the
   survival probabilities) to get a full curve per call: P(still not hit
   by day 30), P(still not hit by day 60), etc. -- and from that curve,
   read off things like "50% chance it's already hit by day X."
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

BIN_DAYS = 30
WINDOW_DAYS = 365
N_BINS = int(np.ceil(WINDOW_DAYS / BIN_DAYS))  # 13 bins (last one slightly short)

FEATURE_COLS = [
    "analyst_hit_rate", "analyst_n_prior_resolved",
    "broker_hit_rate", "broker_n_prior_resolved",
    "target_spread_clipped", "ticker_volatility_90d_clipped",
    "rating_action_code",
]

df = pd.read_csv("model_ready_data.csv", parse_dates=["call_date"])
df = df.sort_values("call_date").reset_index(drop=True)

# event=1 if it hit, time = the day it hit OR the full window length if it
# never did (Broken and Expired both mean "no hit observed within 365 days")
df["event"] = (df["outcome"] == "TARGET_ACHIEVED").astype(int)
df["time"] = np.where(df["event"] == 1, df["days_to_hit"], WINDOW_DAYS)
df["bin_of_event"] = np.clip((df["time"] // BIN_DAYS).astype(int), 0, N_BINS - 1)

# --- Build the person-period (long-format) dataset ------------------------
print("Expanding to person-period format (one row per call per bin survived)...")
long_rows = []
for idx, row in df.iterrows():
    last_bin = row["bin_of_event"]
    for b in range(last_bin + 1):
        long_rows.append({
            "call_idx": idx,
            "call_date": row["call_date"],
            "bin": b,
            "event_in_bin": 1 if (b == last_bin and row["event"] == 1) else 0,
            **{col: row[col] for col in FEATURE_COLS},
        })
long_df = pd.DataFrame(long_rows)
print(f"Original calls: {len(df):,}  ->  Person-period rows: {len(long_df):,}")
print()

# --- Chronological split, by the ORIGINAL call's date (not the bin) -----
# so no future call's data leaks into training via the expansion
cutoff_date = df["call_date"].quantile(0.8)
train_long = long_df[long_df["call_date"] <= cutoff_date]
test_long = long_df[long_df["call_date"] > cutoff_date]

HAZARD_FEATURES = FEATURE_COLS + ["bin"]
X_train, y_train = train_long[HAZARD_FEATURES], train_long["event_in_bin"]
X_test, y_test = test_long[HAZARD_FEATURES], test_long["event_in_bin"]

hazard_model = XGBClassifier(
    n_estimators=150, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    reg_alpha=0.5, reg_lambda=2.0, eval_metric="logloss", random_state=42,
)
hazard_model.fit(X_train, y_train)

test_hazard_preds = hazard_model.predict_proba(X_test)[:, 1]
print(f"Person-period AUC (predicting 'hits in this bin'): "
      f"{roc_auc_score(y_test, test_hazard_preds):.3f}")
print()

# --- Reconstruct per-call survival curves on the TEST set -----------------
test_calls = df[df["call_date"] > cutoff_date].copy()

def predict_survival_curve(row):
    """Chain period hazards into a full survival curve for one call."""
    bins = np.arange(N_BINS)
    feat_rows = pd.DataFrame(
        [{**{c: row[c] for c in FEATURE_COLS}, "bin": b} for b in bins]
    )
    hazards = hazard_model.predict_proba(feat_rows[HAZARD_FEATURES])[:, 1]
    survival = np.cumprod(1 - hazards)  # P(still hasn't hit by end of each bin)
    return survival

print("Computing survival curves for test-set calls (this loops per row)...")
survival_curves = test_calls.apply(predict_survival_curve, axis=1)

def median_hit_day(survival):
    """First bin where survival probability drops to/below 50% -> our
    'expected days to hit' estimate. If it never drops below 50%, we
    report it as 'unlikely to hit within the window'."""
    below_half = np.where(survival <= 0.5)[0]
    if len(below_half) == 0:
        return np.nan  # model thinks <50% chance of ever hitting in-window
    return (below_half[0] + 1) * BIN_DAYS

test_calls["predicted_median_hit_day"] = survival_curves.apply(median_hit_day)
test_calls["predicted_prob_hit_in_window"] = survival_curves.apply(lambda s: 1 - s[-1])

# --- Sanity check against ACTUAL outcomes on Achieved test calls ---------
achieved_test = test_calls[test_calls["outcome"] == "TARGET_ACHIEVED"].dropna(
    subset=["predicted_median_hit_day"]
)
mae = (achieved_test["predicted_median_hit_day"] - achieved_test["days_to_hit"]).abs().mean()
print(f"\nOn {len(achieved_test):,} test calls that DID hit, with a predicted "
      f"median day available:")
print(f"  Mean Absolute Error (predicted vs actual days-to-hit): {mae:.1f} days")
print()
print("Predicted probability-of-hit distribution, split by what actually happened:")
print(test_calls.groupby("outcome")["predicted_prob_hit_in_window"].mean())

test_calls.to_csv("survival_predictions_test.csv", index=False)
hazard_model.save_model("hazard_model.json")
print("\nSaved hazard_model.json and survival_predictions_test.csv")
