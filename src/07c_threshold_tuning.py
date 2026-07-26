"""
Step 7c: Threshold tuning.

The model outputs a PROBABILITY (0 to 1), not a yes/no answer. We chose
0.5 as the cutoff for "predict Hit" somewhat arbitrarily. Different
thresholds trade precision against recall:
- Higher threshold (e.g. 0.65): fewer "Hit" predictions, but more of them
  are correct (higher precision, lower recall) -- good for "only show me
  high-confidence calls"
- Lower threshold (e.g. 0.35): more "Hit" predictions, catching more of
  the true hits, but more false alarms (lower precision, higher recall)
  -- good for "don't want to miss any real opportunities"

We reload the exact same test set (same chronological split) and the
already-trained tuned model, then sweep thresholds to show the tradeoff.
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

df = pd.read_csv("model_ready_data.csv", parse_dates=["call_date"])
df = df.sort_values("call_date").reset_index(drop=True)

FEATURE_COLS = [
    "analyst_hit_rate", "analyst_n_prior_resolved",
    "broker_hit_rate", "broker_n_prior_resolved",
    "target_spread_clipped", "ticker_volatility_90d_clipped",
    "rating_action_code",
]
TARGET_COL = "hit"

# Same split as before -- must match exactly so we're evaluating on the
# identical test set the model was validated against
cutoff_date = df["call_date"].quantile(0.8)
test_df = df[df["call_date"] > cutoff_date]
X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

model = XGBClassifier()
model.load_model("xgb_target_hit_model_tuned.json")
test_probs = model.predict_proba(X_test)[:, 1]

print(f"Test AUC (threshold-independent): {roc_auc_score(y_test, test_probs):.3f}")
print()
print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'# Predicted Hit':>16}")
print("-" * 58)

results = []
for threshold in np.arange(0.30, 0.71, 0.05):
    preds = (test_probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0
    )
    n_predicted_hit = preds.sum()
    results.append((threshold, precision, recall, f1, n_predicted_hit))
    print(f"{threshold:>10.2f} {precision:>10.3f} {recall:>10.3f} {f1:>8.3f} {n_predicted_hit:>16,}")

results_df = pd.DataFrame(results, columns=["threshold", "precision", "recall", "f1", "n_predicted_hit"])
best_f1_row = results_df.loc[results_df["f1"].idxmax()]
print()
print(f"Best F1 at threshold {best_f1_row['threshold']:.2f}: "
      f"precision={best_f1_row['precision']:.3f}, recall={best_f1_row['recall']:.3f}, f1={best_f1_row['f1']:.3f}")

results_df.to_csv("threshold_sweep_results.csv", index=False)
