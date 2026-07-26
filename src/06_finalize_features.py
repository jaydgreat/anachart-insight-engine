"""
Step 6: Clip outliers and assemble the final ML-ready dataset.

Why clip instead of drop: a handful of rows have extreme target_spread
values (likely from tiny reference prices due to data glitches or
pre-split pricing quirks). Dropping them loses real calls; clipping
(winsorizing) caps the extreme VALUE without removing the ROW, so the
model still learns from these calls using a sane, bounded spread.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("labeled_calls_features_v2.csv", parse_dates=["call_date"])

# rating_action got dropped earlier in the pipeline (the labeling scripts
# only carried forward a subset of columns) -- merge it back in from the
# original cleaned analyst data using the natural identifying key.
original = pd.read_csv("../data/analyst_data_final.csv", parse_dates=["date"])
original = original.rename(columns={"date": "call_date"})
original = original.drop_duplicates(
    subset=["call_date", "ticker", "analyst_name", "broker", "price_target_post"]
)
df = df.merge(
    original[["call_date", "ticker", "analyst_name", "broker", "price_target_post", "rating_action"]],
    on=["call_date", "ticker", "analyst_name", "broker", "price_target_post"],
    how="left",
)

before = df["target_spread"].describe()

# Clip to a sane range: -90% (stock nearly worthless) to +200% (a 3x target,
# already an extreme call). Anything beyond this is almost certainly a
# data artifact, not a real analyst thesis.
df["target_spread_clipped"] = df["target_spread"].clip(lower=-0.9, upper=2.0)

# Same idea for volatility -- cap at a generous but sane ceiling
df["ticker_volatility_90d_clipped"] = df["ticker_volatility_90d"].clip(upper=0.08)

n_clipped_spread = (df["target_spread"] != df["target_spread_clipped"]).sum()
n_clipped_vol = (df["ticker_volatility_90d"] != df["ticker_volatility_90d_clipped"]).sum()
print(f"target_spread: {n_clipped_spread} rows clipped (out of {len(df):,})")
print(f"volatility:    {n_clipped_vol} rows clipped (out of {len(df):,})")

# Encode rating_action as a simple numeric feature (models need numbers,
# not strings). REITERATE=0 is the baseline; UPGRADE/DOWNGRADE are momentum
# signals. Missing (no prior rating = initiation) gets its own category.
df["rating_action_filled"] = df["rating_action"].fillna("INITIATION")
action_map = {"INITIATION": 0, "REITERATE": 1, "UPGRADE": 2, "DOWNGRADE": 3}
df["rating_action_code"] = df["rating_action_filled"].map(action_map)

# Final feature set for the model
FEATURE_COLS = [
    "analyst_hit_rate", "analyst_n_prior_resolved",
    "broker_hit_rate", "broker_n_prior_resolved",
    "target_spread_clipped", "ticker_volatility_90d_clipped",
    "rating_action_code",
]
TARGET_COL = "hit"  # 1 = Target Achieved, 0 = Broken/Expired (Active already excluded upstream)

model_df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
model_df = model_df.sort_values("call_date").reset_index(drop=True)

model_df.to_csv("model_ready_data.csv", index=False)

print()
print(f"Final model-ready rows: {len(model_df):,} (dropped {len(df)-len(model_df):,} with missing features/target)")
print(f"Target distribution:\n{model_df[TARGET_COL].value_counts(normalize=True)}")
print()
print("Feature columns:", FEATURE_COLS)
