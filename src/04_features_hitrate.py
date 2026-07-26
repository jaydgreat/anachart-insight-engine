"""
Step 4: Build ML-ready features from labeled_calls.csv.

THE KEY CONCEPT: leakage-safe historical hit rate
--------------------------------------------------
"Analyst X's hit rate" must only reflect calls that happened BEFORE the
call we're computing it for. If we used their whole career average
(including future calls), the model would be implicitly told the answer.

The trick: sort by date, then for each row use the CUMULATIVE hit rate
of all PRIOR rows only. In pandas this is:
    expanding().mean() gives the running average INCLUDING the current row
    .shift(1) BEFORE expanding gives the running average EXCLUDING it

So: df.groupby('analyst_name')['hit'].apply(lambda s: s.shift(1).expanding().mean())

For an analyst's very FIRST call, there's no prior history at all -- that
produces NaN, which is correct and expected (we genuinely don't know their
hit rate yet). We'll fill these with the GLOBAL average hit rate as a
reasonable default, and add a flag column so the model can learn that
"unknown track record" is itself a meaningful signal.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("labeled_calls.csv", parse_dates=["call_date"])

# Only resolved outcomes have a meaningful "did it hit" signal for hit-rate
# calculations. ACTIVE calls haven't resolved yet, so hit rate calculations
# should be based on resolved calls only (Achieved / Broken / Expired).
df = df.sort_values("call_date").reset_index(drop=True)
df["hit"] = (df["outcome"] == "TARGET_ACHIEVED").astype(float)
df.loc[df["outcome"] == "ACTIVE", "hit"] = np.nan  # unresolved, exclude from history calcs

GLOBAL_HIT_RATE = df["hit"].mean()
print(f"Global historical hit rate (baseline): {GLOBAL_HIT_RATE:.1%}")

# Shrinkage constant: how many "fake" baseline observations we blend in.
# k=10 means we don't fully trust an analyst's own record until they have
# roughly 10+ resolved calls; below that, we pull toward the global average.
K = 10

def leakage_safe_shrunk_rate(hit_series, k, global_rate):
    """
    For each row, compute the shrinkage-adjusted hit rate using ONLY prior
    rows (shift(1) before any cumulative calc -- this is what prevents
    leakage). Unresolved (NaN) calls are automatically skipped by
    expanding().sum()/.count(), which is exactly what we want.

    Formula: (sum_of_prior_hits + k * global_rate) / (count_of_prior_resolved + k)
    - With 0 prior calls: reduces to exactly global_rate (no history = baseline)
    - With many prior calls: k's influence shrinks toward the analyst's own rate
    """
    shifted = hit_series.shift(1)
    prior_sum = shifted.expanding().sum().fillna(0)   # NaN-only window -> 0, not NaN
    prior_count = shifted.expanding().count()          # counts only resolved priors
    return (prior_sum + k * global_rate) / (prior_count + k), prior_count

# --- ANALYST hit rate, leakage-safe + shrinkage --------------------------
analyst_rate, analyst_n_resolved = zip(*[
    leakage_safe_shrunk_rate(g["hit"], K, GLOBAL_HIT_RATE)
    for _, g in df.groupby("analyst_name")
])
df["analyst_hit_rate"] = pd.concat(analyst_rate).sort_index()
df["analyst_n_prior_resolved"] = pd.concat(analyst_n_resolved).sort_index().astype(int)
df["analyst_n_prior_calls"] = df.groupby("analyst_name").cumcount()

# --- BROKER hit rate, same technique -------------------------------------
broker_rate, broker_n_resolved = zip(*[
    leakage_safe_shrunk_rate(g["hit"], K, GLOBAL_HIT_RATE)
    for _, g in df.groupby("broker")
])
df["broker_hit_rate"] = pd.concat(broker_rate).sort_index()
df["broker_n_prior_resolved"] = pd.concat(broker_n_resolved).sort_index().astype(int)
df["broker_n_prior_calls"] = df.groupby("broker").cumcount()

# --- Sanity check: verify no leakage by construction ---------------------
# For an analyst's FIRST call, there's no prior history, so the shrinkage
# formula should reduce to EXACTLY the global rate.
first_calls = df[df["analyst_n_prior_calls"] == 0]
assert np.allclose(first_calls["analyst_hit_rate"], GLOBAL_HIT_RATE), "Leakage check failed!"
print(f"Leakage check passed: all {len(first_calls):,} analyst-first-calls "
      f"correctly reduce to the global baseline ({GLOBAL_HIT_RATE:.1%}).")

df.to_csv("labeled_calls_features.csv", index=False)

print()
print(df[["call_date", "analyst_name", "outcome", "analyst_n_prior_calls",
          "analyst_n_prior_resolved", "analyst_hit_rate", "broker",
          "broker_n_prior_resolved", "broker_hit_rate"]].head(10).to_string())
