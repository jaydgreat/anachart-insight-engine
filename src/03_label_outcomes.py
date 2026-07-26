"""
Step 3: Assign the final outcome label to each analyst call.

New SQL concept: CORRELATED SUBQUERY -- a subquery that references the
outer query's current row (e.g. "c.ticker", "c.date"). SQLite runs it
once per outer row, using the (ticker, date) index to keep it fast.

We compute three things per call:
  1. reference_price -> the stock's close price on (or just before) the
     call date, used as the baseline for the thesis-broken 20% threshold
  2. hit_date        -> first date the target was actually reached
  3. broken_date     -> first date price moved >20% against the thesis,
                        BEFORE any hit occurred

Then in Python, we combine these with last_price_date (does this ticker
even have price data past the window yet?) to assign one of the four
final labels, using clear priority: Achieved > Broken > Active/Expired.
"""
import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect("anachart.db")

BROKEN_THRESHOLD = 0.20  # 20% adverse move = thesis broken

query = """
SELECT
    c.date              AS call_date,
    c.ticker,
    c.analyst_name,
    c.broker,
    c.rating_post_clean,
    c.price_target_post,
    DATE(c.date, '+365 days') AS window_end,

    -- reference price: last known close ON OR BEFORE the call date
    (SELECT p2.close FROM prices p2
     WHERE p2.ticker = c.ticker AND p2.date <= c.date
     ORDER BY p2.date DESC LIMIT 1)               AS reference_price,

    -- first date the target was actually hit
    (SELECT MIN(p3.date) FROM prices p3
     WHERE p3.ticker = c.ticker
       AND p3.date > c.date
       AND p3.date <= DATE(c.date, '+365 days')
       AND ((c.rating_post_clean = 'BULLISH' AND p3.high >= c.price_target_post)
         OR (c.rating_post_clean = 'BEARISH' AND p3.low  <= c.price_target_post))
    )                                              AS hit_date,

    -- last price date available for this ticker at all (tells us if the
    -- window has even had a chance to finish yet)
    (SELECT MAX(p4.date) FROM prices p4
     WHERE p4.ticker = c.ticker)                   AS last_price_date

FROM analyst_calls c
WHERE c.price_target_post IS NOT NULL
  AND c.rating_post_clean IN ('BULLISH', 'BEARISH')
"""

df = pd.read_sql(query, conn)

# broken_date needs the reference_price we just computed, so it's a second
# pass (cleaner in pandas than nesting it inside the SQL above)
def find_broken_date(row):
    if pd.isna(row["reference_price"]):
        return None
    if row["rating_post_clean"] == "BULLISH":
        threshold_price = row["reference_price"] * (1 - BROKEN_THRESHOLD)
        cond = f"p.close <= {threshold_price}"
    else:  # BEARISH
        threshold_price = row["reference_price"] * (1 + BROKEN_THRESHOLD)
        cond = f"p.close >= {threshold_price}"

    q = f"""
        SELECT MIN(p.date) FROM prices p
        WHERE p.ticker = ?
          AND p.date > ?
          AND p.date <= ?
          AND {cond}
    """
    result = conn.execute(q, (row["ticker"], row["call_date"], row["window_end"])).fetchone()[0]
    return result

print("Computing thesis-broken dates (this loops per-row, may take a minute)...")
df["broken_date"] = df.apply(find_broken_date, axis=1)

# --- Assign final labels ---------------------------------------------
TODAY = pd.Timestamp.today().strftime("%Y-%m-%d")

def assign_label(row):
    hit, broken, window_end, last_price = (
        row["hit_date"], row["broken_date"], row["window_end"], row["last_price_date"]
    )
    # Achieved wins if it happened, and happened before any break
    if pd.notna(hit) and (pd.isna(broken) or hit <= broken):
        return "TARGET_ACHIEVED"
    # Broken wins if it happened before any hit
    if pd.notna(broken) and (pd.isna(hit) or broken < hit):
        return "THESIS_BROKEN"
    # Neither hit nor broken -- window still open, or fully expired?
    if window_end > last_price:
        return "ACTIVE"
    return "TIME_EXPIRED"

df["outcome"] = df.apply(assign_label, axis=1)

# days_to_hit only meaningful for Achieved calls
df["days_to_hit"] = np.where(
    df["outcome"] == "TARGET_ACHIEVED",
    (pd.to_datetime(df["hit_date"]) - pd.to_datetime(df["call_date"])).dt.days,
    np.nan,
)

df.to_csv("labeled_calls.csv", index=False)

print()
print(f"Total labeled calls: {len(df):,}")
print()
print("Outcome distribution:")
print(df["outcome"].value_counts())
print()
print("Median days_to_hit for Achieved calls:", df.loc[df["outcome"]=="TARGET_ACHIEVED", "days_to_hit"].median())

conn.close()
