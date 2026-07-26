"""
Step 3, SQL-free version: same outcome labeling, done with pandas + numpy
instead of SQLite.

Key technique: np.searchsorted
-------------------------------
For each ticker, we sort its prices by date once. Then, for any call date,
np.searchsorted instantly tells us the array INDEX where that date would
be inserted to keep things sorted -- which is exactly the index of the
first price row on/after that date. This is a binary search: O(log n)
instead of scanning every row. We do this once per ticker (not once per
call), which is why grouping by ticker first matters.

This replaces:
- SQL's indexed range JOIN          -> searchsorted for window boundaries
- SQL's correlated subquery (hit)   -> np.argmax on a boolean mask slice
- SQL's GROUP BY MAX/MIN            -> numpy .max()/.min() on the slice
"""
import pandas as pd
import numpy as np

BROKEN_THRESHOLD = 0.20
WINDOW_DAYS = 365

analyst = pd.read_csv("../data/analyst_data_final.csv", parse_dates=["date"])
prices = pd.read_csv("../data/nasdaq100_prices.csv", parse_dates=["date"])

# Only calls we can actually price-label: directional rating + real target
calls = analyst[
    analyst["rating_post_clean"].isin(["BULLISH", "BEARISH"])
    & analyst["price_target_post"].notna()
].copy()

results = []

# --- Loop over TICKERS (93), not over CALLS (28k) -----------------------
for ticker, ticker_calls in calls.groupby("ticker"):
    ticker_prices = prices[prices["ticker"] == ticker].sort_values("date")
    if ticker_prices.empty:
        continue

    # Convert to numpy arrays once per ticker -- this is what makes it fast
    dates = ticker_prices["date"].values          # sorted datetime64 array
    highs = ticker_prices["high"].values
    lows = ticker_prices["low"].values
    closes = ticker_prices["close"].values
    last_price_date = dates[-1]

    for _, call in ticker_calls.iterrows():
        call_date = call["date"]
        window_end = call_date + pd.Timedelta(days=WINDOW_DAYS)
        target = call["price_target_post"]
        is_bullish = call["rating_post_clean"] == "BULLISH"

        # searchsorted: instantly find index range for "date > call_date
        # AND date <= window_end" -- no scanning, just binary search
        start_idx = np.searchsorted(dates, call_date, side="right")
        end_idx = np.searchsorted(dates, window_end, side="right")

        if start_idx >= end_idx:
            continue  # no price data in window at all

        window_highs = highs[start_idx:end_idx]
        window_lows = lows[start_idx:end_idx]
        window_closes = closes[start_idx:end_idx]
        window_dates = dates[start_idx:end_idx]

        # reference price: last close on/before call_date
        ref_idx = np.searchsorted(dates, call_date, side="right") - 1
        reference_price = closes[ref_idx] if ref_idx >= 0 else np.nan

        # hit detection: boolean mask, then argmax finds the FIRST True
        # (argmax on an all-False array returns 0, so we must check .any())
        if is_bullish:
            hit_mask = window_highs >= target
        else:
            hit_mask = window_lows <= target
        hit_date = window_dates[np.argmax(hit_mask)] if hit_mask.any() else pd.NaT

        # thesis-broken detection: same trick, on the close-price threshold
        broken_date = pd.NaT
        if pd.notna(reference_price):
            if is_bullish:
                broken_mask = window_closes <= reference_price * (1 - BROKEN_THRESHOLD)
            else:
                broken_mask = window_closes >= reference_price * (1 + BROKEN_THRESHOLD)
            if broken_mask.any():
                broken_date = window_dates[np.argmax(broken_mask)]

        # --- assign label: same priority logic as the SQL version -------
        if pd.notna(hit_date) and (pd.isna(broken_date) or hit_date <= broken_date):
            outcome = "TARGET_ACHIEVED"
        elif pd.notna(broken_date) and (pd.isna(hit_date) or broken_date < hit_date):
            outcome = "THESIS_BROKEN"
        elif window_end > last_price_date:
            outcome = "ACTIVE"
        else:
            outcome = "TIME_EXPIRED"

        days_to_hit = (pd.Timestamp(hit_date) - call_date).days if outcome == "TARGET_ACHIEVED" else np.nan

        results.append({
            "call_date": call_date, "ticker": ticker,
            "analyst_name": call["analyst_name"], "broker": call["broker"],
            "rating_post_clean": call["rating_post_clean"],
            "price_target_post": target, "outcome": outcome,
            "days_to_hit": days_to_hit,
        })

df = pd.DataFrame(results)
df.to_csv("labeled_calls_numpy.csv", index=False)

print(f"Total labeled calls: {len(df):,}")
print()
print("Outcome distribution:")
print(df["outcome"].value_counts())
print()
print("Median days_to_hit for Achieved calls:", df.loc[df["outcome"]=="TARGET_ACHIEVED", "days_to_hit"].median())
