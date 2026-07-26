"""
Step 5: Add target spread and historical volatility features.

target_spread: how far the analyst's target is from the price AT THE TIME
of the call. We need the stock's closing price on (or just before) the
call date -- same "reference_price" concept from the labeling step, but
now as a proper feature.

ticker_volatility: how choppy the stock has historically been, computed
as the standard deviation of daily returns over the 90 trading days BEFORE
the call date (leakage-safe: we never look at price data from after the
call). Higher volatility generally means price targets get hit faster
(bigger swings) but also get "thesis broken" more easily.
"""
import pandas as pd
import numpy as np

VOL_WINDOW = 90  # trading days

calls = pd.read_csv("labeled_calls_features.csv", parse_dates=["call_date"])
prices = pd.read_csv("../data/nasdaq100_prices.csv", parse_dates=["date"])
prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

# Precompute daily returns once per ticker (not per call -- much faster)
prices["daily_return"] = prices.groupby("ticker")["close"].pct_change()

results = []
for ticker, ticker_calls in calls.groupby("ticker"):
    tp = prices[prices["ticker"] == ticker]
    dates = tp["date"].values
    closes = tp["close"].values
    returns = tp["daily_return"].values

    for idx, call in ticker_calls.iterrows():
        call_date = call["call_date"]

        # index of the last price row ON OR BEFORE the call date
        ref_idx = np.searchsorted(dates, call_date, side="right") - 1
        if ref_idx < 0:
            reference_price = np.nan
            volatility = np.nan
        else:
            reference_price = closes[ref_idx]
            # volatility window: the VOL_WINDOW returns ending at ref_idx
            # (inclusive), i.e. strictly BEFORE the call date -- leakage-safe
            start = max(0, ref_idx - VOL_WINDOW + 1)
            window_returns = returns[start:ref_idx + 1]
            volatility = np.nanstd(window_returns) if len(window_returns) > 1 else np.nan

        target_spread = (
            (call["price_target_post"] - reference_price) / reference_price
            if pd.notna(reference_price) and reference_price != 0
            else np.nan
        )

        results.append({
            "index": idx,
            "reference_price": reference_price,
            "target_spread": target_spread,
            "ticker_volatility_90d": volatility,
        })

feat_df = pd.DataFrame(results).set_index("index")
calls = calls.drop(columns=["reference_price"], errors="ignore")  # recomputing fresh below
calls = calls.join(feat_df)

calls.to_csv("labeled_calls_features_v2.csv", index=False)

print(f"Rows with target_spread computed: {calls['target_spread'].notna().sum():,} / {len(calls):,}")
print(f"Rows with volatility computed:    {calls['ticker_volatility_90d'].notna().sum():,} / {len(calls):,}")
print()
print("target_spread distribution:")
print(calls["target_spread"].describe())
print()
print("ticker_volatility_90d distribution:")
print(calls["ticker_volatility_90d"].describe())
print()
# Quick sanity: does bigger spread correlate with lower hit rate, as expected?
calls["spread_bucket"] = pd.qcut(calls["target_spread"], 4, duplicates="drop")
print(calls.groupby("spread_bucket", observed=True)["hit"].mean())
