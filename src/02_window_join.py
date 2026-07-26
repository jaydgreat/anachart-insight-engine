"""
Step 2: For each analyst call, find the max high and min low price the
stock reached in the 365 days following the call date.

Key SQL concepts used here:
- DATE(c.date, '+365 days')  -> SQLite's way of adding a time interval
  to a date. This computes each call's window end.
- JOIN ... ON <range condition>  -> instead of joining on equality
  (like ticker = ticker), we ALSO require the price date to fall inside
  the call's window. SQL joins don't have to be on exact matches.
- GROUP BY  -> collapses many matching price rows (one call can match
  up to 365 price rows) down into a single row per call, using MAX/MIN
  as the aggregation.
"""
import sqlite3
import pandas as pd

conn = sqlite3.connect("../data/anachart.db" if False else "anachart.db")

query = """
SELECT
    c.date            AS call_date,
    c.ticker,
    c.analyst_name,
    c.rating_post_clean,
    c.price_target_post,
    DATE(c.date, '+365 days') AS window_end,
    MAX(p.high)       AS window_max_high,
    MIN(p.low)        AS window_min_low,
    COUNT(p.date)     AS days_of_price_data
FROM analyst_calls c
JOIN prices p
    ON p.ticker = c.ticker
    AND p.date > c.date
    AND p.date <= DATE(c.date, '+365 days')
WHERE c.price_target_post IS NOT NULL
    AND c.rating_post_clean IN ('BULLISH', 'BEARISH')
GROUP BY c.date, c.ticker, c.analyst_name, c.rating_post_clean, c.price_target_post
"""

result = pd.read_sql(query, conn)
print(f"Rows produced: {len(result):,}")
print()
print(result.head(10))
print()
print("days_of_price_data distribution (should mostly be ~250-260, since")
print("that's roughly how many trading days occur in a calendar year):")
print(result["days_of_price_data"].describe())

conn.close()
