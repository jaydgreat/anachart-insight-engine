"""
Step 1: Load the two cleaned CSVs into a SQLite database.

Why SQLite: built into Python's standard library (no install needed),
supports real SQL including window functions and CTEs, and gives us a
proper queryable database instead of juggling two separate CSVs.
"""
import sqlite3
import pandas as pd

DB_PATH = "anachart.db"

analyst_df = pd.read_csv("../data/analyst_data_final.csv", parse_dates=["date"])
prices_df = pd.read_csv("../data/nasdaq100_prices.csv", parse_dates=["date"])

conn = sqlite3.connect(DB_PATH)

# to_sql writes the DataFrame straight into a SQL table -- this is the
# bridge between "pandas world" and "SQL world"
analyst_df.to_sql("analyst_calls", conn, if_exists="replace", index=False)
prices_df.to_sql("prices", conn, if_exists="replace", index=False)

# Indexes matter a LOT here: without them, every query has to scan the
# whole prices table (518k rows) for every one of 47k calls. An index on
# (ticker, date) lets SQLite jump straight to the relevant rows.
conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON prices(ticker, date)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_calls_ticker_date ON analyst_calls(ticker, date)")
conn.commit()

# Sanity check
n_calls = conn.execute("SELECT COUNT(*) FROM analyst_calls").fetchone()[0]
n_prices = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
print(f"analyst_calls table: {n_calls:,} rows")
print(f"prices table:        {n_prices:,} rows")

conn.close()
print(f"\nSaved to {DB_PATH}")
