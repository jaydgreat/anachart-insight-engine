"""
Step 9: FastAPI inference microservice.

Wraps the two trained models (XGBoost classifier + discrete-time hazard
model) behind a single /predict endpoint. Given a ticker, analyst, and
target price, it looks up that analyst's/broker's current historical hit
rate and the stock's current price/volatility from the data we've already
built, computes the same features the models were trained on, and returns
both the probability of hitting and an estimated time-to-hit.

Run with:
    uvicorn api:app --reload

Then test at http://127.0.0.1:8000/docs (FastAPI auto-generates an
interactive Swagger UI there -- no separate tool needed to try it out).
"""
from typing import Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xgboost import XGBClassifier

BIN_DAYS = 30
WINDOW_DAYS = 365
N_BINS = int(np.ceil(WINDOW_DAYS / BIN_DAYS))

FEATURE_COLS = [
    "analyst_hit_rate", "analyst_n_prior_resolved",
    "broker_hit_rate", "broker_n_prior_resolved",
    "target_spread_clipped", "ticker_volatility_90d_clipped",
    "rating_action_code",
]
ACTION_MAP = {"INITIATION": 0, "REITERATE": 1, "UPGRADE": 2, "DOWNGRADE": 3}

app = FastAPI(title="AnaChart Insight Engine")

# --- Loaded once at startup, not per-request ------------------------------
classifier = XGBClassifier()
classifier.load_model("xgb_target_hit_model_tuned.json")

hazard_model = XGBClassifier()
hazard_model.load_model("hazard_model.json")

model_data = pd.read_csv("model_ready_data.csv", parse_dates=["call_date"])
GLOBAL_HIT_RATE = model_data["hit"].mean()

prices = pd.read_csv("../data/nasdaq100_prices.csv", parse_dates=["date"])
prices = prices.sort_values(["ticker", "date"])

# Precompute "latest known stats per analyst/broker" once at startup --
# this is a simplification: in a live system you'd recompute these as new
# calls come in, but for this prototype we use each analyst's/broker's
# most recent historical value as a stand-in for "their rate right now."
_analyst_latest = (
    model_data.sort_values("call_date")
    .groupby("analyst_name")
    .last()[["analyst_hit_rate", "analyst_n_prior_resolved"]]
)
_broker_latest = (
    model_data.sort_values("call_date")
    .groupby("broker")
    .last()[["broker_hit_rate", "broker_n_prior_resolved"]]
)


class PredictRequest(BaseModel):
    ticker: str
    analyst_name: str
    price_target: float
    rating: str  # "BULLISH" or "BEARISH"
    broker: Optional[str] = "UNKNOWN"
    rating_action: Optional[str] = "REITERATE"  # UPGRADE / DOWNGRADE / REITERATE / INITIATION


class PredictResponse(BaseModel):
    ticker: str
    analyst_name: str
    current_price: float
    price_target: float
    target_spread_pct: float
    probability_hit: float
    predicted_median_days: Optional[float]
    analyst_hit_rate: float
    analyst_n_prior_resolved: int
    broker_hit_rate: float
    broker_n_prior_resolved: int


def get_current_price_and_volatility(ticker: str):
    tp = prices[prices["ticker"] == ticker]
    if tp.empty:
        raise HTTPException(status_code=404, detail=f"No price data found for ticker '{ticker}'")
    current_price = tp["close"].iloc[-1]
    returns = tp["close"].pct_change().tail(90)
    volatility = returns.std()
    return current_price, volatility


def get_analyst_stats(analyst_name: str):
    key = analyst_name.upper().strip()
    if key in _analyst_latest.index:
        row = _analyst_latest.loc[key]
        return float(row["analyst_hit_rate"]), int(row["analyst_n_prior_resolved"]) + 1
    # Unknown analyst -- no track record on file, fall back to baseline
    return float(GLOBAL_HIT_RATE), 0


def get_broker_stats(broker: str):
    key = (broker or "UNKNOWN").upper().strip()
    if key in _broker_latest.index:
        row = _broker_latest.loc[key]
        return float(row["broker_hit_rate"]), int(row["broker_n_prior_resolved"]) + 1
    return float(GLOBAL_HIT_RATE), 0


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    rating = req.rating.upper().strip()
    if rating not in ("BULLISH", "BEARISH"):
        raise HTTPException(status_code=400, detail="rating must be BULLISH or BEARISH")

    current_price, volatility = get_current_price_and_volatility(req.ticker)
    if current_price == 0 or pd.isna(current_price):
        raise HTTPException(status_code=400, detail="Invalid current price for this ticker")

    target_spread = (req.price_target - current_price) / current_price
    target_spread_clipped = float(np.clip(target_spread, -0.9, 2.0))
    volatility_clipped = float(np.clip(volatility, 0, 0.08)) if pd.notna(volatility) else 0.02

    analyst_hit_rate, analyst_n = get_analyst_stats(req.analyst_name)
    broker_hit_rate, broker_n = get_broker_stats(req.broker)

    action_code = ACTION_MAP.get((req.rating_action or "REITERATE").upper(), 1)

    features = pd.DataFrame([{
        "analyst_hit_rate": analyst_hit_rate,
        "analyst_n_prior_resolved": analyst_n,
        "broker_hit_rate": broker_hit_rate,
        "broker_n_prior_resolved": broker_n,
        "target_spread_clipped": target_spread_clipped,
        "ticker_volatility_90d_clipped": volatility_clipped,
        "rating_action_code": action_code,
    }])[FEATURE_COLS]

    probability_hit = float(classifier.predict_proba(features)[:, 1][0])

    # Build the survival curve across all bins to get a median-hit-day estimate
    hazard_rows = pd.DataFrame([
        {**features.iloc[0].to_dict(), "bin": b} for b in range(N_BINS)
    ])
    hazards = hazard_model.predict_proba(hazard_rows[FEATURE_COLS + ["bin"]])[:, 1]
    survival = np.cumprod(1 - hazards)
    below_half = np.where(survival <= 0.5)[0]
    predicted_median_days = float((below_half[0] + 1) * BIN_DAYS) if len(below_half) > 0 else None

    return PredictResponse(
        ticker=req.ticker.upper(),
        analyst_name=req.analyst_name,
        current_price=round(float(current_price), 2),
        price_target=req.price_target,
        target_spread_pct=round(target_spread * 100, 1),
        probability_hit=round(probability_hit, 3),
        predicted_median_days=predicted_median_days,
        analyst_hit_rate=round(analyst_hit_rate, 3),
        analyst_n_prior_resolved=analyst_n,
        broker_hit_rate=round(broker_hit_rate, 3),
        broker_n_prior_resolved=broker_n,
    )
