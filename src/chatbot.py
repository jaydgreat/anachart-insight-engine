"""
Step 10: Streamlit chatbot -- natural language interface on top of the
FastAPI /predict service, using Claude for extraction + grounded generation.

Run with:
    export ANTHROPIC_API_KEY=your_key_here   (or set it in your shell profile)
    streamlit run chatbot.py

IMPORTANT: the FastAPI server (api.py) must ALSO be running separately
(uvicorn api:app --reload) in another terminal, since this chatbot calls it.

THE RAG PATTERN, made concrete:
--------------------------------
1. EXTRACT: ask Claude to turn the user's free-text question into
   structured fields (ticker, analyst, target, rating) as JSON.
2. RETRIEVE: send that JSON to YOUR OWN /predict endpoint -- this is the
   "retrieval" step. The numbers that come back are real, computed by
   your actual trained models, not invented by the LLM.
3. GENERATE: give Claude the retrieved JSON and ask it to phrase a plain-
   English answer, explicitly forbidding it from adding numbers that
   aren't in the JSON. This is what prevents hallucinated statistics.
"""
import json
import os
import requests
import pandas as pd
import streamlit as st
from anthropic import Anthropic

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")
MODEL = "claude-sonnet-5"

client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment

st.set_page_config(page_title="AnaChart Insight Chatbot", page_icon="📈")
st.title("📈 AnaChart Insight Chatbot")
st.caption("Ask about any analyst's price target -- e.g. "
           "\"How realistic is Gene Munster's $380 target on Apple?\"")

# --- Load the known ticker/analyst universe once, for the sidebar picker ---
@st.cache_data
def load_ticker_analyst_lists():
    df = pd.read_csv("model_ready_data.csv", usecols=["ticker", "analyst_name"])
    tickers = sorted(df["ticker"].unique())
    # ticker -> sorted list of analysts who have actually covered it
    ticker_to_analysts = {
        t: sorted(df.loc[df["ticker"] == t, "analyst_name"].unique())
        for t in tickers
    }
    return tickers, ticker_to_analysts

TICKERS, TICKER_TO_ANALYSTS = load_ticker_analyst_lists()

@st.cache_data
def compute_leaderboard(min_resolved_calls: int = 5):
    """Rank analysts by their hit rate, using each row's own leakage-safe
    hit rate feature (already shrinkage-adjusted) at their MOST RECENT
    call -- this reflects each analyst's full track record to date."""
    df = pd.read_csv(
        "model_ready_data.csv",
        usecols=["analyst_name", "ticker", "call_date", "analyst_hit_rate", "analyst_n_prior_resolved"],
        parse_dates=["call_date"],
    )
    tickers_covered = df.groupby("analyst_name")["ticker"].nunique()

    latest = df.sort_values("call_date").groupby("analyst_name").last().reset_index()
    latest = latest[latest["analyst_n_prior_resolved"] >= min_resolved_calls]
    latest["tickers_covered"] = latest["analyst_name"].map(tickers_covered)

    latest = latest.rename(columns={
        "analyst_name": "Analyst",
        "analyst_hit_rate": "Hit rate",
        "analyst_n_prior_resolved": "Resolved calls",
        "tickers_covered": "Tickers covered",
    })
    latest["Hit rate"] = (latest["Hit rate"] * 100).round(1)
    return latest[["Analyst", "Hit rate", "Resolved calls", "Tickers covered"]].sort_values(
        "Hit rate", ascending=False
    ).reset_index(drop=True)

with st.sidebar:
    st.header("Quick Lookup")
    st.caption("Pick from real tickers/analysts in the dataset -- guarantees a match.")
    picked_ticker = st.selectbox("Ticker", TICKERS)
    picked_analyst = st.selectbox("Analyst", TICKER_TO_ANALYSTS[picked_ticker])
    picked_rating = st.radio("Rating", ["BULLISH", "BEARISH"], horizontal=True)
    picked_target = st.number_input("Price target ($)", min_value=0.0, value=100.0, step=1.0)
    run_lookup = st.button("Ask about this call", type="primary")

    st.divider()
    show_leaderboard = st.checkbox("Show analyst leaderboard")

if "messages" not in st.session_state:
    st.session_state.messages = []

EXTRACTION_SYSTEM_PROMPT = """You extract structured data from questions about
stock analyst price targets. Given a user's question, respond with ONLY a
JSON object (no other text, no markdown fences) with these fields:

{
  "ticker": "<stock ticker, e.g. AAPL>",
  "analyst_name": "<analyst's full name, as best you can infer>",
  "price_target": <number>,
  "rating": "BULLISH" or "BEARISH",
  "broker": "<broker/firm name if mentioned, else null>",
  "needs_clarification": null
}

If the question is missing a REQUIRED field (ticker, price_target, or
whether it's bullish/bearish), instead respond with:
{"needs_clarification": "<a short, specific question asking for exactly what's missing>"}

The analyst_name is required too -- if no analyst is named, ask for one.
Infer BULLISH vs BEARISH from context (a price target ABOVE current
sentiment implies bullish; explicit words like "downgrade," "sell," or
"bearish" imply bearish) but don't guess wildly -- ask if truly unclear."""

GENERATION_SYSTEM_PROMPT = """You are a helpful financial assistant. You will
be given a user's question AND a JSON object containing real, computed
model output (probabilities, historical hit rates, etc.). Answer the
user's question in plain, conversational English using ONLY the numbers
in the JSON -- do not invent, estimate, or add any statistic that isn't
explicitly present in the JSON. If the JSON contains an error field,
explain the issue plainly instead of making up an answer.
Keep the answer to 2-4 sentences."""


def extract_query_params(user_question: str) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_question}],
    )
    raw_text = response.content[0].text.strip()
    return json.loads(raw_text)


def call_predict_api(params: dict) -> dict:
    payload = {
        "ticker": params["ticker"],
        "analyst_name": params["analyst_name"],
        "price_target": params["price_target"],
        "rating": params["rating"],
        "broker": params.get("broker") or "UNKNOWN",
    }
    try:
        resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Could not reach the prediction API. Is 'uvicorn api:app --reload' running?"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"API returned an error: {e.response.text}"}


def show_price_trend(ticker: str, price_target: float = None):
    """Fetch and render a trend line for the ticker, with an optional
    horizontal reference line marking the analyst's price target."""
    try:
        resp = requests.get(f"{API_BASE}/price_history/{ticker}", params={"days": 180}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException:
        return  # silently skip the chart if unavailable -- not critical to the answer

    chart_df = pd.DataFrame({"close": data["close"]}, index=pd.to_datetime(data["dates"]))
    if price_target:
        chart_df["target"] = price_target
    st.line_chart(chart_df, height=220)


def generate_grounded_answer(user_question: str, model_result: dict) -> str:
    grounding_message = (
        f"User's question: {user_question}\n\n"
        f"Model output (JSON):\n{json.dumps(model_result, indent=2)}"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=GENERATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": grounding_message}],
    )
    return response.content[0].text.strip()


# --- Chat UI ---------------------------------------------------------
if show_leaderboard:
    st.subheader("Analyst leaderboard")
    st.caption("Ranked by historical hit rate. Minimum 5 resolved calls, to avoid small-sample noise.")
    st.dataframe(compute_leaderboard(min_resolved_calls=5), hide_index=True, use_container_width=True)
    st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Sidebar "Quick Lookup" button: we already have exact, guaranteed-valid
# values from the dropdowns, so skip the free-text extraction step
# entirely and go straight to RETRIEVE -> GENERATE. More reliable than
# parsing free text, since there's no ambiguity to get wrong.
if run_lookup:
    synthetic_question = (
        f"How realistic is {picked_analyst}'s ${picked_target:.0f} "
        f"{picked_rating.lower()} target on {picked_ticker}?"
    )
    st.session_state.messages.append({"role": "user", "content": synthetic_question})
    with st.chat_message("user"):
        st.write(synthetic_question)
    with st.chat_message("assistant"):
        params = {
            "ticker": picked_ticker, "analyst_name": picked_analyst,
            "price_target": picked_target, "rating": picked_rating,
        }
        with st.spinner(f"Looking up {picked_ticker}..."):
            model_result = call_predict_api(params)
        with st.spinner("Composing answer..."):
            answer = generate_grounded_answer(synthetic_question, model_result)
        st.write(answer)
        show_price_trend(picked_ticker, picked_target)
        st.session_state.messages.append({"role": "assistant", "content": answer})

if user_question := st.chat_input("Ask about an analyst's price target..."):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Extracting details..."):
            try:
                params = extract_query_params(user_question)
            except json.JSONDecodeError:
                params = {"needs_clarification": "Sorry, I couldn't parse that -- could you rephrase with a ticker, analyst name, and price target?"}

        if params.get("needs_clarification"):
            answer = params["needs_clarification"]
        else:
            with st.spinner(f"Looking up {params['ticker']}..."):
                model_result = call_predict_api(params)
            with st.spinner("Composing answer..."):
                answer = generate_grounded_answer(user_question, model_result)

        st.write(answer)
        if not params.get("needs_clarification"):
            show_price_trend(params["ticker"], params.get("price_target"))
        st.session_state.messages.append({"role": "assistant", "content": answer})
