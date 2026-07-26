"""
Shared, testable core logic -- extracted from the pipeline scripts so it
can be unit tested independently (this is what a CI pipeline actually
checks against on every push).
"""
import re
import numpy as np
import pandas as pd

BULLISH = {
    "BUY", "OUTPERFORM", "OVERWEIGHT", "POSITIVE", "STRONG BUY", "STRONGBUY",
    "TOP PICK", "TOPPICK", "ACCUMULATE", "MARKET OUTPERFORM", "MARKETOUTPERFORM",
    "SECTOR OUTPERFORM", "SECTOROUTPERFORM", "SPECULATIVE BUY", "OUTPERFORMER",
    "MKT OUTPERFORM", "MARKET OUTP", "MARKET OUTPERF", "SECTOR OUTP",
    "SECTOR OUTPERF",
}
NEUTRAL = {
    "NEUTRAL", "HOLD", "EQUAL WEIGHT", "EQUALWEIGHT", "MARKET PERFORM",
    "MARKETPERFORM", "SECTOR PERFORM", "SECTORPERFORM", "IN LINE", "INLINE",
    "PERFORM", "SECTOR WEIGHT", "PEER PERFORM", "PEERPERFORM", "MKT PERFORM",
    "MIXED", "MARKET PERFO", "MARKET PERF", "SECTOR PERFO", "SECTOR PERF", "SEC",
}
BEARISH = {
    "UNDERPERFORM", "UNDERWEIGHT", "SELL", "NEGATIVE", "REDUCE", "SHORT",
    "AVOID", "UNDERPERFORMER", "UNDER PERFORM",
}
TOKEN_CLEAN_RE = re.compile(r"[\"'.,;()\\]")


def normalize_rating(raw):
    """Map a raw, messy rating string to BULLISH / NEUTRAL / BEARISH / UNKNOWN."""
    if pd.isna(raw):
        return np.nan
    s = str(raw).upper().strip()
    s = TOKEN_CLEAN_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s in ("", "NAN", "NO RATING AT TIME"):
        return np.nan
    for sep_pattern in [r"\s*È\s*", r"Å", r"\s+TO\s+", r"�+"]:
        parts = re.split(sep_pattern, s)
        if len(parts) > 1:
            s = parts[-1].strip()
    if s in BULLISH:
        return "BULLISH"
    if s in NEUTRAL:
        return "NEUTRAL"
    if s in BEARISH:
        return "BEARISH"
    return "UNKNOWN"


def assign_outcome_label(hit_date, broken_date, window_end, last_price_date):
    """Priority: Achieved > Broken > Active/Expired. All args are
    comparable (datetime-like) or None/NaT for 'did not happen'."""
    hit_happened = pd.notna(hit_date)
    broken_happened = pd.notna(broken_date)

    if hit_happened and (not broken_happened or hit_date <= broken_date):
        return "TARGET_ACHIEVED"
    if broken_happened and (not hit_happened or broken_date < hit_date):
        return "THESIS_BROKEN"
    if pd.isna(last_price_date):
        return "NO_PRICE_DATA"
    if window_end > last_price_date:
        return "ACTIVE"
    return "TIME_EXPIRED"


def shrunk_hit_rate(prior_sum, prior_count, k, global_rate):
    """Bayesian-style shrinkage: blends an entity's own track record with
    the global baseline, trusting the entity's own rate more as prior_count
    grows. With prior_count=0, this reduces exactly to global_rate."""
    return (prior_sum + k * global_rate) / (prior_count + k)
