"""
Unit tests for the core data transformation logic. This is what
GitHub Actions CI runs on every push (see .github/workflows/ci.yml).

Run locally with:
    pip install pytest --break-system-packages
    pytest tests/test_pipeline.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
from labeling_lib import normalize_rating, assign_outcome_label, shrunk_hit_rate


class TestNormalizeRating:
    def test_standard_bullish_variants(self):
        assert normalize_rating("BUY") == "BULLISH"
        assert normalize_rating("Buy ") == "BULLISH"  # trailing whitespace
        assert normalize_rating("outperform") == "BULLISH"  # lowercase
        assert normalize_rating("OVERWEIGHT") == "BULLISH"

    def test_standard_bearish_variants(self):
        assert normalize_rating("SELL") == "BEARISH"
        assert normalize_rating("Underperform") == "BEARISH"

    def test_neutral_variants(self):
        assert normalize_rating("HOLD") == "NEUTRAL"
        assert normalize_rating("Market Perform") == "NEUTRAL"

    def test_missing_values(self):
        assert pd.isna(normalize_rating(np.nan))
        assert pd.isna(normalize_rating(None))

    def test_corrupted_transition_artifact(self):
        # Real data had mangled separators like "BUY È NEUTRAL" -- we
        # resolve to the LAST token, since that's the settled rating.
        assert normalize_rating("BUY È NEUTRAL") == "NEUTRAL"

    def test_truncated_tokens(self):
        assert normalize_rating("MARKET PERFO") == "NEUTRAL"
        assert normalize_rating("SECTOR OUTP") == "BULLISH"

    def test_genuinely_unrecognizable_returns_unknown(self):
        assert normalize_rating("XYZQPR123") == "UNKNOWN"


class TestAssignOutcomeLabel:
    def test_achieved_when_hit_before_broken(self):
        result = assign_outcome_label(
            hit_date=pd.Timestamp("2024-01-10"),
            broken_date=pd.Timestamp("2024-01-20"),
            window_end=pd.Timestamp("2024-12-31"),
            last_price_date=pd.Timestamp("2025-01-01"),
        )
        assert result == "TARGET_ACHIEVED"

    def test_broken_when_broken_before_hit(self):
        result = assign_outcome_label(
            hit_date=pd.Timestamp("2024-06-01"),
            broken_date=pd.Timestamp("2024-01-20"),
            window_end=pd.Timestamp("2024-12-31"),
            last_price_date=pd.Timestamp("2025-01-01"),
        )
        assert result == "THESIS_BROKEN"

    def test_active_when_window_not_yet_elapsed(self):
        result = assign_outcome_label(
            hit_date=pd.NaT, broken_date=pd.NaT,
            window_end=pd.Timestamp("2026-12-31"),
            last_price_date=pd.Timestamp("2026-07-24"),
        )
        assert result == "ACTIVE"

    def test_expired_when_window_elapsed_with_no_event(self):
        result = assign_outcome_label(
            hit_date=pd.NaT, broken_date=pd.NaT,
            window_end=pd.Timestamp("2024-01-01"),
            last_price_date=pd.Timestamp("2025-01-01"),
        )
        assert result == "TIME_EXPIRED"

    def test_no_price_data_edge_case(self):
        result = assign_outcome_label(
            hit_date=pd.NaT, broken_date=pd.NaT,
            window_end=pd.Timestamp("2024-01-01"),
            last_price_date=pd.NaT,
        )
        assert result == "NO_PRICE_DATA"


class TestShrunkHitRate:
    def test_zero_prior_reduces_to_global_rate(self):
        # This is THE leakage-safety guarantee: an analyst's first-ever
        # call must show exactly the global baseline, nothing else.
        rate = shrunk_hit_rate(prior_sum=0, prior_count=0, k=10, global_rate=0.574)
        assert abs(rate - 0.574) < 1e-9

    def test_large_sample_trusts_own_rate_more(self):
        # With 1000 prior calls all hits, the shrinkage should barely
        # pull away from the analyst's own near-100% rate.
        rate = shrunk_hit_rate(prior_sum=1000, prior_count=1000, k=10, global_rate=0.574)
        assert rate > 0.98

    def test_small_sample_pulled_toward_baseline(self):
        # 1 prior call, a miss -- shouldn't crash to 0%, should be pulled
        # substantially toward the global rate.
        rate = shrunk_hit_rate(prior_sum=0, prior_count=1, k=10, global_rate=0.574)
        assert 0.4 < rate < 0.574  # pulled toward baseline, not stuck at 0
