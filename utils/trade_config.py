# =========================
# utils/trade_config.py
# =========================

ATR_LOOKBACK = 14

# Account settings
ACCOUNT_EQUITY  = 10_000.0   # USD
RISK_LIMIT_PCT  = 0.01       # 1% risk per trade

# =========================
# PAIR CONFIG
# sl_mult          : k in SL_Adaptive = P_entry ± (k × ATR)
# rr_ratio         : TP = SL_distance × RR
# pip_value_per_lot: currency units risked per lot per pip (mini lot = 10k units)
# =========================
PAIR_CONFIG = {
    # --- USD Majors ---
    "AUDUSD": {"sl_mult": 1.5, "rr_ratio": 2.0,  "pip_value_per_lot": 1000},
    "EURUSD": {"sl_mult": 1.5, "rr_ratio": 2.0,  "pip_value_per_lot": 1000},
    "GBPUSD": {"sl_mult": 2.0, "rr_ratio": 2.0,  "pip_value_per_lot": 1000},
    "USDCAD": {"sl_mult": 1.5, "rr_ratio": 2.0,  "pip_value_per_lot": 1000},
    "USDCHF": {"sl_mult": 1.5, "rr_ratio": 2.0,  "pip_value_per_lot": 1000},
    "USDJPY": {"sl_mult": 2.0, "rr_ratio": 1.75, "pip_value_per_lot": 1000},

    # --- PHP Exotics ---
    # Higher sl_mult: PHP pairs have wider spreads and more erratic intraday swings
    # Lower rr_ratio: harder to sustain large moves, take profit earlier
    # pip_value_per_lot kept at 1000 for normalized backtesting consistency
    "USDPHP": {"sl_mult": 2.5, "rr_ratio": 1.5,  "pip_value_per_lot": 1000},
    "EURPHP": {"sl_mult": 3.0, "rr_ratio": 1.5,  "pip_value_per_lot": 1000},
    "JPYPHP": {"sl_mult": 3.0, "rr_ratio": 1.5,  "pip_value_per_lot": 1000},
}

DEFAULT_CONFIG = {"sl_mult": 2.0, "rr_ratio": 1.75, "pip_value_per_lot": 1000}


def get_pair_config(pair: str) -> dict:
    return PAIR_CONFIG.get(pair.upper(), DEFAULT_CONFIG)