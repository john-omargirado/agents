# =========================
# utils/trade_config.py
# =========================

ATR_LOOKBACK = 14

# Account settings
ACCOUNT_EQUITY  = 10_000.0   # USD
RISK_LIMIT_PCT  = 0.01       # 1% risk per trade

# =========================
# PAIR CONFIG
# sl_mult  : SL = ATR × sl_mult  (1.0 = 1× ATR, tight; 1.5 = wider swing buffer)
# rr_ratio : TP = SL × rr_ratio  (2.0 = standard 1:2 risk/reward)
# Daily ATR reference:
#   EURUSD ~60-80 pips | GBPUSD ~80-100 pips | USDJPY ~60-80 pips
#   USDCAD ~70-90 pips | USDCHF ~60-80 pips  | AUDUSD ~60-75 pips
# =========================
PAIR_CONFIG = {
    # --- USD Majors ---
    "AUDUSD": {"sl_mult": 1.0, "rr_ratio": 2.0,  "pip_value_per_lot": 1000},
    "EURUSD": {"sl_mult": 1.0, "rr_ratio": 2.0,  "pip_value_per_lot": 1000},
    "GBPUSD": {"sl_mult": 1.0, "rr_ratio": 2.0,  "pip_value_per_lot": 1000},
    "USDCAD": {"sl_mult": 1.0, "rr_ratio": 2.0,  "pip_value_per_lot": 1000},
    "USDCHF": {"sl_mult": 1.0, "rr_ratio": 2.0,  "pip_value_per_lot": 1000},
    "USDJPY": {"sl_mult": 1.0, "rr_ratio": 2.0,  "pip_value_per_lot": 1000},

    # --- Cross pairs ---
    "NZDUSD": {"sl_mult": 1.0, "rr_ratio": 2.0,  "pip_value_per_lot": 1000},
    "EURGBP": {"sl_mult": 1.0, "rr_ratio": 2.0,  "pip_value_per_lot": 1000},
    "EURJPY": {"sl_mult": 1.0, "rr_ratio": 2.0,  "pip_value_per_lot": 1000},
    "GBPJPY": {"sl_mult": 1.0, "rr_ratio": 2.0,  "pip_value_per_lot": 1000},

    # --- PHP Exotics ---
    "USDPHP": {"sl_mult": 1.5, "rr_ratio": 1.5,  "pip_value_per_lot": 1000},
    "EURPHP": {"sl_mult": 1.5, "rr_ratio": 1.5,  "pip_value_per_lot": 1000},
    "JPYPHP": {"sl_mult": 1.5, "rr_ratio": 1.5,  "pip_value_per_lot": 1000},
}

DEFAULT_CONFIG = {"sl_mult": 1.0, "rr_ratio": 2.0, "pip_value_per_lot": 1000}


def get_pair_config(pair: str) -> dict:
    # Normalize: "EUR/USD" → "EURUSD"
    normalized = pair.upper().replace("/", "").replace("-", "")
    return PAIR_CONFIG.get(normalized, DEFAULT_CONFIG)