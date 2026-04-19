import pandas as pd
import numpy as np

def calculate_technical_indicators(full_df: pd.DataFrame, target_date: str):
    target_dt = pd.to_datetime(target_date)
    df = full_df[full_df['timestamp'] <= target_dt].copy()

    row_count = len(df)

    # Hard minimum — can't compute anything useful
    if df.empty or row_count < 30:
        return None

    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col])

    # =========================
    # Sufficiency flags
    # =========================
    sufficient = {
        "ema_50_reliable":  row_count >= 50,   # 3x span
        "ema_200_reliable": row_count >= 200,   # 3x span (strict) or 200 (minimum)
        "rsi_reliable":     row_count >= 28,
        "bb_reliable":      row_count >= 40,
        "breakout_reliable": row_count >= 20,
    }

    # Warn in logs if EMA-200 is being run on thin data
    ema_200_confidence = min(row_count / 200, 1.0)  # 0.0 → 1.0 scale

    last_close = float(df['close'].iloc[-1])

    # --- Staleness check ---
    last_candle = df['timestamp'].iloc[-1]
    days_stale = (target_dt - last_candle).days
    data_stale = days_stale > 5

    # --- EMA ---
    ema_50  = df['close'].ewm(span=50,  adjust=False).mean().iloc[-1]
    ema_200 = df['close'].ewm(span=200, adjust=False).mean().iloc[-1]

    trend_diff = (ema_50 - ema_200) / ema_200

    if trend_diff > 0.001:
        trend = "BULLISH"
    elif trend_diff < -0.001:
        trend = "BEARISH"
    else:
        trend = "SIDEWAYS"

    # Discount trend_strength if EMA-200 unreliable
    raw_trend_strength = min(abs(trend_diff) / 0.02, 1.0)
    trend_strength = raw_trend_strength * ema_200_confidence

    # --- RSI ---
    delta = df['close'].diff()
    gain  = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    rs    = gain.iloc[-1] / (loss.iloc[-1] + 1e-9)
    rsi   = 100 - (100 / (1 + rs))

    rsi_strength = 0.0
    if rsi > 60:
        rsi_strength = (rsi - 60) / 40
    elif rsi < 40:
        rsi_strength = (40 - rsi) / 40
    rsi_strength = min(max(rsi_strength, 0.0), 1.0)

    # --- Bollinger Bands ---
    sma_20  = df['close'].rolling(window=20).mean()
    std_20  = df['close'].rolling(window=20).std()
    upper_bb = (sma_20 + 2 * std_20).iloc[-1]
    lower_bb = (sma_20 - 2 * std_20).iloc[-1]
    band_width = upper_bb - lower_bb

    bb_signal   = "STABLE"
    bb_strength = 0.0

    if last_close >= upper_bb:
        bb_signal   = "OVERBOUGHT"
        bb_strength = min(abs(last_close - upper_bb) / (band_width / 2 + 1e-9), 1.0)
    elif last_close <= lower_bb:
        bb_signal   = "OVERSOLD"
        bb_strength = min(abs(lower_bb - last_close) / (band_width / 2 + 1e-9), 1.0)

    return {
        "price": round(last_close, 5),
        "trend":          trend,
        "trend_strength": round(trend_strength, 4),
        "rsi":          round(rsi, 2),
        "rsi_strength": round(rsi_strength, 4),
        "bb_signal":   bb_signal,
        "bb_strength": round(bb_strength, 4),
        "ema_50":  round(ema_50,  5),
        "ema_200": round(ema_200, 5),
        # flattened — no longer nested under "data_quality"
        "rows_available":     row_count,
        "ema_200_confidence": round(ema_200_confidence, 3),
        "ema_200_reliable":   sufficient["ema_200_reliable"],
        "ema_50_reliable":    sufficient["ema_50_reliable"],
        "data_stale":         data_stale,
        "days_stale":         int(days_stale),
    }
    