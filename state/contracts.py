from typing import TypedDict, Optional, List, Literal


class TTSOutput(TypedDict):
    # ── core ──────────────────────────────────────────────────────────────────
    decision: Literal["BUY", "SELL", "HOLD"]
    tts_score: float
    total_score: float
    price: float
    explanation: str

    # ── trend / EMA ───────────────────────────────────────────────────────────
    ema_trend: Literal["BULLISH", "BEARISH", "SIDEWAYS"]
    ema_200_confidence: float
    ema_200_reliable: bool

    # ── RSI ───────────────────────────────────────────────────────────────────
    rsi: float

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_signal: Literal["OVERBOUGHT", "OVERSOLD", "STABLE"]

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd_direction_score: float
    is_macd_cross: bool

    # ── Breakout ──────────────────────────────────────────────────────────────
    breakout_signal: Literal["BREAKOUT_UP", "BREAKOUT_DOWN", "NONE"]
    breakout_strength: float           # 0.0 – 1.0
    breakout_high: Optional[float]     # resistance level tested
    breakout_low: Optional[float]      # support level tested

    # ── regime / meta ─────────────────────────────────────────────────────────
    regime: Literal["TRENDING", "RANGING", "TRANSITIONAL"]
    data_stale: bool
    error: Optional[str]

    atr: float


class CEOutput(TypedDict):
    sentiment: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    raw_vibe: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]
    ce_score: float
    ce_confidence: float
    article_count: int
    raw_article_count: int
    confidence: Literal["HIGH", "MODERATE", "LOW"]
    explanation: str
    error: Optional[str]

class SIVOutput(TypedDict):
    signal: Literal["COHERENT", "PARTIAL", "INCOHERENT"]
    conflict_type: str
    price_deviation: float
    issues: List[str]
    tts_insufficient: bool
    data_quality_ok: bool
    explanation: str


class VerdictOutput(TypedDict):
    verdict: Literal["BUY", "SELL", "HOLD"]
    weighted_score: float
    risk_multiplier: float
    verdict_reasoning: str

class TradeOutput(TypedDict):
    account_capital: float
    position_size: float
    risk_amount: float
    max_exposure: float
    sl_distance: float
    tp_distance: float
    atr: float