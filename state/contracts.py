from typing import TypedDict, Optional, List, Literal


class TTSOutput(TypedDict):
    total_score: float
    ema_trend: Literal["BULLISH", "BEARISH", "SIDEWAYS"]
    ema_score: float
    rsi_value: float
    rsi_score: float
    bb_signal: Literal["OVERBOUGHT", "OVERSOLD", "STABLE"]
    bb_score: float
    breakout_score: float
    price: float
    ema_200_confidence: float
    ema_200_reliable: bool
    data_stale: bool
    rows_available: int
    tts_insufficient: bool
    error: Optional[str]
    explanation: str   


class CEOutput(TypedDict):
    sentiment: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    raw_vibe: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]
    sentiment_score: float
    mean_score: float
    article_count: int
    confidence: Literal["HIGH", "MODERATE", "LOW"]
    error: Optional[str]
    explanation: str 


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
    reasoning: str