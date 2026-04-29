from typing import TypedDict, Optional, List, Literal


class TTSOutput(TypedDict):
    decision: Literal["BUY", "SELL", "HOLD"]      # was missing entirely
    total_score: float
    price: float
    ema_trend: Literal["BULLISH", "BEARISH", "SIDEWAYS"]
    rsi: float                                      # was rsi_value — wrong name
    bb_signal: Literal["OVERBOUGHT", "OVERSOLD", "STABLE"]
    macd_hist: float                                # new
    ema_200_confidence: float
    ema_200_reliable: bool
    data_stale: bool
    explanation: str
    error: Optional[str]


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
    atr: float
    sl_distance: float
    tp_distance: float
    verdict_reasoning: str      # was "reasoning" — agent returns "verdict_reasoning"
    action: str 