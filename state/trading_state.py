from typing import TypedDict, List, Optional, Literal, Dict, Any
from state.contracts import TTSOutput, CEOutput, SIVOutput
from langgraph.graph.message import add_messages


class TradingState(TypedDict):
    target_date: str
    currency_pair: str
    price: float
    calibration_threshold: float

    ce_output: Dict[str, Any]
    tts_output: Dict[str, Any]
    siv_output: Dict[str, Any]

    ce_score: float
    ce_confidence: float
    tts_score: float
    risk_penalty: float

    verdict: str
    verdict_reasoning: str

    weighted_score: float
    risk_multiplier: float

    debug_log: List[str]
    retry_count: int
    action: str

    backtest_mode: bool
    atr: float
    skip_llm: bool
    regime: str          # ADD — populated from tts_output["regime"]