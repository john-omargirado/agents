from typing import TypedDict, List, Dict, Any

class TradingState(TypedDict):
    target_date: str
    currency_pair: str
    price: float
    calibration_threshold: float
    ce_output: Dict[str, Any]
    tts_output: Dict[str, Any]
    siv_output: Dict[str, Any]
    verdict: str
    verdict_reasoning: str
    risk_multiplier: float
    debug_log: List[str]
    retry_count: int      # FIX: was missing, caused infinite retry loop
    action: str           # FIX: was missing, route_after_verdict couldn't persist it