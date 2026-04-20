from typing import TypedDict, List
from state.contracts import TTSOutput, CEOutput, SIVOutput


class TradingState(TypedDict):
    target_date: str
    currency_pair: str
    price: float
    calibration_threshold: float

    ce_output: CEOutput
    tts_output: TTSOutput
    siv_output: SIVOutput

    verdict: str
    verdict_reasoning: str

    weighted_score: float
    risk_multiplier: float

    debug_log: List[str]
    retry_count: int
    action: str