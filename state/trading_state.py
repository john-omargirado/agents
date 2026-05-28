from typing import TypedDict, List, Dict, Any, Optional
from state.contracts import TTSOutput, CEOutput, SIVOutput


class TradingState(TypedDict):
    # ── INPUT ─────────────────────────────────────────────
    target_date: str
    currency_pair: str
    price: float
    calibration_threshold: float

    # ── AGENT OUTPUTS ─────────────────────────────────────
    ce_output: CEOutput
    tts_output: TTSOutput
    siv_output: SIVOutput

    # ── CORE SCORES ───────────────────────────────────────
    ce_score: float
    ce_confidence: float
    tts_score: float
    risk_penalty: float

    # ── VERDICT ───────────────────────────────────────────
    verdict: str
    verdict_reasoning: str
    weighted_score: float
    risk_multiplier: float

    # ── TRADE OUTPUT (STRUCTURED EXECUTION LAYER) ─────────
    trade_output: Dict[str, Any]
    # expected keys:
    # position_size, risk_amount, max_exposure,
    # sl_distance, tp_distance, atr

    # ── SYSTEM ────────────────────────────────────────────
    debug_log: List[str]
    retry_count: int
    action: str

    # ── MODES ─────────────────────────────────────────────
    backtest_mode: bool
    live_mode: bool
    skip_llm: bool
    calibration_mode: bool 

    # ── MARKET CONTEXT ────────────────────────────────────
    atr: float
    regime: str

    # ── RISK MANAGEMENT ───────────────────────────────────
    account_capital: float
    leverage: str
    risk_per_trade: float

    # ── TRADER PROFILE ─────────────────────────────────────
    experience_level: Optional[str]

    precomputed_indicators: Optional[Any]