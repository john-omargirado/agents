import sys
from pathlib import Path

# ensure project root is importable when running this script directly
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agents.tts_agent import tts_agent
from state.trading_state import TradingState


state: TradingState = {
    "target_date": "08/01/2018",
    "currency_pair": "USDJPY",
    "price": 111.67,
    "calibration_threshold": 0.20,

    "ce_output": {
        "sentiment": "NEUTRAL",
        "raw_vibe": "NEUTRAL",
        "mean_score": 0.0,
        "sentiment_score": 0.0,
        "article_count": 0,
        "confidence": "LOW",
        "error": None
    },

    "tts_output": {
        "total_score": 0.0,
        "ema_trend": "SIDEWAYS",
        "ema_score": 0.0,
        "rsi_value": 50.0,
        "rsi_score": 0.0,
        "bb_signal": "STABLE",
        "bb_score": 0.0,
        "breakout_score": 0.0,
        "price": 0.0,
        "ema_200_confidence": 0.0,
        "ema_200_reliable": False,
        "data_stale": False,
        "rows_available": 0,
        "tts_insufficient": True,
        "error": None
    },

    "siv_output": {
        "signal": "INCOHERENT",
        "conflict_type": "UNCLEAR",
        "price_deviation": 0.0,
        "issues": [],
        "tts_insufficient": True,
        "data_quality_ok": False,
        "explanation": "Not yet run"
    },

    "verdict": "HOLD",
    "verdict_reasoning": "",
    "weighted_score": 0.0,
    "risk_multiplier": 0.0,

    "debug_log": [],
    "retry_count": 0,
    "action": "NONE"
}

print("\n========================")
print("RUNNING TTS AGENT TEST")
print("========================\n")

res = tts_agent(state)

tts = res.get("tts_output", {})

print("\n========================")
print("RESULT SUMMARY")
print("========================")

print("Decision:", tts.get("decision", "HOLD"))
print("Total Score:", tts.get("total_score"))
print("EMA Trend:", tts.get("ema_trend"))
print("RSI:", tts.get("rsi_value"))
print("BB Signal:", tts.get("bb_signal"))
print("Breakout Score:", tts.get("breakout_score"))
print("LLM Error:", tts.get("error"))

print("\n========================")
print("DEBUG LOG (LAST 20)")
print("========================")

for line in state["debug_log"][-20:]:
    print(line)