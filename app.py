"""
app.py — Flask Backend for Live MAS
"""

import os
import uuid
import logging
import traceback
import hmac
from dotenv import load_dotenv
from datetime import datetime, timezone
from typing import Dict, Any, Optional, cast


import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_core.runnables import RunnableConfig
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman

from graph.build_graph import build_graph
from agents.chat_agent import live_chat_agent
from memory.state_memory import state_memory
from state.trading_state import TradingState
from utils.data_loader import (
    load_ohlcv_data,
    load_news_for_pair,
    get_available_dates_for_pair,
    normalize_pair,
    get_next_candles,
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()
IS_DEV = os.environ.get("FLASK_ENV") == "development"

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]  # global fallback
)

if IS_DEV:
    CORS(app, resources={
        r"/api/*": {
            "origins": [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:3000",
            ]
        }
    })
else:
    CORS(app, resources={
        r"/api/*": {
            "origins": [
                "https://forex-mas.me",
                "https://www.forex-mas.me"
            ]
        }
    })

app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

Talisman(app, force_https=False, content_security_policy=False)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_run_cache: Dict[str, Dict[str, Any]] = {}
_MAX_CACHE = 50

@app.before_request
def basic_abuse_protection():
    ip = get_remote_address()

@app.before_request
def check_api_key():
    PUBLIC_ROUTES = ["/api/health", "/api/pairs", "/api/strategies"]

    if IS_DEV:
        return

    if request.path.startswith("/api/") and request.path not in PUBLIC_ROUTES:
        key = request.headers.get("X-API-KEY")

        if not key or not hmac.compare_digest(key, os.environ.get("API_KEY", "")):
            return jsonify({"error": "Unauthorized"}), 401

def validate_payload(body, required_fields):
    for field in required_fields:
        if field not in body:
            return False
    return True


def _cache_run(analysis_id: str, state: Dict[str, Any]):
    _run_cache[analysis_id] = state
    if len(_run_cache) > _MAX_CACHE:
        _run_cache.pop(next(iter(_run_cache)))


def _get_cached_run(analysis_id: Optional[str]) -> Optional[Dict[str, Any]]:
    return _run_cache.get(analysis_id) if analysis_id else None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD",
    "USD/CAD", "USD/CHF", "NZD/USD", "EUR/GBP",
    "EUR/JPY", "GBP/JPY",
]

SUPPORTED_STRATEGIES = [
    {"id": "ema_trend", "name": "EMA Trend"},
    {"id": "rsi", "name": "RSI"},
    {"id": "bollinger", "name": "Bollinger Bands"},
    {"id": "macd", "name": "MACD"},
    {"id": "breakout", "name": "Breakout"},
]


# ---------------------------------------------------------------------------
# State builder
# ---------------------------------------------------------------------------

def _build_initial_state(body: dict) -> TradingState:
    print(f"[STATE_BUILDER] riskThreshold raw='{body.get('riskThreshold')}' → risk_per_trade={float(body.get('riskThreshold') or 1.0)}")
    pair = str(body.get("currency_pair", "EUR/USD")).upper().replace("-", "/")

    return cast(TradingState, {
        "currency_pair": pair,
        "target_date": body.get("target_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d"),

        # FIX: explicit mode support
        "live_mode": body.get("live_mode", True),
        "backtest_mode": body.get("backtest_mode", False),

        "skip_llm": bool(body.get("skip_llm", False)),

        "account_capital": float(body.get("accountCapital") or 0.0),
        "leverage": body.get("leverage", "1:1"),
        "risk_per_trade":  float(body.get("riskThreshold") or 1.0),
        "experience_level": body.get("experience_level") or None,

        "price": 0.0,
        "atr": 0.0,

        "ce_score": 0.0,
        "ce_confidence": 0.0,
        "tts_score": 0.0,
        "risk_penalty": 0.0,
        "weighted_score": 0.0,
        "risk_multiplier": 0.0,

        "verdict": "",
        "verdict_reasoning": "",
        "action": "",
        "regime": "",

        "retry_count": 0,
        "debug_log": [],

        "ce_output": {},
        "tts_output": {},
        "siv_output": {},

        "sl_distance": None,
        "tp_distance": None,
    })



# ---------------------------------------------------------------------------
# Serializer (kept same but safe CE access)
# ---------------------------------------------------------------------------

def _serialize_state(state: dict, analysis_id: str, pair: str) -> dict:
    tts = state.get("tts_output", {}) or {}
    ce = state.get("ce_output", {}) or {}
    siv = state.get("siv_output", {}) or {}
    trade      = state.get("trade_output", {}) or {}

    def _f(v):
        try:
            return float(v)
        except:
            return None

    sl_dist = _f(state.get("sl_distance"))
    tp_dist = _f(state.get("tp_distance"))
    price = _f(tts.get("price"))

    return {
        "analysis_id": analysis_id,
        "currency_pair": pair,
        "target_date": state.get("target_date"),
        "timestamp": datetime.now(timezone.utc).isoformat(),

        "verdict": {
            "decision": state.get("verdict", "N/A"),
            "action": state.get("action", "N/A"),
            "weighted_score": state.get("weighted_score", 0.0),
            "verdict_reasoning": state.get("verdict_reasoning", ""),
        },

        "tts": tts,

        "ce": {
            "sentiment": ce.get("sentiment"),
            "raw_vibe": ce.get("raw_vibe"),
            "ce_score": ce.get("ce_score"),
            "ce_confidence": ce.get("ce_confidence"),
            "article_count": ce.get("article_count"),
            "raw_article_count": ce.get("raw_article_count"),
            "confidence": ce.get("confidence"),
            "explanation": ce.get("explanation"),
        },

        "siv": siv,
        
         "trade": {
            "position_size": _f(trade.get("position_size")),
            "risk_amount":   _f(trade.get("risk_amount")),
            "max_exposure":  _f(trade.get("max_exposure")),
            "sl_distance":   _f(trade.get("sl_distance")),
            "tp_distance":   _f(trade.get("tp_distance")),
            "atr":           _f(trade.get("atr")),
        },

        "meta": {
            "debug_log": state.get("debug_log", [])[-10:],
            "retry_count": state.get("retry_count", 0),
        },
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/pairs")
def pairs():
    return jsonify({"pairs": SUPPORTED_PAIRS})


@app.route("/api/strategies")
def strategies():
    return jsonify({"strategies": SUPPORTED_STRATEGIES})


# ---------------------------------------------------------------------------
# ANALYZE (LIVE)
# ---------------------------------------------------------------------------

@app.route("/api/analyze", methods=["POST"])
@limiter.limit("10 per minute")
def analyze(): 
    try:
        body = request.get_json(force=True) or {}
        pair = str(body.get("currency_pair", "EUR/USD")).upper()

        initial_state = _build_initial_state(body)

        analysis_id = str(uuid.uuid4())

        graph = build_graph()
        config: RunnableConfig = {
            "configurable": {"thread_id": f"{pair}_{analysis_id[:8]}"}
        }

        final_state = graph.invoke(initial_state, config=config)
        _cache_run(analysis_id, final_state)

        return jsonify(_serialize_state(final_state, analysis_id, pair))

    except Exception as e:
        logger.error(traceback.format_exc())  # log server-side only
        return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------------------------

@app.route("/api/backtest/analyze", methods=["POST"])
@limiter.limit("5 per minute")
def backtest_analyze():
    try:
        body = request.get_json(force=True) or {}
        pair = str(body.get("currency_pair", "EUR/USD")).upper()
        target_date: str = body.get("date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")  # ← typed + fallback

        initial_state = _build_initial_state(body)
        initial_state["live_mode"] = False      # ← was True (bug)
        initial_state["backtest_mode"] = True   # ← was False (bug)
        initial_state["target_date"] = target_date

        analysis_id = str(uuid.uuid4())

        graph = build_graph()
        config: RunnableConfig = {
            "configurable": {"thread_id": f"{pair}_{analysis_id[:8]}"}
        }

        final_state = graph.invoke(initial_state, config=config)
        _cache_run(analysis_id, final_state)

        return jsonify(_serialize_state(final_state, analysis_id, pair))

    except Exception as e:
        logger.error(traceback.format_exc())  # log server-side only
        return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# CHAT (FIXED CE STATE MAPPING)
# ---------------------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
@limiter.limit("30 per minute")
def chat():
    try:
        body = request.get_json(force=True) or {}
        message = body.get("message")
        pair = str(body.get("currency_pair", "")).upper()
        experience_level = body.get("experience_level") or None
        sim_result = body.get("sim_result") or None

        if not message:
            return jsonify({"error": "message required"}), 400

        state = _get_cached_run(body.get("analysis_id"))

        if not state and pair:
            recent = state_memory.get_history_for_chat(pair, limit=1)
            if recent:
                r = recent[0]
                state = {
                    "currency_pair": pair,
                    "verdict": r.get("verdict"),
                    "tts_output": r.get("tts_output"),
                    "ce_output": {
                        "sentiment": r.get("ce_sentiment"),
                        "ce_score": r.get("ce_score"),
                        "ce_confidence": r.get("ce_confidence"),
                        "article_count": r.get("ce_articles"),
                        "raw_article_count": r.get("ce_raw_articles"),
                        "confidence": r.get("ce_confidence_level"),
                        "raw_vibe": r.get("ce_raw_vibe"),
                    },
                }

        response = live_chat_agent.chat(
            message=message,
            state=state,
            history=body.get("history", []),
            experience_level=experience_level,
            sim_result=sim_result, 
        )

        return jsonify({"response": response})

    except Exception as e:
        logger.error(traceback.format_exc())  # log server-side only
        return jsonify({"error": "Internal server error"}), 500 


# ---------------------------------------------------------------------------
# HISTORY
# ---------------------------------------------------------------------------

@app.route("/api/history")
def history():
    pair = str(request.args.get("currency_pair", "")).upper()
    return jsonify({
        "pair": pair,
        "runs": state_memory.get_history_for_chat(pair, limit=10)
    })


# ---------------------------------------------------------------------------
# BACKTEST SUPPORT
# ---------------------------------------------------------------------------

@app.route("/api/backtest/dates")
def backtest_dates():
    pair = str(request.args.get("currency_pair", "")).upper()
    return jsonify({
        "pair": pair,
        "dates": get_available_dates_for_pair(pair)
    })


@app.route("/api/backtest/news")
def backtest_news():
    pair = str(request.args.get("currency_pair", "")).upper()
    date = request.args.get("date")

    news = load_news_for_pair(pair, date)
    base, quote = normalize_pair(pair)[:3], normalize_pair(pair)[3:]

    return jsonify({
        "pair": pair,
        "date": date,
        "base_articles": news["base"],
        "quote_articles": news["quote"],
        "base_currency": base,
        "quote_currency": quote,
    })


# ---------------------------------------------------------------------------
# SIMULATION HELPER
# ---------------------------------------------------------------------------

def _simulate_outcome(action, entry, sl_dist, tp_dist, candles, pair):
    pip = 0.001 if "JPY" in pair else 0.0001
    tp_price = entry + tp_dist if action == "BUY" else entry - tp_dist
    sl_price = entry - sl_dist if action == "BUY" else entry + sl_dist

    for i, c in enumerate(candles):
        h, l, o = c["high"], c["low"], c["open"]

        if action == "BUY":
            sl_hit = l <= sl_price
            tp_hit = h >= tp_price
        else:
            sl_hit = h >= sl_price
            tp_hit = l <= tp_price

        if sl_hit and tp_hit:
            # ── Ambiguous candle: use open-proximity to resolve ──
            dist_to_sl = abs(o - sl_price)
            dist_to_tp = abs(o - tp_price)
            sl_first = dist_to_sl < dist_to_tp

            outcome = "STOP_LOSS" if sl_first else "TAKE_PROFIT"
            exit_px = sl_price if sl_first else tp_price
            pnl = -sl_dist if sl_first else tp_dist

            return {
                "outcome": outcome,
                "exit_price": round(exit_px, 5),
                "exit_candle": i + 1,
                "pnl_pips": round(pnl / pip, 1),
                "tp_price": round(tp_price, 5),
                "sl_price": round(sl_price, 5),
                "candles": candles,
                "ambiguous": True,           # ← flag it so the UI can warn
            }

        if sl_hit:
            return {
                "outcome": "STOP_LOSS",
                "exit_price": round(sl_price, 5),
                "exit_candle": i + 1,
                "pnl_pips": round(-sl_dist / pip, 1),
                "tp_price": round(tp_price, 5),
                "sl_price": round(sl_price, 5),
                "candles": candles,
                "ambiguous": False,
            }

        if tp_hit:
            return {
                "outcome": "TAKE_PROFIT",
                "exit_price": round(tp_price, 5),
                "exit_candle": i + 1,
                "pnl_pips": round(tp_dist / pip, 1),
                "tp_price": round(tp_price, 5),
                "sl_price": round(sl_price, 5),
                "candles": candles,
                "ambiguous": False,
            }

    exit_price = candles[-1]["close"] if candles else entry
    raw_pnl = (exit_price - entry) if action == "BUY" else (entry - exit_price)
    return {
        "outcome": "TIME_EXIT",
        "exit_price": round(exit_price, 5),
        "exit_candle": len(candles),
        "pnl_pips": round(raw_pnl / pip, 1),
        "tp_price": round(tp_price, 5),
        "sl_price": round(sl_price, 5),
        "candles": candles,
        "ambiguous": False,
    }

# ---------------------------------------------------------------------------
# SIMULATE TRADE
# ---------------------------------------------------------------------------

@app.route("/api/simulate-trade", methods=["POST"])
@limiter.limit("20 per minute")
def simulate_trade():
    try:
        body = request.get_json(force=True) or {}
        pair        = str(body.get("currency_pair", "EUR/USD")).upper()
        action      = str(body.get("action", "BUY")).upper()
        entry_price = float(body.get("entry_price") or 0)
        sl_distance = float(body.get("sl_distance") or 0)
        tp_distance = float(body.get("tp_distance") or 0)
        target_date = body.get("target_date") or \
                      datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # REPLACE WITH:
        if entry_price <= 0:
            return jsonify({"error": "entry_price must be positive"}), 400

        candles = get_next_candles(pair, target_date, n=5)
        if not candles:
            return jsonify({"error": f"No OHLCV data found after {target_date} for {pair}"}), 404

        # HOLD or missing TP/SL — time-exit only, no TP/SL levels
        if action == "HOLD" or sl_distance <= 0 or tp_distance <= 0:
            pip = 0.001 if "JPY" in pair else 0.0001
            exit_price = candles[-1]["close"]
            raw_pnl = exit_price - entry_price  # always long-perspective for HOLD
            return jsonify({
                "outcome": "TIME_EXIT",
                "action": action,
                "entry_price": round(entry_price, 5),
                "exit_price": round(exit_price, 5),
                "exit_candle": len(candles),
                "pnl_pips": round(raw_pnl / pip, 1),
                "tp_price": None,
                "sl_price": None,
                "candles": candles,
                "pair": pair,
                "ambiguous": False,
            })

        result = _simulate_outcome(action, entry_price, sl_distance, tp_distance, candles, pair)
        result["action"]      = action
        result["entry_price"] = round(entry_price, 5)
        result["pair"]        = pair
        return jsonify(result)
    except Exception as e:
        logger.error(traceback.format_exc())  # log server-side only
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)