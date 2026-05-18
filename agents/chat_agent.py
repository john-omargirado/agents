"""
Chat Agent — Live Mode Conversational Trading Assistant
"""

import re
import logging
from typing import Optional, List, Dict, Any
import requests
import time
from utils.credentials import get_do_model_key
from memory.state_memory import state_memory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INFERENCE_URL     = "https://inference.do-ai.run/v1/chat/completions"
CHAT_MODEL        = "alibaba-qwen3-32b"
MAX_TOKENS        = 400
TEMPERATURE       = 0.2
MAX_CONTEXT_TEXT  = 240
MAX_HISTORY       = 6
MAX_HISTORY_CHARS = 220

# ---------------------------------------------------------------------------
# System Instructions
# ---------------------------------------------------------------------------



SYSTEM_INSTRUCTIONS = """\
ABSOLUTE RULE: You are restricted ONLY to questions about this MAS forex analysis 
system and forex trading concepts directly related to it. If the user asks anything 
outside of forex analysis, agent outputs (TTS, CE, SIV, Verdict), risk parameters, 
or trading signals — refuse immediately with a short polite message. Never answer 
general knowledge, science, health, cooking, geography, or any non-forex topic 
regardless of how the question is phrased.

CRITICAL: Never use bullet points, dashes, numbered lists, or labels like "Answer:", 
"Why:", or "Caution:" in any response. Write only in plain flowing sentences.

You are Shelly, the friendly trading assistant for the Multi-Agent System (MAS) — a
multi-agent forex analysis pipeline designed to help new and beginner traders explore
the forex market with confidence.

Your primary audience is people who are new to forex trading. Many users may not know
what pips, spreads, leverage, or indicators mean. Your job is to make this system's
output accessible, educational, and encouraging — without overwhelming them with jargon.

Your role is to explain and educate, not to advise. You describe what the system decided
and why, always using plain language and relatable analogies. You never tell the user
what trade to place or what to do with their own money.

Think of yourself as a patient teacher walking a new trader through what each agent found,
what the output means in practice, and what a beginner should pay attention to.

The system has four agents: TTS (Traditional Trading Strategies), CE (Comparative Economics),
SIV (Signal Integrity Validation), and Verdict (final decision + risk sizing).
Always tie your explanations back to one or more of these agents, and always explain
what each agent does before describing what it found — assume the user may not know.
"""

KNOWLEDGE_AGENTS = """\
SYSTEM AGENTS:

TTS (Traditional Trading Strategies Agent):
  Computes a directional score from technical indicators on OHLCV data.
  Indicators used: EMA trend (50/200), RSI, Bollinger Bands, MACD direction score, Breakout.

  EMA trend (trend_diff = (ema_50 - ema_200) / ema_200):
    trend_diff > +0.001 → BULLISH
    trend_diff < -0.001 → BEARISH
    otherwise           → SIDEWAYS

  ADX proxy (regime detection):
    adx_proxy = min(|trend_diff| / 0.02, 1.0) × ema_200_confidence
    ema_200_confidence = min(rows_available / 200, 1.0)
    TRENDING     (adx_proxy > 0.40)
    RANGING      (adx_proxy < 0.15)
    TRANSITIONAL (0.15 – 0.40)

  Trend strength dampening (RSI divergence vs EMA):
    BEARISH trend + RSI > 70  → trend_strength ×0.2  (extreme divergence)
    BEARISH trend + RSI > 60  → trend_strength ×0.4  (moderate)
    BULLISH trend + RSI < 30  → trend_strength ×0.2
    BULLISH trend + RSI < 40  → trend_strength ×0.4

  RSI scoring (mean-reversion):
    RSI > 60 → rsi_score = -min((RSI-60)/40, 1.0)  (overbought → bearish)
    RSI < 40 → rsi_score = +min((40-RSI)/40, 1.0)  (oversold  → bullish)
    RSI 40–60→ 0.0 (neutral)

  BB scoring:
    OVERBOUGHT (close > upper_bb): bb_score = -min(bb_strength, 1.0)
    OVERSOLD   (close < lower_bb): bb_score = +min(bb_strength, 1.0)
    STABLE:                        bb_score = 0.0
    bb_strength = |close - band_edge| / band_width   (band_width = upper_bb - lower_bb)

  MACD direction score (based on MACD line vs signal line crossover):
    Bearish cross (prev above, curr below): macd_direction_score = -0.6
    Bullish cross (prev below, curr above): macd_direction_score = +0.6
    No cross:                               macd_direction_score = ±0.2 (direction-normalized)
    is_macd_cross = True when |macd_direction_score| >= 0.6

  Breakout (20-period rolling high/low, excluding current bar):
    close > breakout_high → BREAKOUT_UP,   strength = min((close - high) / range, 1.0)
    close < breakout_low  → BREAKOUT_DOWN, strength = min((low - close) / range, 1.0)
    otherwise             → NONE,          strength = 0.0
    trading_range = breakout_high - breakout_low

  Breakout weights by regime: TRENDING=0.25, TRANSITIONAL=0.15, RANGING=0.05
  remaining = 1.0 - breakout_weight

  Score weights (MACD cross active, |macd_direction_score| >= 0.6):
    MACD×(0.50×remaining) + RSI×(0.30×remaining) + BB×(0.10×remaining) + breakout×breakout_weight
  Score weights (no MACD cross):
    RSI×(0.45×remaining) + BB×(0.25×remaining) + MACD×(0.30×remaining) + breakout×breakout_weight

  Decision thresholds:
    BUY  when total_score > +0.15
    SELL when total_score < -0.15
    HOLD otherwise

  ATR: not computed in tts_tools. tts_agent derives ATR as a fallback:
    atr = |breakout_high - breakout_low| if both available, else 0.0
    Verdict clamps atr to minimum 0.0001.

  Outputs: decision, total_score (–1 to +1), ema_trend, rsi, bb_signal, bb_strength,
           macd_direction_score, is_macd_cross, breakout_signal, breakout_strength,
           breakout_high, breakout_low, regime, adx_proxy, ema_200_confidence,
           ema_200_reliable, data_stale, price, atr, explanation.

CE (Comparative Economic Agent):
  Fetches and scores news articles for the currency pair using FinBERT polarity.
  Outputs a continuous ce_score (–1 to +1) and a ce_confidence (0–1).

  Sentiment labels:
    BULLISH when ce_score > +0.05
    BEARISH when ce_score < -0.05
    NEUTRAL otherwise

  Confidence tiers:
    HIGH     when article_count >= 25
    MODERATE when article_count >= 20
    LOW      otherwise

  If article_count == 0, CE returns NEUTRAL with ce_score=0.0, ce_confidence=0.0,
  and explanation="no_data". CE contributes near-zero directional impact.
  In backtest_mode or skip_llm=True, explanation is set to "skipped".

SIV (Signal Integrity Validation Agent):
  Validates CE and TTS outputs by checking directional alignment and price integrity.

  First checks price: if actual_price or tts_price is missing, or they differ → INCOHERENT.

  Direction mapping:
    BULLISH/BUY  → UP
    BEARISH/SELL → DOWN
    NEUTRAL/HOLD → FLAT
    unknown      → UNKNOWN → PARTIAL (unrecognized_signal, multiplier=0.5)

  Outcomes and exact values:
    COHERENT                  score_multiplier=1.0,  risk_penalty=0.0
    PARTIAL (one_signal_neutral) score_multiplier=0.95, risk_penalty=0.2
    PARTIAL (signal_mismatch) score_multiplier=0.85, risk_penalty=0.5
    PARTIAL (unrecognized)    score_multiplier=0.5,  risk_penalty=0.0
    INCOHERENT                score_multiplier=0.0,  risk_penalty=1.0

  LLM explanation only runs in live_mode=True and skip_llm=False. Otherwise "skipped".

Verdict Agent:
  Reads ATR from state (written by TTS agent). Clamped to minimum 0.0001.

  Combines TTS and CE into a weighted_score:
    ce_weight  = 0.35 + (0.30 × ce_confidence)   →  ranges 0.35 to 0.65
    tts_weight = 1.0 – ce_weight
    weighted_score = (ce_weight × ce_score) + (tts_weight × tts_score)
  Then scaled: weighted_score × siv_score_multiplier, clamped to [–1, +1].

  HARD BLOCK: SIV==INCOHERENT → immediately HOLD, action=NONE (no further processing).

  Signal Quality Gate:
    ce_strong  = article_count >= 10 AND |ce_score| >= 0.05
    tts_strong = |tts_score| >= 0.08
    If SIV PARTIAL + signal_mismatch: tradeable = ce_strong only
    If not ce_strong AND not tts_strong: HOLD, action=SKIP
    Otherwise: proceed

  Decision thresholds:
    LLM mode (live_mode=True, skip_llm=False):
      BUY  if weighted_score >= +0.05
      SELL if weighted_score <= -0.05
      HOLD otherwise
    Deterministic mode (backtest_mode=True or skip_llm=True):
      Uses calibration_threshold (default 0.05) — same logic

  Dynamic SL/TP:
    atr_pct = atr / price
    vol_mult: < 0.002 → 1.6 | < 0.005 → 2.2 | else → 3.0
    trend_factor: EMA BULLISH or BEARISH → ×1.15 (TRENDING) | else → ×0.90 (RANGING)
    sl_distance = round(atr × sl_mult, 5)
    tp_distance = round(sl_distance × rr_ratio × vol_mult × trend_factor, 5)

  Position sizing:
    risk_amount   = capital × (risk_per_trade / 100)
    position_size = risk_amount / (sl_distance × 100_000)
    Capped at max_exposure / (price × 100_000),  max_exposure = capital × leverage

  risk_multiplier: |weighted_score| > 0.5 → 0.8 | > 0.2 → 0.6 | else → 0.4
    Special case: atr == 0.0 → 0.4

  Output keys (top-level AND inside trade_output):
    verdict, weighted_score, risk_multiplier, verdict_reasoning, action,
    sl_distance, tp_distance, atr, position_size, risk_amount, max_exposure.
    action = verdict if BUY/SELL, else NONE (or SKIP if quality gate failed).
"""

KNOWLEDGE_WORKFLOW = """\
WORKFLOW AND ENSEMBLE:

Flow: [TTS + CE run concurrently] → SIV validates → Verdict → Output

TTS reads OHLCV from local JSON files (data/backtesting/forex_pairs/{PAIR}.json).
Indicators are precomputed once per file load and cached.
CE fetches live or historical news articles and scores them with FinBERT.

Ensemble scoring:
  ce_weight  = 0.35 + (0.30 × ce_confidence)   →  0.35 to 0.65
  tts_weight = 1.0 – ce_weight
  weighted_score = (ce_weight × ce_score) + (tts_weight × tts_score)
  After SIV: weighted_score × score_multiplier, clamped to [–1, +1]

Signal quality gate (pre-verdict filter):
  ce_strong  = article_count >= 10 AND |ce_score| >= 0.05
  tts_strong = |tts_score| >= 0.08
  If neither strong → HOLD/SKIP
  If SIV partial+mismatch → needs ce_strong only (TTS unreliable)

Verdict thresholds (both LLM and deterministic default to 0.05):
  BUY  if weighted_score >= +0.05
  SELL if weighted_score <= -0.05
  HOLD otherwise

Risk parameters:
  ATR derived in tts_agent from breakout range; clamped ≥ 0.0001 in verdict_agent.
  sl_distance = ATR × sl_mult  (from pair config)
  tp_distance = sl_distance × rr_ratio × vol_mult × trend_factor
  position_size = risk_amount / (sl_distance × 100_000)
"""

GUARDRAILS = """\
Rules:
1) Stay in forex MAS analysis scope only. Do not answer general trading or finance questions
   unrelated to what this system produced.
2) If data is missing or marked as "skipped" / "no_data" / "pending", say so clearly
   and explain what that means — do not guess or fill in values.
3) Keep answers ≤150 words unless the user explicitly asks for more detail.
4) Never give personal BUY/SELL recommendations or tell the user what trade to take.
   You explain what the system produced — the user decides what to do with it.
5) Use measured, grounded wording: "based on current signals", "this run shows",
   "the system produced", "according to the analysis".
6) Avoid sarcasm, slang, and confrontational phrasing.
7) When explaining numbers, always add context — never drop a raw value without
   saying what it means in practice. e.g. don't just say "ATR=0.00312",
   say "ATR=0.00312, meaning recent daily price swings averaged about 31 pips."
8) If a verdict is HOLD or action is SKIP, explain the actual reason
   (weak signals, quality gate, SIV block) — do not just say "no trade was taken."
9) Spell out acronyms on first use in each response: SL (stop loss), TP (take profit),
   ATR (Average True Range), EMA (Exponential Moving Average), RSI (Relative Strength Index),
   BB (Bollinger Bands), MACD (Moving Average Convergence Divergence).
10) You are talking to beginner traders. Never assume prior knowledge. If you use a
    concept like "overbought", "trend", or "signal mismatch", immediately explain what
    it means in one plain sentence — as if the user has never heard of it.
11) When a user seems confused or asks a very basic question, respond with extra warmth
    and encouragement. Remind them that learning forex takes time and that the system
    is here to help them understand, not to pressure them into trades.
12) If the user asks what they should do with a signal (e.g. "should I trade this?"),
    remind them gently that this system produces analysis to help them learn and
    understand market conditions — final decisions always belong to them, and they
    should consider consulting a licensed financial advisor before trading real money.
"""

OUTPUT_STRUCTURE = """\
Format:
Write 2–4 short sentences in a natural, conversational tone — like a knowledgeable
friend explaining what just happened in the market. Do not use bullet points, labels
like "Answer:" or "Why:", or structured headers of any kind. Just talk.

Structure your response naturally in this order, without announcing the sections:
  1. Start with what the system found or decided — keep it plain and direct.
  2. Follow with why it happened — explain the key data point(s) that drove it,
     and what those numbers mean in everyday terms.
  3. If there is a meaningful risk, limitation, or caveat, weave it in at the end
     naturally — don't label it, just say it as part of the flow.

Tone and style rules:
- You are Shelly, a patient and encouraging guide for beginner forex traders.
  Your goal is not just to answer the question — it is to help the user understand
  the forex market a little better with every response.
- Write as if the person has never traded forex before. Use everyday analogies
  where possible (e.g. "think of leverage like borrowing money to invest —
  it amplifies both gains and losses").
- Never just list raw numbers alone — always say what they mean in plain terms.
  BAD:  "weighted_score=+0.23, ce_weight=0.50"
  GOOD: "The combined score came in at +0.23, leaning bullish — news carried a bit
         more weight here because there were enough articles to trust the sentiment."
- Spell out jargon on first use: e.g. "SL (stop loss)", "TP (take profit)",
  "ATR (a measure of how much price typically moves each day)".
- For SL/TP pips: say what it means in practice, e.g. "the stop loss is set 18 pips
  away, which means the trade would close automatically if price moved 18 pips against you."
- For HOLD/SKIP: explain the real reason in plain words — don't just say it was skipped.
  e.g. "The system decided to sit this one out because neither the news nor the technical
        side was strong enough to feel confident about a direction."
- For INCOHERENT: explain that the price check caught a mismatch, not just that SIV blocked it.
- For regime: explain what it means for this trade in simple terms, e.g. "the market is
  moving sideways right now, so the system leaned more on mean-reversion signals."
- For risk_multiplier: say what tier it represents in plain English, e.g. "the system
  is treating this as a medium-confidence signal."
- End responses involving a BUY or SELL signal with a brief, natural reminder that this
  is for learning purposes and is not financial advice — don't make it feel like a legal
  disclaimer, keep it warm and human.

NEVER respond like this:
- Answer: ...
- Why: ...
- Caution: ...

ALWAYS respond like this:
"The system came back with a Hold this run. Neither the news side nor the technical 
side had enough confidence to push in a clear direction, so the quality gate stepped in and blocked a trade. 
It is worth noting that the market is ranging right now, which already makes signals harder to read."

Do not use bullet points, numbered lists, or any labels in your response.
Keep it under 150 words unless the user explicitly asks for more detail.
"""


# ---------------------------------------------------------------------------
# Chat Agent
# ---------------------------------------------------------------------------

class LiveChatAgent:

    def __init__(self):
        pass

    def _call_llm(self, prompt: str) -> str:
        key     = get_do_model_key()
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"

        payload = {
            "model":       CHAT_MODEL,
            "messages":    [{"role": "user", "content": f"/no_think\n{prompt}"}],
            "max_tokens":  MAX_TOKENS,
            "temperature": TEMPERATURE,
        }

        backoff = 10
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = requests.post(
                    INFERENCE_URL, headers=headers, json=payload, timeout=90
                )

                if resp.status_code == 429:
                    print(f"[ChatAgent] Rate limited (attempt {attempt}). Waiting {backoff}s...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue

                if resp.status_code in (400, 401, 403):
                    print(f"[ChatAgent] Fatal HTTP {resp.status_code}: {resp.text}")
                    return "explanation_unavailable"

                if resp.status_code != 200:
                    print(f"[ChatAgent] HTTP {resp.status_code} on attempt {attempt} — retrying...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue

                result  = resp.json()
                message = result.get("choices", [{}])[0].get("message", {})
                content = message.get("content") or message.get("reasoning_content")

                if not content:
                    print(f"[ChatAgent] Empty content on attempt {attempt} — retrying...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue

                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                return content

            except Exception as e:
                print(f"[ChatAgent] Attempt {attempt}: {e} — retrying...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)

    # ── Helpers ─────────────────────────────────────────────────────────────


    def _clip(self, text: Any, max_len: int = MAX_CONTEXT_TEXT) -> str:
        s = str(text or "")
        return s if len(s) <= max_len else s[: max_len - 3] + "..."

    def _to_float(self, value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _fmt(self, value: Any, fmt: str = "+.3f") -> str:
        parsed = self._to_float(value)
        return f"{parsed:{fmt}}" if parsed is not None else "N/A"

    # ── State unpacker ──────────────────────────────────────────────────────
    # Handles THREE formats:
    #   1. Raw TradingState (from graph.invoke / _run_cache)
    #   2. Serialized API response (from _serialize_state — verdict is a dict,
    #      risk fields are under a "risk" sub-dict)
    #   3. state_memory fallback (partial dict reconstructed in /api/chat)

    def _unpack(self, state: Dict[str, Any]):
        tts = state.get("tts_output", {}) or {}
        ce  = state.get("ce_output",  {}) or {}
        siv = state.get("siv_output", {}) or {}

        # ── verdict ── may be a raw string OR a serialized dict ──────────────
        raw_verdict = state.get("verdict", {})
        if isinstance(raw_verdict, dict):
            verdict_str       = str(raw_verdict.get("decision", "N/A")).upper()
            action_str        = str(raw_verdict.get("action", "N/A")).upper()
            weighted_score    = raw_verdict.get("weighted_score", 0.0)
            verdict_reasoning = raw_verdict.get("verdict_reasoning", "")
        else:
            verdict_str       = str(raw_verdict or "N/A").upper()
            action_str        = str(state.get("action", "N/A")).upper()
            weighted_score    = state.get("weighted_score", 0.0)
            verdict_reasoning = state.get("verdict_reasoning", "")

        risk_sub  = state.get("risk", {}) or {}
        trade_sub = state.get("trade_output", {}) or {}
        sl_distance = (
            state.get("sl_distance")
            or risk_sub.get("sl_distance")
            or trade_sub.get("sl_distance")
        )
        tp_distance = (
            state.get("tp_distance")
            or risk_sub.get("tp_distance")
            or trade_sub.get("tp_distance")
        )
        atr = (
            state.get("atr", 0.0)
            or risk_sub.get("atr", 0.0)
            or trade_sub.get("atr", 0.0)
        )
        risk_mult = (
            state.get("risk_multiplier")
            or risk_sub.get("risk_multiplier")
            or trade_sub.get("risk_multiplier")
        )

        # ── verdict risk_parameters (from serialized verdict dict) ────────────
        rp = {}
        if isinstance(raw_verdict, dict):
            rp = raw_verdict.get("risk_parameters", {}) or {}
            if not sl_distance and rp.get("sl_pips"):
                pair  = str(state.get("currency_pair", "")).upper()
                pip   = 0.01 if "JPY" in pair else 0.0001
                sl_distance = float(rp["sl_pips"]) * pip if rp.get("sl_pips") else None
                tp_distance = float(rp["tp_pips"]) * pip if rp.get("tp_pips") else None

        # ── regime ── may be in TTS or top-level ─────────────────────────────
        regime = state.get("regime") or tts.get("regime", "N/A")

        verdict_fields = {
            "verdict":           verdict_str,
            "action":            action_str,
            "weighted_score":    weighted_score,
            "verdict_reasoning": verdict_reasoning,
            "atr":               atr,
            "sl_distance":       sl_distance,
            "tp_distance":       tp_distance,
            "risk_multiplier":   risk_mult,
            "regime":            regime,
            "risk_parameters":   rp,
        }
        return tts, ce, siv, verdict_fields

    # ── Grounded response ───────────────────────────────────────────────────

    def _try_grounded_response(
        self,
        message: str,
        state: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        if not state:
            return None

        msg = (message or "").lower()
        tts, ce, siv, vf = self._unpack(state)

        verdict      = vf["verdict"]
        w_score      = vf["weighted_score"]
        atr          = vf["atr"]
        sl_dist      = vf["sl_distance"]
        tp_dist      = vf["tp_distance"]
        risk_mult    = vf["risk_multiplier"]
        action       = vf["action"]

        tts_decision = str(tts.get("decision",  "N/A")).upper()
        tts_score    = tts.get("total_score",   0.0)
        tts_rsi      = tts.get("rsi",           None)
        tts_bb       = str(tts.get("bb_signal", "N/A")).upper()
        tts_ema      = str(tts.get("ema_trend", "N/A")).upper()
        tts_regime   = str(tts.get("regime",    "N/A")).upper()
        tts_breakout = str(tts.get("breakout_signal", "NONE")).upper()
        tts_macd     = tts.get("macd_direction_score", 0.0)
        tts_macd_x   = bool(tts.get("is_macd_cross", False))
        tts_price    = tts.get("price", None)

        ce_sentiment = str(ce.get("sentiment",    "N/A")).upper()
        ce_score     = ce.get("ce_score",          0.0)
        ce_conf      = ce.get("ce_confidence",     0.0)
        ce_articles  = ce.get("article_count",     0)
        ce_tier      = str(ce.get("confidence",   "N/A")).upper()
        ce_no_data   = (ce_articles == 0 or ce.get("explanation") == "no_data")

        siv_signal   = str(siv.get("signal",        "N/A")).upper()
        siv_mult     = siv.get("score_multiplier",  1.0)
        siv_penalty  = siv.get("risk_penalty",      0.0)
        siv_issues   = siv.get("issues",            []) or []

        ce_weight_calc  = round(0.35 + 0.30 * float(ce_conf), 3)
        tts_weight_calc = round(1.0 - ce_weight_calc, 3)

        why_tokens = ["why", "explain", "reason", "behind", "how come", "cause"]
        pip_tokens = ["pip", "pips", "sl", "tp", "stop loss", "take profit", "atr", "distance", "risk"]
        is_why     = any(t in msg for t in why_tokens)
        is_pip     = any(t in msg for t in pip_tokens)

        if not is_why and not is_pip:
            return None

        mentions_tts     = any(t in msg for t in ["tts", "technical", "rsi", "bollinger", "bb", "macd", "ema", "breakout", "regime", "strategy"])
        mentions_ce      = any(t in msg for t in ["ce", "fundamental", "news", "sentiment", "finbert", "article", "economic", "calendar", "bullish", "bearish"])
        mentions_siv     = any(t in msg for t in ["siv", "coherent", "incoherent", "partial", "validation", "multiplier", "mismatch"])
        mentions_verdict = any(t in msg for t in ["verdict", "final", "decision", "ensemble", "weighted", "weighted_score"])

        requested = None
        for tok in ["buy", "sell", "hold", "bullish", "bearish", "neutral",
                    "coherent", "partial", "incoherent"]:
            if tok in msg:
                requested = tok.upper()
                break

        # ── TTS ──────────────────────────────────────────────────────────────
        if mentions_tts:
            if requested in {"BUY", "SELL", "HOLD"} and requested != tts_decision:
                return (
                    f"The technical side (TTS) actually isn't showing {requested} this run — "
                    f"it came back as {tts_decision} with a score of {self._fmt(tts_score)}, "
                    f"while the final verdict is {verdict}."
                )

            indicator_parts = []
            if tts_rsi is not None:
                indicator_parts.append(f"RSI at {tts_rsi:.1f}")
            indicator_parts.append(f"Bollinger Bands (BB) showing {tts_bb}")
            if tts_macd_x:
                indicator_parts.append(f"a MACD crossover (score {self._fmt(tts_macd, '.2f')})")
            else:
                indicator_parts.append(f"MACD score of {self._fmt(tts_macd, '.2f')}")
            if tts_breakout != "NONE":
                indicator_parts.append(f"a {tts_breakout.lower()} breakout")
            indicator_parts.append(f"regime is {tts_regime.lower()}")
            indicator_parts.append(f"EMA trend is {tts_ema.lower()}")

            base = (
                f"The technical analysis agent (TTS) came back {tts_decision} with an overall "
                f"score of {self._fmt(tts_score)}, driven mainly by {', '.join(indicator_parts)}."
            )

            if verdict != tts_decision:
                base += (
                    f" Keep in mind the final verdict is {verdict} — that's because the news "
                    f"agent (CE) was {ce_sentiment} and the combined weighted score landed at "
                    f"{self._fmt(w_score)} after the integrity check (SIV was {siv_signal})."
                )

            return base

        # ── CE ───────────────────────────────────────────────────────────────
        if mentions_ce:
            ce_dir_map = {"SELL": "BEARISH", "BUY": "BULLISH", "HOLD": "NEUTRAL",
                          "BULLISH": "BULLISH", "BEARISH": "BEARISH", "NEUTRAL": "NEUTRAL"}
            requested_ce = ce_dir_map.get(requested or "", None)

            if requested_ce and requested_ce != ce_sentiment:
                return (
                    f"The news sentiment agent (CE) isn't reading {requested} this run — "
                    f"it came back {ce_sentiment} with a score of {self._fmt(ce_score)}, "
                    f"based on {ce_articles} article(s) at {ce_tier.lower()} confidence."
                )

            base = (
                f"The news agent (CE) read the market as {ce_sentiment} — it scored "
                f"{self._fmt(ce_score)} after scanning {ce_articles} article(s), "
                f"which puts confidence at the {ce_tier.lower()} tier."
            )

            if ce_no_data:
                base += (
                    " Since no articles were found, the news side had almost no directional "
                    "impact and the verdict leaned more on technical signals."
                )
            elif ce_tier == "LOW":
                base += (
                    f" With only {ce_articles} articles, news carried a smaller weight "
                    f"({ce_weight_calc * 100:.0f}%) in the final score — more data would "
                    "increase its influence."
                )

            return base

        # ── SIV ──────────────────────────────────────────────────────────────
        if mentions_siv:
            if requested in {"COHERENT", "PARTIAL", "INCOHERENT"} and requested != siv_signal:
                return (
                    f"The integrity check (SIV) isn't {requested} this run — it came back "
                    f"{siv_signal} with a score multiplier of {siv_mult} "
                    f"and issues: {siv_issues or 'none'}."
                )

            ce_dir  = "up" if ce_sentiment == "BULLISH" else "down" if ce_sentiment == "BEARISH" else "flat"
            tts_dir = "up" if tts_decision == "BUY" else "down" if tts_decision == "SELL" else "flat"
            issue_str = ", ".join(siv_issues) if siv_issues else "none flagged"

            base = (
                f"The signal integrity check (SIV) — think of it as a quality-control "
                f"step that makes sure the news and technical agents are telling the same "
                f"story — came back {siv_signal}. News was pointing {ce_dir} and technicals "
                f"were pointing {tts_dir}, issues: {issue_str}. That gave a score multiplier "
                f"of {siv_mult}."
            )

            if siv_signal == "INCOHERENT":
                base += (
                    " Because the check failed completely, the weighted score was zeroed out "
                    "and the system automatically held — it won't trade on conflicting data."
                )
            elif siv_issues:
                base += f" The reduced multiplier ({siv_mult}) slightly dampened the final score."

            return base

        # ── Verdict / ensemble ────────────────────────────────────────────────
        if mentions_verdict or (is_why and any(k in msg for k in ["buy", "sell", "hold"])):
            base = (
                f"The system landed on {verdict} after combining the technical score "
                f"({tts_decision} at {self._fmt(tts_score)}) with news sentiment "
                f"({ce_sentiment} at {self._fmt(ce_score)}). News carried "
                f"{ce_weight_calc * 100:.0f}% of the weight because confidence was "
                f"{ce_tier.lower()}, and after the integrity check (SIV was {siv_signal} "
                f"with a ×{siv_mult} multiplier), the combined score came out at "
                f"{self._fmt(w_score)}."
            )

            if siv_signal == "INCOHERENT":
                base += (
                    " The INCOHERENT result zeroed out that score entirely and forced a HOLD "
                    "— the system won't act when data consistency can't be verified."
                )
            elif siv_signal == "PARTIAL":
                base += f" The PARTIAL result reduced confidence slightly (multiplier={siv_mult})."
            if ce_no_data:
                base += " Since no news articles were found, the decision was driven almost entirely by technicals."

            return base

        # ── Pip / SL / TP / ATR ───────────────────────────────────────────────
        if is_pip:
            price    = self._to_float(tts_price)
            pair_str = str(state.get("currency_pair", "")).upper()
            is_jpy   = "JPY" in pair_str
            pip_mult = 100 if is_jpy else 100000

            rp = vf.get("risk_parameters", {}) or {}
            sl_pips_raw = rp.get("sl_pips")

            lot_size = self._to_float(rp.get("lot_size"))
            if lot_size is not None and lot_size < 0.01:
                return (
                    f"The system calculated a position size of {lot_size:.4f} lots for this run, "
                    f"but the minimum tradeable lot size is 0.01 — so this trade can't be placed "
                    f"as-is. To fix this, you can either increase your capital (more funds means "
                    f"a larger position can be sized) or increase your leverage (which amplifies "
                    f"your buying power, though it also increases risk). Think of leverage like a "
                    f"loan from your broker — it lets you control a bigger position with less "
                    f"capital, but losses are amplified the same way gains are, so use it carefully."
                )

            tp_pips_raw = rp.get("tp_pips")

            sl_pips = (
                float(sl_pips_raw) if sl_pips_raw is not None
                else round(float(sl_dist) * pip_mult, 1) if sl_dist is not None and self._to_float(sl_dist) is not None
                else None
            )
            tp_pips = (
                float(tp_pips_raw) if tp_pips_raw is not None
                else round(float(tp_dist) * pip_mult, 1) if tp_dist is not None and self._to_float(tp_dist) is not None
                else None
            )

            if tp_pips is not None and sl_pips is not None:
                rr_ratio = round(tp_pips / sl_pips, 2) if sl_pips else "N/A"
                conf_label = (
                    "high" if risk_mult == 0.8
                    else "medium" if risk_mult == 0.6
                    else "lower"
                )
                return (
                    f"For this {pair_str} run the stop loss (SL) is {sl_pips} pips and the "
                    f"take profit (TP) is {tp_pips} pips, giving a risk/reward ratio of about "
                    f"{rr_ratio}:1. That means for every pip risked, the system targets roughly "
                    f"{rr_ratio} pips of gain. The TP distance is wider than the SL because the "
                    f"ATR (how much the pair typically moves each day) and the current "
                    f"{'trending' if tts_ema in ['BULLISH','BEARISH'] else 'ranging'} market "
                    f"conditions both influenced the calculation. The system rated this a "
                    f"{conf_label}-confidence signal overall."
                )
            elif verdict in ("HOLD",) or action in ("HOLD", "SKIP", "NONE"):
                reasoning = str(vf.get("verdict_reasoning", "") or "")

                if "INCOHERENT" in reasoning:
                    skip_plain = (
                        "the system's price integrity check (SIV) caught a mismatch between "
                        "the price the news agent saw and the chart price, so it blocked the "
                        "trade entirely to avoid acting on inconsistent data."
                    )
                elif "signal_mismatch" in reasoning:
                    skip_plain = (
                        f"the news agent (CE) and the technical agent (TTS) were pointing in "
                        f"opposite directions — CE was {ce_sentiment} while TTS said {tts_decision}. "
                        f"To trade through a disagreement like that, the system needs strong news "
                        f"confirmation (10+ articles with a clear score), and this run only had "
                        f"{ce_articles} article(s) with a ce_score of {abs(float(ce_score or 0)):.3f}."
                    )
                elif "weak signals" in reasoning:
                    skip_plain = (
                        f"neither the technical score ({self._fmt(tts_score)}) nor the news score "
                        f"({self._fmt(ce_score)}) was strong enough to act on — the system needs "
                        f"at least one of them to clear a minimum threshold before it'll size a trade."
                    )
                else:
                    skip_plain = (
                        f"the combined score ({self._fmt(w_score)}) didn't reach the ±0.05 "
                        "threshold needed to commit to a direction — it's a 'wait and see' result."
                    )

                return (
                    f"No SL (stop loss) or TP (take profit) was calculated this run because "
                    f"the system didn't take a trade — {skip_plain} Those distances are only "
                    f"worked out once a signal clears all the quality checks."
                )
            else:
                return (
                    "The SL and TP distances aren't available in the current context — "
                    "they may not have been computed yet or the verdict agent output is missing "
                    "those fields. Try running a fresh analysis to get updated values."
                )

        return None

    # ── Grounded guard ──────────────────────────────────────────────────────

    def _should_use_grounded_response(self, message: str, intent: str) -> bool:
        msg = (message or "").lower()

        why_tokens = ["why", "explain", "reason", "how come", "cause", "behind"]
        has_why    = any(t in msg for t in why_tokens)

        agent_intents = {"tts", "ce", "siv", "signal"}
        if intent in agent_intents and has_why:
            return True

        if intent == "risk":
            pip_tokens = ["pip", "pips", "atr", "tp", "take profit", "sl", "stop loss",
                        "rr", "risk/reward", "risk reward", "ratio", "lot", "distance"]
            return any(p in msg for p in pip_tokens)   # removed the calc-word gate

        return False

    # ── Intent detection ────────────────────────────────────────────────────

    def detect_intent(self, message: str) -> str:
        msg = (message or "").lower()
        if any(k in msg for k in ["simulate", "simulation", "what happened", "outcome", "take profit", "stop loss hit", "candle", "replay","win", "loss", "pips", "result"]):
            return "simulation"
        if any(k in msg for k in ["stop loss", "take profit", "risk", "rr", "lot", "leverage", "atr", "pip", "pips", "sl_distance", "tp_distance"]):
            return "risk"
        if any(k in msg for k in ["siv", "coherent", "incoherent", "partial", "multiplier", "mismatch", "validation", "audit", "price mismatch"]):
            return "siv"
        if any(k in msg for k in ["ce", "sentiment", "fundamental", "finbert", "article", "economic", "news", "calendar", "event", "bullish", "bearish", "ce_score", "ce_confidence", "no_data", "no news"]):
            return "ce"
        if any(k in msg for k in ["tts", "rsi", "macd", "breakout", "bollinger", "bb", "ema", "moving average", "strategy", "regime", "trending", "ranging", "transitional", "technical", "adx"]):
            return "tts"
        if any(k in msg for k in ["why", "signal", "buy", "sell", "hold", "decision", "verdict", "confidence", "ensemble", "weighted", "weighted_score"]):
            return "signal"
        if any(k in msg for k in ["how it works", "pipeline", "flow", "process", "workflow", "system", "what is", "explain", "orchestrat"]):
            return "process"
        return "general"

    def _knowledge_for_intent(self, intent: str) -> str:
        if intent in {"tts", "ce", "siv"}:
            return KNOWLEDGE_AGENTS
        return KNOWLEDGE_WORKFLOW + "\n\n" + KNOWLEDGE_AGENTS

    # ── Context filter ──────────────────────────────────────────────────────

    def filter_state(self, state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not state:
            return None

        tts, ce, siv, vf = self._unpack(state)

        return {
            "currency_pair":     state.get("currency_pair", "N/A"),
            "target_date":       state.get("target_date",   "N/A"),
            "atr":               vf["atr"],
            "regime":            vf["regime"],
            "verdict":           vf["verdict"],
            "action":            vf["action"],
            "weighted_score":    vf["weighted_score"],
            "verdict_reasoning": vf["verdict_reasoning"],
            "sl_distance":       vf["sl_distance"],
            "tp_distance":       vf["tp_distance"],
            "risk_multiplier":   vf["risk_multiplier"],
            "risk_parameters":   vf["risk_parameters"],
            "tts_output": {
                "decision":             tts.get("decision",             "N/A"),
                "total_score":          tts.get("total_score",          None),
                "price":                tts.get("price",                None),
                "ema_trend":            tts.get("ema_trend",            "N/A"),
                "rsi":                  tts.get("rsi",                  None),
                "bb_signal":            tts.get("bb_signal",            "N/A"),
                "macd_direction_score": tts.get("macd_direction_score", None),
                "is_macd_cross":        tts.get("is_macd_cross",        False),
                "breakout_signal":      tts.get("breakout_signal",      "NONE"),
                "breakout_strength":    tts.get("breakout_strength",    0.0),
                "regime":               tts.get("regime",               "N/A"),
                "ema_200_confidence":   tts.get("ema_200_confidence",   None),
                "ema_200_reliable":     tts.get("ema_200_reliable",     None),
                "data_stale":           tts.get("data_stale",           False),
                "explanation":          tts.get("explanation",          ""),
            },
            "ce_output": {
                "sentiment":     ce.get("sentiment",    "N/A"),
                "ce_score":      ce.get("ce_score",      None),
                "ce_confidence": ce.get("ce_confidence", None),
                "article_count": ce.get("article_count", 0),
                "confidence":    ce.get("confidence",   "N/A"),
                "raw_vibe":      ce.get("raw_vibe",     "N/A"),
                "explanation":   ce.get("explanation",  ""),
            },
            "siv_output": {
                "signal":           siv.get("signal",           "N/A"),
                "score_multiplier": siv.get("score_multiplier", None),
                "risk_penalty":     siv.get("risk_penalty",     None),
                "issues":           siv.get("issues",           []),
                "conflict_type":    siv.get("conflict_type",    ""),
                "explanation":      siv.get("explanation",      ""),
            },
        }

    # ── Context builder ─────────────────────────────────────────────────────

    def build_context_block(
        self,
        state: Optional[Dict[str, Any]],
        intent: str = "general",
        experience_level: Optional[str] = None,
        sim_result:       Optional[Dict[str, Any]] = None,

    ) -> str:
        if not state:
            return "\n[No analysis has been run yet.]\n"

        tts = state.get("tts_output", {}) or {}
        ce  = state.get("ce_output",  {}) or {}
        siv = state.get("siv_output", {}) or {}

        ce_weight_display  = round(0.35 + 0.30 * float(ce.get("ce_confidence") or 0), 3)
        tts_weight_display = round(1.0 - ce_weight_display, 3)

        pair_str = str(state.get("currency_pair", "N/A"))
        is_jpy   = "JPY" in pair_str.upper()
        pip_mult = 100 if is_jpy else 100000
        price    = self._to_float(tts.get("price"))

        rp      = state.get("risk_parameters", {}) or {}
        sl_dist = self._to_float(state.get("sl_distance"))
        tp_dist = self._to_float(state.get("tp_distance"))

        sl_pips = float(rp["sl_pips"]) if rp.get("sl_pips") is not None else (round(sl_dist * pip_mult, 1) if sl_dist else None)
        tp_pips = float(rp["tp_pips"]) if rp.get("tp_pips") is not None else (round(tp_dist * pip_mult, 1) if tp_dist else None)

        lines = ["--- LIVE ANALYSIS CONTEXT ---"]
        lines.append(f"Intent Focus: {intent}")

        if experience_level:
            level_label = {
                "beginner":     "Complete beginner — no prior trading knowledge. Use maximum analogies, spell out everything.",
                "basic":        "Basic familiarity — knows pips and charts. Explain concepts but skip the most elementary definitions.",
                "intermediate": "Some trading experience — just learning this system. Be concise, skip hand-holding.",
            }.get(experience_level, "Unknown")
            lines.append(f"User Experience Level: {level_label}")

        lines.append(f"Currency Pair: {pair_str}")

        verdict = state.get("verdict", "N/A")
        action  = state.get("action",  "N/A")
        w_score = state.get("weighted_score", "N/A")
        lines.append(f"Verdict: {verdict} (action={action}, weighted_score={self._fmt(w_score)})")
        if state.get("verdict_reasoning"):
            lines.append(f"Reasoning: {self._clip(state.get('verdict_reasoning'), 200)}")

        atr = state.get("atr", None)
        if atr:
            lines.append(f"ATR: {self._fmt(atr, '.5f')}")
        if sl_dist is not None:
            sl_line = f"SL Distance: {sl_dist:.5f}"
            if sl_pips:
                sl_line += f" ({sl_pips} pips)"
            lines.append(sl_line)
        if tp_dist is not None:
            tp_line = f"TP Distance: {tp_dist:.5f}"
            if tp_pips:
                tp_line += f" ({tp_pips} pips)"
            lines.append(tp_line)
        if state.get("risk_multiplier") is not None:
            lines.append(f"Risk Multiplier: {state.get('risk_multiplier')}")
        if rp.get("lot_size") is not None:
            lot_size = self._to_float(rp.get("lot_size"))
            lot_line = f"Lot Size: {rp['lot_size']}"
            if lot_size is not None and lot_size < 0.01:
                lot_line += " [BELOW MINIMUM — 0.01 lot required; advise user to increase capital or leverage]"
            lines.append(lot_line)

        if tts:
            tts_line = (
                f"TTS: {tts.get('decision','N/A')} | "
                f"score={self._fmt(tts.get('total_score'), '.4f')}"
            )
            if price:
                tts_line += f" | price={price:.5f}"
            lines.append(tts_line)
            rsi    = tts.get("rsi")
            bb     = tts.get("bb_signal", "N/A")
            ema    = tts.get("ema_trend",  "N/A")
            reg    = tts.get("regime",     "N/A")
            bk     = tts.get("breakout_signal", "NONE")
            macd_x = "yes" if tts.get("is_macd_cross") else "no"
            lines.append(
                f"TTS Indicators: RSI={self._fmt(rsi, '.1f') if rsi else 'N/A'} | "
                f"BB={bb} | EMA={ema} | MACD-cross={macd_x} | breakout={bk} | regime={reg}"
            )
            if tts.get("data_stale"):
                lines.append("TTS Note: data_stale=True — OHLCV data may be outdated")
            if tts.get("explanation") and tts.get("explanation") not in ("skipped", "pending", ""):
                lines.append(f"TTS Explanation: {self._clip(tts.get('explanation'), 200)}")

        if ce:
            no_data = (int(ce.get("article_count") or 0) == 0 or ce.get("explanation") == "no_data")
            lines.append(
                f"CE: {ce.get('sentiment','N/A')} | "
                f"ce_score={self._fmt(ce.get('ce_score'))} | "
                f"ce_confidence={self._fmt(ce.get('ce_confidence'), '.3f')} | "
                f"articles={ce.get('article_count', 0)} ({ce.get('confidence', 'N/A')})"
            )
            lines.append(f"CE Ensemble Weight: {ce_weight_display} (TTS weight={tts_weight_display})")
            if no_data:
                lines.append("CE Note: no articles found — near-zero directional impact")
            if ce.get("explanation") and ce.get("explanation") not in ("skipped", "pending", "no_data", ""):
                lines.append(f"CE Explanation: {self._clip(ce.get('explanation'), 200)}")

        if siv:
            issue_str = ", ".join(siv.get("issues") or []) or "none"
            lines.append(
                f"SIV: {siv.get('signal','N/A')} | "
                f"score_multiplier={siv.get('score_multiplier','N/A')} | "
                f"risk_penalty={siv.get('risk_penalty','N/A')} | "
                f"issues=[{issue_str}]"
            )
            if siv.get("explanation") and siv.get("explanation") not in ("skipped", "pending", ""):
                lines.append(f"SIV Explanation: {self._clip(siv.get('explanation'), 200)}")

        lines.append("--- END CONTEXT ---")
        return "\n".join(lines)

    # ── Prompt builder ──────────────────────────────────────────────────────

    def build_prompt(
        self,
        message:          str,
        state:            Optional[Dict[str, Any]],
        history:          Optional[List[Dict[str, str]]] = None,
        intent:           str = "general",
        memory_block:     str = "",
        experience_level: Optional[str] = None,
        sim_result:       Optional[Dict[str, Any]] = None,
    ) -> str:
        context_block = self.build_context_block(state, intent=intent, experience_level=experience_level,sim_result=sim_result,   )
        knowledge_block = self._knowledge_for_intent(intent)

        history_text = ""
        if history:
            for msg in history[-MAX_HISTORY:]:
                role    = msg.get("role", "user").capitalize()
                content = self._clip(msg.get("content", ""), MAX_HISTORY_CHARS)
                history_text += f"\n{role}: {content}"

        return (
            f"{SYSTEM_INSTRUCTIONS}\n"
            f"{knowledge_block}\n\n"
            f"{GUARDRAILS}\n"
            f"{OUTPUT_STRUCTURE}\n\n"
            f"{memory_block}\n\n"
            f"{context_block}\n"
            f"{history_text}\n"
            f"User: {self._clip(message, 260)}\n"
            f"Assistant:"
        )

    # ── Response cleaning ───────────────────────────────────────────────────

    def clean_response(self, raw: str) -> str:
        text = raw.strip()

        # Strip thinking tags if present
        if "<think>" in text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # Strip "Assistant:" prefix if the model echoed it
        if text.lower().startswith("assistant:"):
            text = text[len("assistant:"):].strip()

        # Remove any lingering Answer/Why/Caution labels if the model slips back
        text = re.sub(
            r"^\s*-?\s*(Answer|Why|Caution)\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        # Collapse multiple blank lines into one
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        return text

    # ── Topic guard ─────────────────────────────────────────────────────────

    TRADING_KEYWORDS = {
        "buy", "sell", "hold", "signal", "confidence", "verdict", "analysis",
        "forex", "currency", "pair", "trade", "trading", "market", "price",
        "eur", "usd", "gbp", "jpy", "aud", "cad", "chf",
        "tts", "ce", "siv", "agent", "strategy", "strategies",
        "ema", "rsi", "macd", "breakout", "reversion", "bollinger", "moving average",
        "atr", "adx", "ma20", "ma50", "ema50", "ema200",
        "take profit", "stop loss", "risk", "reward", "lot", "leverage",
        "entry", "pip", "spread", "volatility", "position",
        "sentiment", "finbert", "news", "calendar", "event", "economic", "indicator",
        "inflation", "interest rate", "gdp", "employment", "monetary",
        "fundamental", "macroeconomic", "macro",
        "no news", "no_data", "article", "article_count", "ce_score", "ce_confidence",
        "weighted", "weighted_score", "ensemble",
        "bullish", "bearish", "neutral", "strong", "weak",
        "backtest", "historical", "lookback", "date",
        "validation", "coherence", "coherent", "incoherent", "partial", "multiplier",
        "regime", "trending", "ranging", "transitional",
        "score_multiplier", "risk_penalty", "risk_multiplier",
        "ohlcv", "candle", "chart", "support", "resistance", "trend",
        "recommend", "suggest", "explain", "why", "how", "what",
        "profit", "loss", "system", "pipeline", "workflow", "breakout_signal",
        "bb_signal", "ema_trend", "total_score", "tts_score", "action",
        "sl_distance", "tp_distance", "sl", "tp",
         "simpler", "simple", "explain", "clarify", "understand", "meaning",
        "output", "result", "decision", "your", "the", "it", "this",
        "sentiment", "verdict", "sell", "buy", "hold", "signal",
    }

    REFUSAL = (
        "I'm Shelly, and I can only help with questions about this system's forex analysis — "
        "things like signals, agent reasoning, risk parameters, and verdict explanations. "
        "That question is outside what I can help with here."
    )

    OFF_TOPIC_SIGNALS = {
    "bone", "anatomy", "biology", "recipe", "cook", "weather", "movie",
    "music", "sport", "history", "geography", "math", "physics", "chemistry",
    "animal", "planet", "country", "capital", "president", "celebrity",
    "game", "food", "medicine", "doctor", "hospital", "religion",
}

        
    def is_on_topic(self, message: str) -> bool:
        ml = (message or "").lower().strip()
        
        # Short follow-up messages (under 8 words) are almost always
        # continuations of the forex conversation — let them through
        word_count = len(ml.split())
        if word_count <= 8:
            # Only hard-block if it's clearly off-topic
            hard_off = {"recipe", "cook", "weather", "movie", "sport", "anatomy",
                        "biology", "animal", "planet", "president", "celebrity"}
            if not any(kw in ml for kw in hard_off):
                return True

        if any(kw in ml for kw in self.OFF_TOPIC_SIGNALS):
            # Don't block if it also contains trading keywords
            if any(kw in ml for kw in self.TRADING_KEYWORDS):
                return True
            return False

        return any(kw in ml for kw in self.TRADING_KEYWORDS)

    # ── Main entry ──────────────────────────────────────────────────────────

    def chat(
        self,
        message:          str,
        state:            Optional[Dict[str, Any]] = None,
        history:          Optional[List[Dict[str, str]]] = None,
        experience_level: Optional[str] = None,
        sim_result:       Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self.is_on_topic(message):
            logger.info("[LiveChatAgent] Off-topic: %s", message[:60])
            return self.REFUSAL

        intent = self.detect_intent(message)


        filtered     = self.filter_state(state)
        pair         = (state or {}).get("currency_pair", "")
        memory_block = state_memory.format_memory_block(pair)

        prompt = self.build_prompt(
            message, filtered, history,
            intent=intent,
            memory_block=memory_block,
            experience_level=experience_level,
            sim_result=sim_result,  
        )

        logger.info("[LiveChatAgent] intent=%s  message=%.60s", intent, message)
        raw      = self._call_llm(prompt)
        response = self.clean_response(raw)
        logger.info("[LiveChatAgent] Response: %.60s", response)
        return response


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
live_chat_agent = LiveChatAgent()