"""
MAS State Memory — Live Mode
Reads LangGraph SQLite checkpoints to give LiveChatAgent
awareness of all past live workflow runs.

Field mapping vs old version:
  OLD                              NEW (TradingState flat)
  ─────────────────────────────────────────────────────────
  state["verdict_output"]          state (verdict fields are top-level)
  verdict["verdict"]               state["verdict"]
  verdict["weighted_score"]        state["weighted_score"]
  verdict["entry_price"]           state["tts_output"]["price"]
  verdict["risk_parameters"]       state (atr/sl_distance/tp_distance flat)
  state["tts_output"]["signal"]    state["tts_output"]["decision"]
  state["tts_output"]["confidence_score"] → state["tts_output"]["total_score"]
  state["tts_output"]["strategy_signals"] → individual indicator fields
  state["ce_output"]["fundamental_sentiment"] → state["ce_output"]["sentiment"]
  state["ce_output"]["sentiment_score"]  → state["ce_output"]["ce_score"]
  state["siv_validation"]          state["siv_output"]
  siv["overall_coherence"]         siv["signal"]
  siv["overall_confidence"]        siv["score_multiplier"]
"""

import sqlite3
import json
import logging
from typing import Optional, List, Dict, Any

try:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
except Exception:
    JsonPlusSerializer = None

logger = logging.getLogger(__name__)


class MASStateMemory:

    def __init__(self, db_path: str = "mas_memory.db"):
        self.db_path = db_path
        self._serde  = JsonPlusSerializer() if JsonPlusSerializer is not None else None

    # ── DB helpers ──────────────────────────────────────────────────────────

    def _connect(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _table_columns(self, conn: sqlite3.Connection, table_name: str) -> List[str]:
        try:
            cursor = conn.execute(f"PRAGMA table_info({table_name})")
            return [str(row[1]) for row in cursor.fetchall()]
        except Exception:
            return []

    def _decode_checkpoint_blob(self, payload: Any) -> Dict[str, Any]:
        if payload is None:
            return {}
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str):
            try:
                return json.loads(payload)
            except Exception:
                return {}
        if isinstance(payload, memoryview):
            payload = payload.tobytes()
        if isinstance(payload, (bytes, bytearray)):
            raw = bytes(payload)
            if self._serde is not None:
                try:
                    parsed = self._serde.loads_typed(("msgpack", raw))
                    return parsed if isinstance(parsed, dict) else {}
                except Exception:
                    pass
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception:
                return {}
        return {}

    # ── Core read ───────────────────────────────────────────────────────────

    def get_all_runs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Read past runs from LangGraph checkpointer table."""
        try:
            conn    = self._connect()
            columns = self._table_columns(conn, "checkpoints")

            if "thread_id" not in columns or "checkpoint" not in columns:
                logger.warning("[Memory] checkpoints schema missing columns: %s", columns)
                conn.close()
                return []

            order_col = "created_at"
            if order_col not in columns:
                order_col = "checkpoint_id" if "checkpoint_id" in columns else "rowid"

            query = (
                f"SELECT thread_id, checkpoint, {order_col} as sort_key "
                f"FROM checkpoints ORDER BY {order_col} DESC LIMIT ?"
            )
            cursor = conn.execute(query, (limit,))

            runs = []
            for row in cursor.fetchall():
                thread_id  = row[0]
                checkpoint = self._decode_checkpoint_blob(row[1])
                sort_key   = row[2]
                state      = checkpoint.get("channel_values", {})
                created_at = checkpoint.get("ts") or str(sort_key)
                runs.append({
                    "thread_id":  thread_id,
                    "created_at": created_at,
                    "state":      state,
                })

            conn.close()
            return runs

        except Exception as e:
            logger.warning("[Memory] Could not read runs: %s", e)
            return []

    def get_runs_by_pair(
        self,
        currency_pair: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Filter runs whose thread_id starts with the pair key.
        Expects thread_ids like: USDJPY_2025-05-06 or EURUSD_<uuid>
        """
        pair_key = currency_pair.replace("/", "").upper()
        all_runs = self.get_all_runs(limit=200)
        return [
            r for r in all_runs
            if str(r.get("thread_id", "")).upper().startswith(pair_key)
        ][:limit]

    # ── Summarizer — maps NEW TradingState to readable facts ────────────────

    def summarize_run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maps a decoded TradingState (from channel_values) to a flat summary dict.

        TradingState layout (live mode):
          Top-level: verdict, weighted_score, action, atr, sl_distance,
                     tp_distance, risk_multiplier, regime, currency_pair,
                     target_date, retry_count
          Sub-dicts: tts_output, ce_output, siv_output
        """
        tts = state.get("tts_output", {}) or {}
        ce  = state.get("ce_output",  {}) or {}
        siv = state.get("siv_output", {}) or {}

        # ── TTS ────────────────────────────────────────────────────────────
        tts_decision = tts.get("decision",             "N/A")
        tts_score    = tts.get("total_score",           0.0)
        tts_price    = tts.get("price",                 None)
        tts_ema      = tts.get("ema_trend",            "N/A")
        tts_rsi      = tts.get("rsi",                   None)
        tts_bb       = tts.get("bb_signal",            "N/A")
        tts_macd_x   = bool(tts.get("is_macd_cross",  False))
        tts_breakout = tts.get("breakout_signal",      "NONE")
        tts_regime   = tts.get("regime",               "N/A")
        tts_stale    = bool(tts.get("data_stale",      False))

        # ── CE ─────────────────────────────────────────────────────────────
        ce_sentiment  = ce.get("sentiment",        "N/A")   # BULLISH/BEARISH/NEUTRAL
        ce_score_val  = ce.get("ce_score",          0.0)
        ce_conf       = ce.get("ce_confidence",     0.0)
        ce_articles   = int(ce.get("article_count") or 0)
        ce_tier       = ce.get("confidence",       "N/A")   # HIGH/MODERATE/LOW
        ce_no_data    = (ce_articles == 0 or ce.get("explanation") == "no_data")

        # Adaptive CE weight (mirrors verdict_agent formula)
        ce_weight  = round(0.35 + 0.30 * float(ce_conf or 0), 3)
        tts_weight = round(1.0 - ce_weight, 3)

        # ── SIV ────────────────────────────────────────────────────────────
        siv_signal = siv.get("signal",           "N/A")   # COHERENT/PARTIAL/INCOHERENT
        siv_mult   = siv.get("score_multiplier",  None)
        siv_pen    = siv.get("risk_penalty",       None)
        siv_issues = siv.get("issues",             []) or []

        # ── Verdict (flat on TradingState) ──────────────────────────────────
        verdict    = state.get("verdict",          "N/A")
        w_score    = state.get("weighted_score",    0.0)
        action     = state.get("action",           "N/A")
        atr        = state.get("atr",               None)
        sl_dist    = state.get("sl_distance",       None)
        tp_dist    = state.get("tp_distance",       None)
        risk_mult  = state.get("risk_multiplier",   None)
        regime     = state.get("regime") or tts_regime

        # ── Pip conversion ─────────────────────────────────────────────────
        pair_str = str(state.get("currency_pair", "")).upper()
        is_jpy   = "JPY" in pair_str
        pip_mult = 100 if is_jpy else 10000

        try:
            sl_pips = round(float(sl_dist) * pip_mult, 1) if sl_dist is not None else None
        except (TypeError, ValueError):
            sl_pips = None
        try:
            tp_pips = round(float(tp_dist) * pip_mult, 1) if tp_dist is not None else None
        except (TypeError, ValueError):
            tp_pips = None

        return {
            # Verdict
            "verdict":      verdict,
            "weighted":     w_score,
            "action":       action,
            "risk_mult":    risk_mult,
            # TTS
            "tts_decision": tts_decision,
            "tts_score":    tts_score,
            "tts_price":    tts_price,
            "tts_ema":      tts_ema,
            "tts_rsi":      tts_rsi,
            "tts_bb":       tts_bb,
            "tts_macd_x":   tts_macd_x,
            "tts_breakout": tts_breakout,
            "tts_regime":   tts_regime,
            "tts_stale":    tts_stale,
            # CE
            "ce_sentiment": ce_sentiment,
            "ce_score":     ce_score_val,
            "ce_conf":      ce_conf,
            "ce_articles":  ce_articles,
            "ce_tier":      ce_tier,
            "ce_no_data":   ce_no_data,
            "ce_weight":    ce_weight,
            "tts_weight":   tts_weight,
            # SIV
            "siv_signal":   siv_signal,
            "siv_mult":     siv_mult,
            "siv_penalty":  siv_pen,
            "siv_issues":   siv_issues,
            # Risk
            "atr":          atr,
            "sl_dist":      sl_dist,
            "tp_dist":      tp_dist,
            "sl_pips":      sl_pips,
            "tp_pips":      tp_pips,
            "regime":       regime,
            # Meta
            "pair":         pair_str,
            "target_date":  state.get("target_date",   "N/A"),
            "retries":      state.get("retry_count",    0),
        }

    # ── ChatAgent-ready outputs ─────────────────────────────────────────────

    def get_history_for_chat(
        self,
        currency_pair: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Last N run summaries for a pair, ready for prompt injection."""
        runs   = self.get_runs_by_pair(currency_pair, limit=limit)
        result = []
        for run in runs:
            summary               = self.summarize_run(run["state"])
            summary["thread_id"]  = run["thread_id"]
            summary["created_at"] = run["created_at"]
            result.append(summary)
        return result

    def detect_patterns(self, currency_pair: str) -> Dict[str, Any]:
        """Aggregate stats over recent runs — answers trend/pattern questions."""
        summaries = self.get_history_for_chat(currency_pair, limit=50)
        if not summaries:
            return {"total_runs": 0, "message": "No historical runs found"}

        total     = len(summaries)
        verdicts  = [s["verdict"]      for s in summaries]
        decisions = [s["tts_decision"] for s in summaries]
        ce_sents  = [s["ce_sentiment"] for s in summaries]
        siv_sigs  = [s["siv_signal"]   for s in summaries]
        regimes   = [s["regime"]       for s in summaries if s["regime"] != "N/A"]

        scores   = [s["weighted"]  for s in summaries if isinstance(s["weighted"],  (int, float))]
        atrs     = [s["atr"]       for s in summaries if s["atr"]  is not None]
        tp_pips  = [s["tp_pips"]   for s in summaries if s["tp_pips"] is not None]
        sl_pips  = [s["sl_pips"]   for s in summaries if s["sl_pips"] is not None]
        ce_confs = [s["ce_conf"]   for s in summaries if s["ce_conf"] is not None]
        articles = [s["ce_articles"] for s in summaries]

        from collections import Counter

        def pct(lst, key):
            return f"{lst.count(key) / total:.0%}" if total else "0%"

        return {
            "total_runs":    total,
            "verdict_dist":  {k: pct(verdicts,  k) for k in ("BUY", "SELL", "HOLD")},
            "tts_dist":      {k: pct(decisions, k) for k in ("BUY", "SELL", "HOLD")},
            "ce_dist":       {k: pct(ce_sents,  k) for k in ("BULLISH", "BEARISH", "NEUTRAL")},
            "siv_dist":      {k: pct(siv_sigs,  k) for k in ("COHERENT", "PARTIAL", "INCOHERENT")},
            "regime_dist":   dict(Counter(regimes)),
            "avg_score":     round(sum(scores)   / len(scores),   4) if scores   else 0.0,
            "avg_atr":       round(sum(atrs)     / len(atrs),     5) if atrs     else None,
            "avg_tp_pips":   round(sum(tp_pips)  / len(tp_pips),  1) if tp_pips  else None,
            "avg_sl_pips":   round(sum(sl_pips)  / len(sl_pips),  1) if sl_pips  else None,
            "avg_ce_conf":   round(sum(ce_confs) / len(ce_confs), 3) if ce_confs else None,
            "avg_articles":  round(sum(articles) / len(articles), 1) if articles else None,
            "most_recent":   summaries[0] if summaries else None,
        }

    def format_memory_block(self, currency_pair: str) -> str:
        """
        Compact text block ready to inject into LiveChatAgent prompt.
        Called by LiveChatAgent.build_prompt().
        """
        history  = self.get_history_for_chat(currency_pair, limit=5)
        patterns = self.detect_patterns(currency_pair)

        if not history:
            return f"[No historical runs in memory for {currency_pair} yet]"

        lines = [f"--- HISTORICAL MEMORY ({currency_pair}) ---"]

        # Aggregate summary
        lines.append(
            f"Runs: {patterns['total_runs']} | "
            f"Verdicts: {patterns['verdict_dist']} | "
            f"Avg weighted_score: {patterns.get('avg_score', 0):+.4f}"
        )
        lines.append(
            f"TTS decisions: {patterns['tts_dist']} | "
            f"CE sentiment: {patterns['ce_dist']}"
        )
        lines.append(
            f"SIV history: {patterns['siv_dist']} | "
            f"Regime mix: {patterns['regime_dist']}"
        )
        if patterns.get("avg_atr"):
            lines.append(
                f"Avg ATR: {patterns['avg_atr']:.5f} | "
                f"Avg TP: {patterns.get('avg_tp_pips','N/A')} pips | "
                f"Avg SL: {patterns.get('avg_sl_pips','N/A')} pips"
            )
        if patterns.get("avg_ce_conf") is not None:
            lines.append(
                f"Avg CE confidence: {patterns['avg_ce_conf']:.3f} | "
                f"Avg articles: {patterns.get('avg_articles','N/A')}"
            )

        lines.append("Recent runs (newest first):")
        for r in history:
            tp_str  = f"{r['tp_pips']}pips"      if r["tp_pips"]  is not None else "N/A"
            sl_str  = f"{r['sl_pips']}pips"      if r["sl_pips"]  is not None else "N/A"
            atr_str = f"{r['atr']:.5f}"           if r["atr"]      is not None else "N/A"
            mult_str= f"×{r['siv_mult']}"         if r["siv_mult"] is not None else ""
            rsi_str = f"RSI={r['tts_rsi']:.1f}"  if r["tts_rsi"]  is not None else ""
            bb_str  = f"BB={r['tts_bb']}"
            bk_str  = f"BK={r['tts_breakout']}"  if r["tts_breakout"] != "NONE" else ""

            indicator_parts = [x for x in [rsi_str, bb_str, bk_str] if x]
            indicator_str   = " ".join(indicator_parts)

            lines.append(
                f"  [{r.get('target_date','?')}] "
                f"Verdict={r['verdict']}({r['weighted']:+.4f}) action={r['action']} | "
                f"TTS={r['tts_decision']}({r['tts_score']:+.3f}) "
                f"CE={r['ce_sentiment']}({r['ce_score']:+.3f},art={r['ce_articles']}) "
                f"SIV={r['siv_signal']}{mult_str} | "
                f"{indicator_str} regime={r['regime']} | "
                f"TP={tp_str} SL={sl_str} ATR={atr_str} retries={r['retries']}"
            )

        lines.append("--- END MEMORY ---")
        return "\n".join(lines)


# Singleton — imported by LiveChatAgent
state_memory = MASStateMemory()