"""
explanation_pipeline.py

Reads the _raw.json produced by backtesting.py,
generates CE / TTS / SIV explanations concurrently,
then re-runs verdict reasoning using the real explanations as context,
then merges everything into a final _explained.csv.

Usage:
    python explanation_pipeline.py USDJPY -y 2023
    python explanation_pipeline.py USDJPY -y 2023,2024,2025
    python explanation_pipeline.py USDJPY -y 2023-2025
"""

import sys
import json
import time
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.credentials import get_do_model_key
from utils.trade_config import get_pair_config

URL = "https://inference.do-ai.run/v1/chat/completions"

# =========================
# HOW MANY ROWS IN PARALLEL
# Each row fires 4 LLM calls (CE + TTS + SIV + Verdict)
# =========================
MAX_ROW_WORKERS = 5


# =========================
# SHARED LLM CALLER
# =========================
def call_llm(prompt: str, max_tokens: int = 700, label: str = "") -> str:
    key = get_do_model_key()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    data = {
        "model": "alibaba-qwen3-32b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }

    max_retries = 3
    backoff = 5

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(URL, headers=headers, json=data, timeout=90)
            result = resp.json()
            message = result.get("choices", [{}])[0].get("message", {})
            content = message.get("content") or message.get("reasoning_content")
            return str(content).strip() if content else "explanation_unavailable"
        except Exception as e:
            print(f"[{label}] Attempt {attempt}/3 failed: {e}")
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2

    return "explanation_unavailable"


# =========================
# CE EXPLANATION
# =========================
def explain_ce(ce: dict) -> str:
    prompt = f"""/no_think
You are a sentiment analysis explanation engine.

Explain briefly:
- What the article sentiment distribution suggests
- Why confidence is {ce.get('confidence')}
- Whether the signal is reliable given article count
- Why the final sentiment is {ce.get('sentiment')}

Be concise and factual. No structured format.

INPUT:
{json.dumps(ce)}
"""
    return call_llm(prompt, max_tokens=700, label="CE EXPLANATION")


# =========================
# TTS EXPLANATION
# =========================
def explain_tts(tts: dict) -> str:
    prompt = f"""
You are a technical analysis explanation engine for the TTS Agent.

Explain briefly:
- Why the EMA/RSI/BB signals produced this score
- What the dominant signal was
- Any conflicting indicators
- Why the final decision is {tts.get('decision')}
- Note EMA 200 confidence and whether it affects reliability

Be concise and factual. No structured format.

INPUT:
{json.dumps(tts)}
"""
    return call_llm(prompt, max_tokens=700, label="TTS EXPLANATION")


# =========================
# SIV EXPLANATION
# =========================
def explain_siv(siv: dict, ce: dict, tts: dict) -> str:
    payload = {
        "siv_signal":       siv.get("signal"),
        "siv_issues":       siv.get("issues"),
        "ce_signal":        ce.get("sentiment"),
        "tts_signal":       tts.get("decision"),
        "ce_confidence":    ce.get("confidence"),
        "ce_article_count": ce.get("article_count"),
        "tts_score":        tts.get("total_score"),
    }
    prompt = f"""
You are a financial system integrity explanation engine.

Agent definitions:
- CE: analyzes macroeconomic news sentiment
- TTS: analyzes price action and technical indicators

Explain briefly:
- CE vs TTS signal alignment
- Why signals conflict or align
- Data quality issues if any

Be concise and factual. No structured format.

INPUT:
{json.dumps(payload)}
"""
    return call_llm(prompt, max_tokens=700, label="SIV EXPLANATION")


# =========================
# VERDICT REASONING
# Re-runs verdict reasoning using real explanations as context
# =========================
def explain_verdict(
    row: dict,
    ce_exp: str,
    tts_exp: str,
    siv_exp: str,
    pair: str,
) -> str:
    ce  = row.get("ce_output", {})
    tts = row.get("tts_output", {})
    siv = row.get("siv_output", {})

    weighted_score = row.get("weighted_score", 0.0)
    verdict        = row.get("verdict", "HOLD")
    atr            = float(row.get("atr", tts.get("atr", 0.0)))

    pair_cfg    = get_pair_config(pair)
    sl_mult     = float(pair_cfg.get("sl_mult", 1.0))
    rr_ratio    = float(pair_cfg.get("rr_ratio", 2.0))
    sl_distance = round(atr * sl_mult, 5)
    tp_distance = round(sl_distance * rr_ratio, 5)

    prompt = f"""
You are an expert forex trading decision engine.

OUTPUT FORMAT (STRICT):
First line: BUY or SELL or HOLD (one word only)
Second line onwards: your reasoning

YOUR TASK:
Make a trading decision for {pair} based on all available information below.

------------------------------------------------------------
DECISION FRAMEWORK (STRICT)
------------------------------------------------------------

1. WEIGHTED_SCORE IS THE ONLY DIRECTION RULE:

weighted_score = (0.6 * CE_signal) + (0.4 * TTS_signal)

- >= 0.15  → BUY
- <= -0.15 → SELL
- between -0.15 and 0.15 → HOLD (mandatory, no override)

The weighted_score already accounts for CE/TTS conflict mathematically.
Do NOT override BUY or SELL to HOLD because CE and TTS disagree directionally.
Signal conflict is already priced into the score.

------------------------------------------------------------
2. CONVICTION ADJUSTMENTS (DO NOT CHANGE VERDICT)
------------------------------------------------------------

These only affect reasoning strength, never the BUY/SELL/HOLD decision:

Reduce conviction if:
- CE confidence is LOW or article count < 10
- CE and TTS point in opposite directions
- SIV signal = PARTIAL with signal_mismatch

Increase conviction if:
- CE and TTS align directionally
- CE confidence HIGH or MODERATE with sufficient articles
- EMA_200_confidence >= 0.8

------------------------------------------------------------
3. REASONING REQUIREMENTS
------------------------------------------------------------

- State weighted_score and its zone
- Evaluate CE (confidence, article count, sentiment strength)
- Evaluate TTS (EMA, RSI, BB, MACD, EMA200 reliability)
- Evaluate SIV (alignment or mismatch — conviction impact only)
- Justify decision from weighted_score threshold only
- Always reference actual values

------------------------------------------------------------
INPUT DATA
------------------------------------------------------------

weighted_score: {round(weighted_score, 4)}
atr: {round(atr, 5)}
sl_distance: {sl_distance}
tp_distance: {tp_distance}

TTS:
decision: {tts.get("decision")}
total_score: {tts.get("total_score")}
ema_trend: {tts.get("ema_trend")}
rsi: {tts.get("rsi")}
bb_signal: {tts.get("bb_signal")}
ema_200_confidence: {tts.get("ema_200_confidence")}
ema_200_reliable: {tts.get("ema_200_reliable")}
data_stale: {tts.get("data_stale")}
tts_explanation: {tts.get("explanation", "none")}

CE:
sentiment: {ce.get("sentiment")}
confidence: {ce.get("confidence")}
articles: {ce.get("article_count")}
raw_vibe: {ce.get("raw_vibe")}
sentiment_score: {ce.get("sentiment_score")}
mean_score: {ce.get("mean_score")}
ce_explanation: {ce.get("explanation", "none")}

SIV:
signal: {siv.get("signal")}
issues: {siv.get("issues")}
siv_explanation: {siv.get("explanation", "none")}
"""


    return call_llm(prompt, max_tokens=1024, label="VERDICT REASONING")


# =========================
# PER-ROW EXPLANATION
# =========================
def explain_row(row: dict, pair: str) -> dict:
    date = row["date"]
    ce   = row.get("ce_output", {})
    tts  = row.get("tts_output", {})
    siv  = row.get("siv_output", {})

    print(f"[EXPLAIN] {date} | generating CE + TTS + SIV concurrently...")

    # Step 1 — CE, TTS, SIV explanations fire concurrently
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_ce  = ex.submit(explain_ce,  ce)
        f_tts = ex.submit(explain_tts, tts)
        f_siv = ex.submit(explain_siv, siv, ce, tts)

        ce_exp  = f_ce.result()
        tts_exp = f_tts.result()
        siv_exp = f_siv.result()

    # Step 2 — Verdict reasoning using real explanations as context
    print(f"[EXPLAIN] {date} | generating verdict reasoning with full context...")
    verdict_reasoning = explain_verdict(row, ce_exp, tts_exp, siv_exp, pair)

    print(f"[EXPLAIN] {date} | done")

    return {
        "date":              date,
        "CE_Explanation":    ce_exp,
        "TTS_Explanation":   tts_exp,
        "SIV_Explanation":   siv_exp,
        "Verdict_Reasoning": verdict_reasoning,
    }


# =========================
# MAIN PIPELINE
# =========================
def run_explanation_pipeline(target_pair: str, target_year: int):

    total_start = time.perf_counter()

    project_root = Path(__file__).resolve().parents[1]
    reports_dir  = project_root / "reports"

    raw_path = reports_dir /'raw'/f"backtest_{target_pair}_{target_year}_raw.json"
    csv_path = reports_dir / f"backtest_{target_pair}_{target_year}.csv"
    out_path = reports_dir /'final'/ f"backtest_{target_pair}_{target_year}_explained.csv"

    if not raw_path.exists():
        print(f"[ERROR] Raw file not found: {raw_path}")
        print("Run backtesting.py first.")
        return

    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    with open(raw_path, "r") as f:
        raw_rows = json.load(f)

    print(f"\n--- EXPLANATION PIPELINE: {target_pair} {target_year} ---")
    print(f"Rows to explain  : {len(raw_rows)}")
    print(f"Row parallelism  : {MAX_ROW_WORKERS} rows at a time")
    print(f"LLM calls per row: 4 (CE + TTS + SIV concurrent, then Verdict)\n")

    # =========================
    # GENERATE EXPLANATIONS IN PARALLEL
    # =========================
    explanations = {}

    with ThreadPoolExecutor(max_workers=MAX_ROW_WORKERS) as executor:
        futures = {
            executor.submit(explain_row, row, target_pair): row["date"]
            for row in raw_rows
        }
        for future in as_completed(futures):
            date = futures[future]
            try:
                result = future.result()
                explanations[result["date"]] = result
            except Exception as e:
                print(f"[EXPLAIN ERROR] {date}: {e}")
                explanations[date] = {
                    "date":              date,
                    "CE_Explanation":    "explanation_failed",
                    "TTS_Explanation":   "explanation_failed",
                    "SIV_Explanation":   "explanation_failed",
                    "Verdict_Reasoning": "explanation_failed",
                }

    # =========================
    # MERGE INTO CSV
    # =========================
    df = pd.read_csv(csv_path)

    for i, row in df.iterrows():
        date = str(row.get("Date", ""))
        if date in explanations:
            exp = explanations[date]
            df.at[i, "CE_Explanation"]    = exp["CE_Explanation"]
            df.at[i, "TTS_Explanation"]   = exp["TTS_Explanation"]
            df.at[i, "SIV_Explanation"]   = exp["SIV_Explanation"]
            df.at[i, "Verdict_Reasoning"] = exp["Verdict_Reasoning"]

    df.to_csv(out_path, index=False)

    elapsed = round(time.perf_counter() - total_start, 2)
    print(f"\n[DONE] Explanations merged.")
    print(f"Output : {out_path}")
    print(f"Time   : {elapsed}s\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pair", nargs="?", default="USDJPY")
    parser.add_argument("-y", "--years", default="2018")

    args = parser.parse_args()

    raw_years = args.years.strip()
    if "-" in raw_years and "," not in raw_years:
        start, end = raw_years.split("-")
        years = list(range(int(start), int(end) + 1))
    else:
        years = [int(y) for y in raw_years.split(",")]

    for year in years:
        print(f"\n{'='*50}")
        print(f"EXPLAINING: {args.pair.upper()} | YEAR={year}")
        print(f"{'='*50}")
        run_explanation_pipeline(args.pair.upper(), year)