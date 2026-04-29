"""
explanation_pipeline.py

Reads the _raw.json produced by backtesting.py,
generates CE / TTS / SIV explanations concurrently,
then explains the verdict reasoning using real explanations as context,
then merges everything into a final _explained.csv.

Usage:
    python explanation_pipeline.py USDJPY -y 2023
    python explanation_pipeline.py USDJPY -y 2023,2024,2025
    python explanation_pipeline.py USDJPY -y 2023-2025

Fixes applied:
    1. Semaphore is released BEFORE sleeping on rate-limit (was blocking
       all other threads during backoff while holding the slot).
    2. Row-level in-flight semaphore (_row_sem) limits how many rows are
       being processed simultaneously, preventing the initial burst of
       250 rows × 3 calls = 750 simultaneous requests.
    3. Removed the useless `time.sleep(1.0) every 5 rows` — it did
       nothing because _submit_row is non-blocking.
    4. Dedicated verdict semaphore (_verdict_sem) limits concurrent verdict
       calls to VERDICT_CONCURRENCY=2. Without this, all in-flight rows
       finish phase-1 near-simultaneously and fire N verdict calls at once,
       saturating the rate limit right at the most token-heavy step.
"""

import sys
import json
import threading
import time
import random
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.credentials import get_do_model_key
from utils.trade_config import get_pair_config

URL = "https://inference.do-ai.run/v1/chat/completions"

MAX_WORKERS       = 25  # thread pool size
LLM_CONCURRENCY   = 8   # max simultaneous in-flight LLM requests
MAX_INFLIGHT_ROWS = 4   # max rows being processed at once


VERDICT_CONCURRENCY = 1
_verdict_sem = threading.Semaphore(VERDICT_CONCURRENCY)

_llm_sem = threading.Semaphore(LLM_CONCURRENCY)
_row_sem = threading.Semaphore(MAX_INFLIGHT_ROWS)


class RateLimiter:
    def __init__(self, rate_per_sec: float):
        self.rate = rate_per_sec
        self.tokens = rate_per_sec
        self.last = time.time()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last
                self.last = now

                self.tokens += elapsed * self.rate
                if self.tokens > self.rate:
                    self.tokens = self.rate

                if self.tokens >= 1:
                    self.tokens -= 1
                    return

            time.sleep(0.01)

_rate_limiter = RateLimiter(rate_per_sec=2.5)


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

    attempt = 0
    backoff = 10
    max_backoff = 60

    while True:
        attempt += 1

        # FIX 1: Acquire semaphore only for the actual HTTP request.
        # All response data captured inside the block so no variable
        # is referenced outside it — fixes Pylance unbound warning.
        status   = None
        raw_text = ""
        body     = None  # parsed JSON, only populated on HTTP 200

        _rate_limiter.acquire()

        with _llm_sem:
            try:
                resp     = requests.post(URL, headers=headers, json=data, timeout=90)
                status   = resp.status_code
                raw_text = resp.text
                if status == 200:
                    body = resp.json()
            except Exception as e:
                # Network error — semaphore releases, retry below
                raw_text = str(e)

        # --- All retries/sleeps happen OUTSIDE the semaphore ---

        if status is None:
            print(f"[{label} ERROR] Attempt {attempt}: {raw_text} — retrying in {backoff}s...")
            time.sleep(backoff + random.uniform(0, 2))
            backoff = min(backoff * 2, max_backoff)
            continue

        if status == 429:
            print(f"[{label}] Rate limited on attempt {attempt}. Waiting {backoff}s...")
            time.sleep(backoff + random.uniform(0, 2))
            backoff = min(backoff * 2, max_backoff)
            continue

        if status in [400, 401, 403]:
            print(f"[{label} FATAL] HTTP {status}: {raw_text}")
            return "explanation_unavailable"

        if status != 200:
            print(f"[{label} ERROR] HTTP {status} on attempt {attempt} — retrying in {backoff}s...")
            time.sleep(backoff + random.uniform(0, 2))
            backoff = min(backoff * 2, max_backoff)
            continue

        result  = body or {}
        choice  = result.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content") or message.get("reasoning_content")

        if not content:
            print(f"[{label} ERROR] Empty content on attempt {attempt} — retrying in {backoff}s...")
            time.sleep(backoff + random.uniform(0, 2))
            backoff = min(backoff * 2, max_backoff)
            continue

        return str(content).strip()


# =========================
# COMBINED EXPLANATION (RETURNS CE + TTS + SIV USING ORIGINAL PROMPTS)
# =========================
def explain_combined(ce: dict, tts: dict, siv: dict) -> dict:

    ce_prompt = f"""/no_think
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

    tts_prompt = f"""
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

    payload = {
        "siv_signal":       siv.get("signal"),
        "siv_issues":       siv.get("issues"),
        "ce_signal":        ce.get("sentiment"),
        "tts_signal":       tts.get("decision"),
        "ce_confidence":    ce.get("confidence"),
        "ce_article_count": ce.get("article_count"),
        "tts_score":        tts.get("total_score"),
    }

    siv_prompt = f"""
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

    # 🔥 ONE CALL, 3 PROMPTS
    master_prompt = f"""
You will execute THREE independent prompts and return results in JSON.

--- PROMPT 1 (CE) ---
{ce_prompt}

--- PROMPT 2 (TTS) ---
{tts_prompt}

--- PROMPT 3 (SIV) ---
{siv_prompt}

Return STRICT JSON:
{{
  "ce": "...",
  "tts": "...",
  "siv": "..."
}}
"""

    raw = call_llm(master_prompt, max_tokens=1000, label="COMBINED EXPLANATION")

    try:
        return json.loads(raw)
    except:
        return {
            "ce": raw,
            "tts": raw,
            "siv": raw
        }


# =========================
# VERDICT EXPLANATION
# The verdict (BUY/SELL/HOLD) is already decided deterministically
# by weighted_score thresholds in verdict_agent. This function only
# explains WHY — it never re-decides.
# =========================
def explain_verdict(
    row: dict,
    ce_exp: str,
    tts_exp: str,
    siv_exp: str,
    pair: str,
) -> str:
    # Stagger verdict calls: acquire a dedicated verdict slot so at most
    # VERDICT_CONCURRENCY verdicts run simultaneously, preventing the burst
    # that happens when all rows finish phase-1 at the same time.
    _verdict_sem.acquire()
    try:
        return _do_explain_verdict(row, ce_exp, tts_exp, siv_exp, pair)
    finally:
        _verdict_sem.release()


def _do_explain_verdict(
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
You are an expert forex trading analyst writing a post-trade explanation.

The system has already made its decision: {verdict}
This was determined by: weighted_score = {round(weighted_score, 4)}
  - weighted_score >= 0.15  → BUY
  - weighted_score <= -0.15 → SELL
  - between -0.15 and 0.15 → HOLD

YOUR TASK:
Explain in clear prose WHY the decision is {verdict}.
Do NOT re-decide. Do NOT output BUY/SELL/HOLD as a standalone word.
Write only the reasoning — reference actual values from the data below.

------------------------------------------------------------
WHAT TO COVER
------------------------------------------------------------

1. WEIGHTED SCORE
   - State the score ({round(weighted_score, 4)}) and which zone it falls in
   - Explain how CE and TTS contributed to it

2. CE ANALYSIS
   - Confidence level and article count reliability
   - Sentiment strength and direction
   - Summary from CE explanation below

3. TTS ANALYSIS
   - Dominant indicator and score
   - Any conflicting signals
   - EMA 200 reliability impact
   - Summary from TTS explanation below

4. SIV ALIGNMENT
   - Whether CE and TTS aligned or conflicted
   - Impact on conviction (not on the decision)

------------------------------------------------------------
INPUT DATA
------------------------------------------------------------

weighted_score : {round(weighted_score, 4)}
verdict        : {verdict}
atr            : {round(atr, 5)}
sl_distance    : {sl_distance}
tp_distance    : {tp_distance}

TTS:
  decision          : {tts.get("decision")}
  total_score       : {tts.get("total_score")}
  ema_trend         : {tts.get("ema_trend")}
  rsi               : {tts.get("rsi")}
  bb_signal         : {tts.get("bb_signal")}
  ema_200_confidence: {tts.get("ema_200_confidence")}
  ema_200_reliable  : {tts.get("ema_200_reliable")}
  data_stale        : {tts.get("data_stale")}
  explanation       : {tts_exp}

CE:
  sentiment      : {ce.get("sentiment")}
  confidence     : {ce.get("confidence")}
  articles       : {ce.get("article_count")}
  raw_vibe       : {ce.get("raw_vibe")}
  sentiment_score: {ce.get("sentiment_score")}
  mean_score     : {ce.get("mean_score")}
  explanation    : {ce_exp}

SIV:
  signal     : {siv.get("signal")}
  issues     : {siv.get("issues")}
  explanation: {siv_exp}
"""

    return call_llm(prompt, max_tokens=700, label="VERDICT EXPLANATION")


# =========================
# PER-ROW PIPELINE (non-blocking, callback-chained)
# =========================
def _submit_row(executor, row: dict, pair: str, on_complete):

    date = row["date"]
    ce   = row.get("ce_output", {})
    tts  = row.get("tts_output", {})
    siv  = row.get("siv_output", {})

    def combined_done(future):
        try:
            result = future.result()
        except Exception as e:
            result = {"ce": str(e), "tts": str(e), "siv": str(e)}

        print(f"[EXPLAIN] {date} | combined done → submitting verdict...")

        vf = executor.submit(
            explain_verdict,
            row,
            result["ce"],
            result["tts"],
            result["siv"],
            pair,
        )
        vf.add_done_callback(lambda f: verdict_done(f, result))

    def verdict_done(future, combined_result):
        try:
            reasoning = future.result()
        except Exception as e:
            reasoning = f"explanation_failed: {e}"

        print(f"[EXPLAIN] {date} | done")

        _row_sem.release()

        on_complete({
            "date": date,
            "CE_Explanation": combined_result["ce"],
            "TTS_Explanation": combined_result["tts"],
            "SIV_Explanation": combined_result["siv"],
            "Verdict_Reasoning": reasoning,
        })

    print(f"[EXPLAIN] {date} | submitting COMBINED...")

    f = executor.submit(explain_combined, ce, tts, siv)
    f.add_done_callback(combined_done)


# =========================
# MAIN PIPELINE
# =========================
def run_explanation_pipeline(target_pair: str, target_year: int):

    total_start = time.perf_counter()

    project_root = Path(__file__).resolve().parents[1]
    reports_dir  = project_root / "reports"

    raw_path = reports_dir / "raw"   / f"backtest_{target_pair}_{target_year}_raw.json"
    csv_path = reports_dir           / f"backtest_{target_pair}_{target_year}.csv"
    out_path = reports_dir / "final" / f"backtest_{target_pair}_{target_year}_explained.csv"

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
    print(f"Rows to explain   : {len(raw_rows)}")
    print(f"LLM concurrency   : {LLM_CONCURRENCY} simultaneous requests")
    print(f"Max in-flight rows: {MAX_INFLIGHT_ROWS}")
    print(f"LLM calls per row : 2 (COMBINED + Verdict)\n")

    # =========================
    # GENERATE EXPLANATIONS IN PARALLEL
    # =========================
    explanations = {}
    exp_lock     = threading.Lock()
    remaining    = [len(raw_rows)]
    all_done     = threading.Event()

    def on_row_complete(result: dict):
        with exp_lock:
            explanations[result["date"]] = result
            remaining[0] -= 1
            if remaining[0] == 0:
                all_done.set()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for row in raw_rows:
            # FIX 2: Block here (main thread) until a row slot is free.
            # This is cheap — main thread just waits, no worker threads wasted.
            _row_sem.acquire()
            _submit_row(executor, row, target_pair, on_row_complete)

            time.sleep(0.1)

        all_done.wait()  # block main thread until every row finishes

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

    out_path.parent.mkdir(parents=True, exist_ok=True)
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