import json
from datetime import datetime
from graph.build_graph import build_graph


def pretty(obj):
    return json.dumps(obj, indent=2)


# =========================
# NORMALIZE PAIR
# =========================
def normalize_pair(pair: str):
    return pair.replace("/", "").replace("\\", "").strip().upper()


# =========================
# NORMALIZE DATE
# =========================
def normalize_date(date_str: str):
    if not date_str:
        return None

    for fmt in ("%m-%d-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    print("[WARN] Invalid date format. Expected MM-DD-YYYY.")
    return None


# =========================
# STATE BUILDER
# =========================
def create_state(pair, capital, leverage, risk, date, live_mode, backtest_mode, skip_llm):

    pair_raw = pair.strip()
    pair_safe = normalize_pair(pair_raw)

    return {
        # INPUTS
        "currency_pair": pair_safe,
        "pair_raw": pair_raw,
        "pair_safe": pair_safe,

        "account_capital": float(capital),
        "leverage": leverage.strip(),
        "risk_per_trade": float(risk),
        "target_date": date,

        # ENGINE STATE
        "price": 0.0,
        "calibration_threshold": 0.2,

        "ce_output": {},
        "tts_output": {},
        "siv_output": {},

        "verdict": "",
        "verdict_reasoning": "",
        "weighted_score": 0.0,
        "risk_multiplier": 0.0,

        # NEW STRUCTURE
        "trade_output": {},

        # SYSTEM
        "debug_log": [],
        "retry_count": 0,
        "action": "",

        # MODES
        "backtest_mode": bool(backtest_mode),
        "live_mode": bool(live_mode),
        "skip_llm": bool(skip_llm),

        # CONTEXT
        "atr": 0.0,
        "regime": ""
    }


# =========================
# RUN ENGINE
# =========================
def run(graph, state):
    result = graph.invoke(state)

    trade = result.get("trade_output", {})

    print("\n================ RESULT ================\n")

    print("PAIR USED:")
    print(result.get("currency_pair"))

    print("TARGET DATE USED:")
    print(result.get("target_date"))

    print("\nVERDICT:")
    print(result.get("verdict"))

    print("\nVERDICT DETAILS (TRADE OUTPUT):")
    print("Position Size:", trade.get("position_size"))
    print("Risk Amount:", trade.get("risk_amount"))
    print("Max Exposure:", trade.get("max_exposure"))
    print("SL Distance:", trade.get("sl_distance"))
    print("TP Distance:", trade.get("tp_distance"))
    print("ATR:", trade.get("atr"))

    print("\nWEIGHTED SCORE:")
    print(result.get("weighted_score"))

    print("\nCE / TTS SUMMARY:")
    print("CE Score:", result.get("ce_output", {}).get("ce_score"))
    print("TTS Score:", result.get("tts_output", {}).get("total_score"))

    print("\nDEBUG LOG:")
    for log in result.get("debug_log", []):
        print(" -", log)

    return result


# =========================
# INTERACTIVE LOOP
# =========================
def main():
    graph = build_graph()

    print("\n================ FOREX AI INTERACTIVE MODE ================\n")

    while True:
        print("\n--- NEW ANALYSIS ---")

        pair = input("Forex Pair (e.g. USD/JPY): ").strip()

        try:
            capital = float(input("Account Capital (e.g. 1000): ").strip())
        except:
            print("Invalid capital, using 1000")
            capital = 1000

        leverage = input("Leverage (e.g. 1:10): ").strip()

        try:
            risk = float(input("Risk % per trade (e.g. 1): ").strip())
        except:
            print("Invalid risk, using 1%")
            risk = 1.0

        mode = input("Mode (live/backtest): ").strip().lower()
        live_mode = mode == "live"
        backtest_mode = mode == "backtest"

        date = input("Date (MM-DD-YYYY): ").strip()

        skip_llm = input("Skip LLM? (y/n): ").strip().lower() == "y"

        state = create_state(
            pair=pair,
            capital=capital,
            leverage=leverage,
            risk=risk,
            date=date,
            live_mode=live_mode,
            backtest_mode=backtest_mode,
            skip_llm=skip_llm
        )

        print("\n================ RUNNING ANALYSIS ================\n")

        run(graph, state)

        again = input("\nRun another trade? (y/n): ").strip().lower()
        if again != "y":
            break

    print("\nSession ended.")


if __name__ == "__main__":
    main()