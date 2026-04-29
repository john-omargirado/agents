from langgraph.graph import StateGraph, END, START
from state.trading_state import TradingState

from agents.ce_agent import ce_agent
from agents.tts_agent import tts_agent
from agents.siv_agent import siv_agent
from agents.verdict_agent import verdict_agent


USE_PARALLEL = False


# =========================
# RETRY LOGIC
# =========================
def retry_fanout(state):
    state["retry_count"] = state.get("retry_count", 0) + 1
    state["debug_log"].append(f"Retry fanout triggered ({state['retry_count']})")

    if state["retry_count"] >= 2:
        state["action"] = "NONE"

    state["backtest_mode"] = True
    return state


def route_after_verdict(state):
    return "retry" if state.get("action", "NONE") == "RETRY_TTS_CE" else "end"


# =========================
# GRAPH BUILD
# =========================
def build_graph():
    graph = StateGraph(TradingState)

    graph.add_node("ce", ce_agent)
    graph.add_node("tts", tts_agent)
    graph.add_node("siv", siv_agent)
    graph.add_node("verdict", verdict_agent)
    graph.add_node("retry_fanout", retry_fanout)

    # =========================
    # MAIN FLOW (SEQUENTIAL SAFE)
    # =========================
    graph.add_edge(START, "tts")
    graph.add_edge("tts", "ce")
    graph.add_edge("ce", "siv")
    graph.add_edge("siv", "verdict")

    # =========================
    # RETRY FLOW (SAFE ISOLATED PATH)
    # =========================
    graph.add_conditional_edges(
        "verdict",
        route_after_verdict,
        {
            "retry": "retry_fanout",
            "end": END
        }
    )

    # 🔥 CRITICAL FIX:
    # Retry does NOT re-run CE or SIV anymore
    graph.add_edge("retry_fanout", "tts")

    return graph.compile()