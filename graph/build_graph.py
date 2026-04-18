from langgraph.graph import StateGraph, END, START
from state.trading_state import TradingState

from agents.ce_agent import ce_agent
from agents.tts_agent import tts_agent
from agents.siv_agent import siv_agent
from agents.verdict_agent import verdict_agent


USE_PARALLEL = False


def retry_fanout(state):
    state["retry_count"] = state.get("retry_count", 0) + 1
    state["debug_log"].append(f"Retry fanout triggered ({state['retry_count']})")

    # HARD STOP after max retries
    if state["retry_count"] >= 2:
        state["action"] = "NONE"

    return state


def route_after_verdict(state):
    action = state.get("action", "NONE")
    if action == "RETRY_TTS_CE":
        return "retry"
    return "end"


def build_graph():
    graph = StateGraph(TradingState)

    graph.add_node("ce", ce_agent)
    graph.add_node("tts", tts_agent)
    graph.add_node("siv", siv_agent)
    graph.add_node("verdict", verdict_agent)
    graph.add_node("retry_fanout", retry_fanout)

    # =========================
    # MAIN FLOW
    # =========================
    if USE_PARALLEL:
        graph.add_edge(START, "ce")
        graph.add_edge(START, "tts")

        graph.add_edge("ce", "siv")
        graph.add_edge("tts", "siv")

    else:
        graph.add_edge(START, "tts")
        graph.add_edge("tts", "ce")
        graph.add_edge("ce", "siv")

    graph.add_edge("siv", "verdict")

    # =========================
    # RETRY LOGIC
    # =========================
    graph.add_conditional_edges(
        "verdict",
        route_after_verdict,
        {
            "retry": "retry_fanout",
            "end": END
        }
    )

    # 🔥 FIX: ONLY RETRY TTS (not CE)
    graph.add_edge("retry_fanout", "tts")

    return graph.compile()