import re
from tools.siv_tools import check_data_integrity, calculate_technical_conflict
from llm.ollama_client import siv_llm as llm

def siv_agent(state):
    state["debug_log"].append("SIV agent: starting Deterministic Audit")

    tts_output = state.get("tts_output", {})
    ce_output = state.get("ce_output", {})
    
    # 1. RUN TOOLS
    integrity_result = check_data_integrity({
        "tts_output": tts_output,
        "ce_output": ce_output,
        "price": state.get("price") 
    })

    conflict_type = calculate_technical_conflict(
        tts_output.get("decision", "HOLD"),
        ce_output.get("overall_sentiment", "NEUTRAL")
    )

    # 2. RULE-BASED LOGIC (Deterministic Signal)
    # Define rules for the integrity signal
    if not integrity_result['pass'] or "missing_tts_price" in integrity_result['issues']:
        signal = "INCOHERENT"
    elif conflict_type == "DIRECTIONAL_MISMATCH":
        signal = "PARTIAL"
    elif conflict_type in ["TECHNICAL_ONLY", "SENTIMENT_ONLY", "UNCLEAR"]:
        signal = "PARTIAL"
    elif conflict_type in ["ALIGNED", "NO_SIGNAL"]:
        signal = "COHERENT"
    else:
        signal = "INCOHERENT"

    # 3. LLM EXPLANATION LAYER
    # The SLM no longer chooses the signal; it only explains the choice made by Python.
    prompt = f"""
        You are the Signal Integrity Verifier. 

        Your deterministic audit has resulted in a signal of: {signal}

        Audit Details:
        - Data Integrity Pass: {integrity_result['pass']}
        - Price Deviation: {integrity_result['deviation']:.5f}
        - Logic Conflict Type: {conflict_type}
        - TTS Decision: {tts_output.get('decision')}
        - CE Sentiment: {ce_output.get('overall_sentiment')}
        - Articles Analyzed: {ce_output.get('articles_analyzed', 0)}

        Task: Explain why the signal is {signal} based on the alignment (or lack thereof) between Technicals (TTS) and Sentiment (CE).
        Output: 2 to 3 concise, professional sentences. No filler.
        """

    response = llm.invoke(prompt)
    explanation = getattr(response, "content", str(response)).strip()

    print(f"\n[SIV OUTPUT] Signal: {signal}")
    print(f"[SIV EXPLANATION] {explanation}\n")

    return {
        "siv_output": {
            "integrity_signal": signal,
            "explanation": explanation,
            "conflict_type": conflict_type,
            "deviation": integrity_result["deviation"],
            "issues": integrity_result.get("issues", []),
            "tts_insufficient": not bool(tts_output.get("indicators"))
        }
    }