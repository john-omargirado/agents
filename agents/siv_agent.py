import re
from tools.siv_tools import check_data_integrity, calculate_technical_conflict

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
            ce_output.get("sentiment", "NEUTRAL")
        )

    tts_insufficient = tts_output.get("tts_insufficient", False)

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

    # 3. OUTPUT STRUCTURE
    if signal == "INCOHERENT":
        explanation = "Data integrity issues or missing alignment detected."
    elif signal == "PARTIAL":
        explanation = f"Partial alignment due to {conflict_type.lower()}."
    elif signal == "COHERENT":
        explanation = "Technical and sentiment signals are aligned."
    else:
        explanation = "Unclear signal state."

    print(f"\n[SIV OUTPUT] Signal: {signal}")
    print(f"[SIV EXPLANATION] {explanation}\n")

    return {
        "siv_output": {
            "signal": signal,
            "conflict_type": conflict_type,
            "price_deviation": integrity_result["deviation"],
            "issues": integrity_result.get("issues", []),
            "tts_insufficient": tts_insufficient,
            "data_quality_ok": integrity_result["pass"],
            "explanation": explanation
        }
    }