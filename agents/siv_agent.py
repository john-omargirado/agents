import json
import os
import requests
from tools.siv_tools import check_data_integrity, calculate_technical_conflict
from utils.credentials import get_do_model_key

URL = "https://inference.do-ai.run/v1/chat/completions"


def call_qwen(payload):
    key = get_do_model_key()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    data = {
        "model": "alibaba-qwen3-32b",
        "messages": [
            {"role": "user", "content": json.dumps(payload)}
        ],
        "max_tokens": 200
    }

    response = requests.post(URL, headers=headers, json=data)

    try:
        result = response.json()
    except Exception as e:
        raise ValueError(f"Invalid JSON response: {response.text}") from e

    # =========================
    # SAFE EXTRACTION
    # =========================
    if "choices" in result:
        try:
            return result["choices"][0]["message"]["content"]
        except Exception:
            raise ValueError(f"Malformed choices response: {result}")

    # fallback error handling
    if "error" in result:
        raise ValueError(f"LLM error: {result['error']}")

    raise ValueError(f"Unknown response format: {result}")


def siv_agent(state):
    state["debug_log"].append("SIV agent: GPT mini integrity check")

    tts_output = state.get("tts_output", {})
    ce_output = state.get("ce_output", {})

    integrity_result = check_data_integrity({
        "tts_output": tts_output,
        "ce_output": ce_output,
        "price": state.get("price")
    })

    conflict_type = calculate_technical_conflict(
        tts_output.get("decision", "HOLD"),
        ce_output.get("sentiment", "NEUTRAL")
    )

    payload = {
        "integrity": integrity_result,
        "conflict": conflict_type,
        "tts": tts_output,
        "ce": ce_output
    }

    raw = call_qwen(payload)

    try:
        llm = json.loads(raw)
    except:
        llm = {}

    signal = llm.get("signal", "INCOHERENT")

    return {
        "siv_output": {
            "signal": signal,
            "conflict_type": conflict_type,
            "price_deviation": integrity_result["deviation"],
            "issues": integrity_result.get("issues", []),
            "tts_insufficient": tts_output.get("tts_insufficient", False),
            "data_quality_ok": integrity_result["pass"],
            "explanation": llm.get("explanation", "deterministic fallback")
        }
    }