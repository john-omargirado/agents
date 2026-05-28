import json
from typing import Any, Dict
from datetime import datetime

try:
    import numpy as np
    import pandas as pd
except Exception:
    np = None
    pd = None


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    # primitive
    if isinstance(value, (str, bool, int, float)):
        return value

    # numpy scalars
    if np is not None:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)

    # sequences
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]

    if np is not None and pd is not None:
        if isinstance(value, (np.ndarray, pd.Series)):
            return [_json_safe(v) for v in list(value)]
        if isinstance(value, pd.Timestamp) or isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, pd.DataFrame):
            return [_json_safe(row.to_dict()) for _, row in value.iterrows()]

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    # fallback: try JSON dump/load, else string
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return str(value)


def prepare_siv_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    """Construct a JSON-serializable SIV input dict from the `state`.

    Returns a dict safe to pass to `json.dumps()` or to include in LLM prompts.
    """
    ce = state.get("ce_output") or {}
    tts = state.get("tts_output") or {}

    payload = {
        "ce_signal": (ce.get("signal") or ce.get("sentiment") or "NEUTRAL"),
        "tts_signal": (tts.get("decision") or tts.get("signal") or "HOLD"),
        "actual_price": float(state.get("price") or 0.0),
        "tts_price": float(tts.get("price") or 0.0),
        "ce_article_count": int(ce.get("article_count") or 0),
    }

    return _json_safe(payload)