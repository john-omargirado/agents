import os
import requests

URL = "https://inference.do-ai.run/v1/chat/completions"
KEY = os.getenv("DO_MODEL_ACCESS_KEY")


def call_llm(prompt: str, model: str, max_tokens: int = 500):
    if not KEY:
        raise ValueError("DO_MODEL_ACCESS_KEY is missing")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {KEY}"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2
    }

    response = requests.post(URL, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(response.text)

    return response.json()["choices"][0]["message"]["content"]