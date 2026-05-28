# llm_clients.py

from langchain_ollama import ChatOllama

verdict_llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0,
    num_ctx=2048,       # verdict prompt is short, 2048 is sufficient
    format="json"       # forces Ollama to constrain output to valid JSON
)