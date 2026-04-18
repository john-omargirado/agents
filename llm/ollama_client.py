# llm_clients.py

from langchain_ollama import ChatOllama

tts_llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
    num_ctx=4096
)

siv_llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
    num_ctx=4096
)

verdict_llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    num_ctx=4096

)