from functools import lru_cache

from langchain_ollama import ChatOllama

from app.config import get_settings


@lru_cache
def get_llm(temperature: float = 0.0, quality: bool = False) -> ChatOllama:
    s = get_settings()
    return ChatOllama(
        model=s.ollama_model_quality if quality else s.ollama_model,
        base_url=s.ollama_base_url,
        temperature=temperature,
        num_ctx=8192,
        reasoning=s.llm_reasoning,
    )
