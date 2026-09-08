import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    ollama_model: str = "qwen3.5:2b"
    ollama_model_quality: str = "qwen3.5:9b"
    ollama_base_url: str = "http://localhost:11434"
    llm_reasoning: bool = False
    tavily_api_key: str = ""
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "phone-finder"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    checkpoint_db: str = "state.db"
    database_url: str = "sqlite:///state.db"
    fetch_timeout: float = 10.0
    llm_timeout: float = 60.0
    turn_timeout: float = 150.0
    max_questions: int = 4
    cache_ttl_hours: int = 24
    price_reference: str = "US retail, USD"
    debug_ui: bool = False
    debug: bool = False
    cors_origins: str = ""
    rate_limit_requests: int = 10
    rate_limit_window_seconds: float = 60.0
    global_rate_limit_requests: int = 100
    global_rate_limit_window_seconds: float = 60.0
    max_concurrent_turns: int = 4
    max_request_bytes: int = 10 * 1024 * 1024
    api_keys: str = ""
    allowed_domains: str = (
        "amazon.com,bestbuy.com,bhphotovideo.com,gsmarena.com,"
        "store.google.com,apple.com,samsung.com,walmart.com"
    )
    disable_redirects: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def allowed_domain_set(self) -> frozenset[str]:
        """The outbound fetch allow-list. Env wins over the label map in tools.py."""
        return frozenset(
            domain.strip().lower().removeprefix("www.")
            for domain in self.allowed_domains.split(",")
            if domain.strip()
        )

    @property
    def checkpoint_path(self) -> str:
        """DATABASE_URL is canonical; CHECKPOINT_DB stays as the fallback."""
        url = self.database_url.strip()
        if url.startswith("sqlite:///"):
            return url.removeprefix("sqlite:///") or self.checkpoint_db
        if url.startswith("sqlite://"):
            return url.removeprefix("sqlite://") or self.checkpoint_db
        return self.checkpoint_db


def _export_tracing_env(s: Settings) -> None:
    flag = "true" if s.langsmith_tracing else "false"
    os.environ["LANGSMITH_TRACING"] = flag
    os.environ["LANGCHAIN_TRACING_V2"] = flag
    os.environ["LANGSMITH_PROJECT"] = s.langsmith_project
    os.environ["LANGCHAIN_PROJECT"] = s.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = s.langsmith_endpoint
    if s.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = s.langsmith_api_key
        os.environ["LANGCHAIN_API_KEY"] = s.langsmith_api_key


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    _export_tracing_env(s)
    return s
