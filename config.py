from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # EDGAR
    edgar_user_agent: str = "LedgerLens dev@example.com"

    # Database
    database_url: str = "postgresql+psycopg://ledgerlens:ledgerlens@localhost:5432/ledgerlens"

    # Data paths
    data_dir: str = "./data"

    # Models
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    hf_token: str = ""
    colpali_checkpoint: str = "vidore/colpali-v1.2"
    vlm_model: str = "Qwen/Qwen2-VL-7B-Instruct"
    planner_model: str = "gpt-4o-mini"
    synthesizer_model: str = "gpt-4o"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # API
    api_key_hashes: str = ""
    frontend_url: str = "http://localhost:3000"

    # Retrieval
    retrieval_stage1_n: int = 50
    retrieval_top_k: int = 5

    # Eval
    eval_min_numeric_em: float = 0.0
    eval_max_hallucination: float = 1.0

    # Observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Rate limiting
    rate_limit_per_minute: int = 10

    @property
    def valid_api_key_hashes(self) -> set[str]:
        return {h.strip() for h in self.api_key_hashes.split(",") if h.strip()}

    @property
    def embeddings_dir(self) -> str:
        return f"{self.data_dir}/embeddings"

    @property
    def pages_dir(self) -> str:
        return f"{self.data_dir}/pages"

    @property
    def html_dir(self) -> str:
        return f"{self.data_dir}/html"

    @property
    def pdf_dir(self) -> str:
        return f"{self.data_dir}/pdf"


settings = Settings()
