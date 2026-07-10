from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    provider: str = "extractive"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    max_chars_per_chunk: int = 12_000
    overlap_chars: int = 800
    log_level: str = "INFO"

    @classmethod
    def from_env(
        cls,
        *,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_chars_per_chunk: int | None = None,
        overlap_chars: int | None = None,
        log_level: str | None = None,
    ) -> "Settings":
        load_dotenv()
        resolved_provider = provider or os.getenv("PAPER_DISTILLER_PROVIDER", "extractive")
        resolved_api_key = (
            api_key
            or os.getenv("PAPER_DISTILLER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
        )
        return cls(
            provider=resolved_provider.lower(),
            model=model or os.getenv("PAPER_DISTILLER_MODEL"),
            base_url=base_url or os.getenv("PAPER_DISTILLER_BASE_URL"),
            api_key=resolved_api_key,
            max_chars_per_chunk=max_chars_per_chunk
            or _int_env("PAPER_DISTILLER_MAX_CHARS_PER_CHUNK", 12_000),
            overlap_chars=overlap_chars or _int_env("PAPER_DISTILLER_OVERLAP_CHARS", 800),
            log_level=log_level or os.getenv("PAPER_DISTILLER_LOG_LEVEL", "INFO"),
        )


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def load_dotenv(path: Path | None = None) -> None:
    dotenv_path = path or Path.cwd() / ".env"
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
