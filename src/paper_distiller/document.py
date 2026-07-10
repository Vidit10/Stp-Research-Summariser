from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Page:
    number: int
    text: str


@dataclass(frozen=True)
class FigureRef:
    kind: str
    label: str
    page: int | None
    context: str


@dataclass(frozen=True)
class PaperDocument:
    source_path: Path
    pages: list[Page]
    metadata: dict[str, str] = field(default_factory=dict)
    visual_refs: list[FigureRef] = field(default_factory=list)

    @property
    def text(self) -> str:
        parts = []
        for page in self.pages:
            if page.text.strip():
                parts.append(f"[Page {page.number}]\n{page.text.strip()}")
        return "\n\n".join(parts)

    @property
    def char_count(self) -> int:
        return len(self.text)

