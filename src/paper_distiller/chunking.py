from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    start_char: int
    end_char: int


def chunk_text(text: str, max_chars: int = 12_000, overlap_chars: int = 800) -> list[Chunk]:
    clean = text.strip()
    if not clean:
        return []
    if max_chars <= overlap_chars:
        raise ValueError("max_chars must be greater than overlap_chars")

    chunks: list[Chunk] = []
    start = 0
    index = 1
    while start < len(clean):
        target_end = min(len(clean), start + max_chars)
        end = _find_boundary(clean, start, target_end)
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(Chunk(index=index, text=chunk, start_char=start, end_char=end))
            index += 1
        if end >= len(clean):
            break
        start = max(0, end - overlap_chars)
    return chunks


def _find_boundary(text: str, start: int, target_end: int) -> int:
    if target_end >= len(text):
        return len(text)
    window_start = max(start + 1, target_end - 1_500)
    for marker in ("\n\n", ". ", "\n"):
        pos = text.rfind(marker, window_start, target_end)
        if pos > start:
            return pos + len(marker)
    return target_end

