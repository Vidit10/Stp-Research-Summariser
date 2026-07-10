from __future__ import annotations

import re
from collections import Counter

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])")
WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")
SECTION_PATTERN = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\s+)?(?P<title>"
    r"abstract|introduction|background|related work|methodology|methods|experimental setup|"
    r"experiment|experiments|results|findings|discussion|limitations?|conclusion|future work|"
    r"references|appendix"
    r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)

STOPWORDS = {
    "about",
    "after",
    "also",
    "although",
    "among",
    "because",
    "between",
    "could",
    "during",
    "each",
    "from",
    "have",
    "into",
    "more",
    "most",
    "other",
    "paper",
    "result",
    "results",
    "should",
    "study",
    "such",
    "than",
    "that",
    "their",
    "there",
    "these",
    "this",
    "through",
    "using",
    "were",
    "where",
    "which",
    "while",
    "with",
    "would",
}


def split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    return [sentence.strip() for sentence in SENTENCE_PATTERN.split(normalized) if sentence.strip()]


def keywords(text: str, limit: int = 20) -> list[str]:
    words = [w.lower() for w in WORD_PATTERN.findall(text)]
    counts = Counter(w for w in words if w not in STOPWORDS)
    return [word for word, _ in counts.most_common(limit)]


def select_relevant_sentences(
    text: str,
    *,
    query: str | None = None,
    limit: int = 12,
    required_terms: list[str] | None = None,
) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    doc_terms = set(keywords(text, 40))
    query_terms = set(keywords(query or "", 20))
    required_terms_lower = [term.lower() for term in required_terms or []]

    scored: list[tuple[float, int, str]] = []
    for idx, sentence in enumerate(sentences):
        lower = sentence.lower()
        sentence_terms = set(keywords(sentence, 30))
        score = len(sentence_terms & doc_terms) * 0.4
        score += len(sentence_terms & query_terms) * 2.0
        score += sum(2.5 for term in required_terms_lower if term in lower)
        if re.search(r"\b\d+(?:\.\d+)?\s*(%|percent|p\s*[<=>]|n\s*=|participants|samples)\b", lower):
            score += 2.5
        if any(term in lower for term in ("limitation", "weakness", "threat", "future work")):
            score += 1.5
        if any(term in lower for term in ("we find", "we found", "shows", "demonstrates")):
            score += 1.0
        if len(sentence) < 45:
            score -= 1.0
        scored.append((score, idx, sentence))

    selected = sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]
    return [sentence for _, _, sentence in sorted(selected, key=lambda item: item[1])]


def detect_sections(text: str) -> list[str]:
    titles = []
    seen = set()
    for match in SECTION_PATTERN.finditer(text):
        title = match.group("title").strip().title()
        if title.lower() not in seen:
            titles.append(title)
            seen.add(title.lower())
    return titles

