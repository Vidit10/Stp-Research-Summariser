from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from paper_distiller.text_utils import detect_sections, keywords, select_relevant_sentences


@dataclass(frozen=True)
class Message:
    role: str
    content: str


class ModelProvider(ABC):
    name: str

    @abstractmethod
    def complete(self, messages: Sequence[Message], *, temperature: float = 0.1) -> str:
        raise NotImplementedError


class ExtractiveProvider(ModelProvider):
    name = "extractive"

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.1) -> str:
        prompt = "\n\n".join(message.content for message in messages if message.role != "system")
        if prompt.startswith("PASS 1:"):
            return _extractive_pass_1(prompt)
        if prompt.startswith("PASS 2:"):
            return _extractive_pass_2(prompt)
        if prompt.startswith("PASS 3:"):
            return _extractive_pass_3(prompt)
        if prompt.startswith("PASS 4:"):
            return _extractive_pass_4(prompt)
        if prompt.startswith("PASS 5:") and "query-driven extraction" in prompt:
            return _extractive_query_synthesis(prompt)
        if prompt.startswith("PASS 5:"):
            return _extractive_standard_synthesis(prompt)
        return _extractive_notes(prompt)


class OpenAIProvider(ModelProvider):
    name = "openai"

    def __init__(self, *, model: str, api_key: str | None = None, base_url: str | None = None):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install provider dependencies with `python -m pip install -e .[providers]`.") from exc
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.1) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": msg.role, "content": msg.content} for msg in messages],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


class AnthropicProvider(ModelProvider):
    name = "anthropic"

    def __init__(self, *, model: str, api_key: str | None = None):
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install provider dependencies with `python -m pip install -e .[providers]`.") from exc
        self.model = model
        self.client = Anthropic(api_key=api_key)

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.1) -> str:
        system = "\n\n".join(msg.content for msg in messages if msg.role == "system")
        user_messages = [msg for msg in messages if msg.role != "system"]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": msg.content} for msg in user_messages],
        )
        parts = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts)


def build_provider(provider: str, *, model: str | None, api_key: str | None, base_url: str | None):
    normalized = provider.lower()
    if normalized == "extractive":
        return ExtractiveProvider()
    if normalized == "bharatcode":
        return OpenAIProvider(
            model=model or "default",
            api_key=api_key,
            base_url=base_url or "https://bharatcode.ai/api/model/v1",
        )
    if normalized in {"openai", "openai-compatible"}:
        return OpenAIProvider(
            model=model or "gpt-4.1-mini",
            api_key=api_key,
            base_url=base_url if normalized == "openai-compatible" else base_url,
        )
    if normalized == "anthropic":
        return AnthropicProvider(model=model or "claude-3-5-sonnet-latest", api_key=api_key)
    raise ValueError(f"Unknown provider: {provider}")


def list_openai_compatible_models(*, base_url: str, api_key: str | None) -> list[str]:
    url = base_url.rstrip("/") + "/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=30) as response:
            import json

            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Model catalog request failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach model catalog at {url}: {exc.reason}") from exc

    data = payload.get("data", payload if isinstance(payload, list) else [])
    models: list[str] = []
    for item in data:
        if isinstance(item, str):
            models.append(item)
        elif isinstance(item, dict):
            model_id = item.get("id") or item.get("name") or item.get("model")
            if model_id:
                models.append(str(model_id))
    return models


def _extract_query(prompt: str) -> str | None:
    marker = "USER QUERY:"
    if marker not in prompt:
        return None
    after = prompt.split(marker, 1)[1].strip()
    return after.splitlines()[0].strip() or None


def _extractive_pass_1(prompt: str) -> str:
    text = _after_marker(prompt, "CHUNK")
    if ":" in text:
        text = text.split(":", 1)[1]
    problem = select_relevant_sentences(text, limit=3, required_terms=["problem", "examines", "asks"])
    methods = select_relevant_sentences(
        text,
        limit=4,
        required_terms=["method", "experiment", "dataset", "sample", "participants", "controls"],
    )
    findings = select_relevant_sentences(
        text,
        limit=6,
        required_terms=["result", "found", "increased", "significant", "conclude"],
    )
    limitations = select_relevant_sentences(
        text,
        limit=4,
        required_terms=["limitation", "hypothetical", "not", "may not", "did not"],
    )
    return "\n".join(
        [
            "Problem / research focus:",
            _bullets(problem),
            "",
            "Method signals:",
            _bullets(methods),
            "",
            "Finding signals:",
            _bullets(findings),
            "",
            "Limitation signals:",
            _bullets(limitations) or "- No explicit limitations detected in this chunk.",
        ]
    ).strip()


def _extractive_pass_2(prompt: str) -> str:
    notes = _after_marker(prompt, "PASS 1 NOTES:")
    sections_text = _between(prompt, "DETECTED SECTIONS:", "VISUAL REFERENCES:").strip()
    sections = [line.strip("- ").strip() for line in sections_text.splitlines() if line.strip()]
    if not sections or sections == ["Not detected"]:
        sections = detect_sections(notes)
    methods = select_relevant_sentences(
        notes,
        limit=8,
        required_terms=["participants", "sample", "experiment", "dataset", "controls", "regression"],
    )
    research = select_relevant_sentences(notes, limit=3, required_terms=["asks", "whether", "question"])
    visual_refs = _after_marker(prompt, "VISUAL REFERENCES:").strip()
    return "\n".join(
        [
            "Main Sections",
            _bullets(sections) or "- Not detected.",
            "",
            "Research Question",
            _bullets(research) or "- Not clearly stated in extracted text.",
            "",
            "Methodology Details Needed for Credibility",
            _bullets(methods) or "- No clear methodology details detected.",
            "",
            "Direct Review Candidates",
            visual_refs if visual_refs else "- No figure, table, or equation references detected.",
        ]
    ).strip()


def _extractive_pass_3(prompt: str) -> str:
    material = _after_marker(prompt, "PASS 1 NOTES:")
    claims = select_relevant_sentences(
        material,
        limit=8,
        required_terms=["found", "increased", "significant", "conclude", "effect", "improvement"],
    )
    quantitative = select_relevant_sentences(
        material,
        limit=6,
        required_terms=["percent", "p <", "participants", "sample", "statistically"],
    )
    limitations = select_relevant_sentences(
        material,
        limit=6,
        required_terms=["limitation", "not", "may not", "did not", "hypothetical"],
    )
    return "\n".join(
        [
            "Claims and Findings",
            _bullets(claims) or "- No clear claims detected.",
            "",
            "Evidence and Validation Signals",
            _bullets(quantitative) or "- No quantitative validation detected.",
            "",
            "Limitations, Assumptions, and Threats",
            _bullets(limitations) or "- No explicit limitations detected.",
            "",
            "Faithfulness Notes",
            "- Statistical significance is reported only where the extracted text explicitly reports it.",
            "- Items above are extractive signals, not independent verification.",
        ]
    ).strip()


def _extractive_pass_4(prompt: str) -> str:
    query = _extract_query(prompt) or ""
    material = _after_marker(prompt, "EVIDENCE:")
    relevant = select_relevant_sentences(material, query=query, limit=8)
    limitations = select_relevant_sentences(
        material, query=query, limit=5, required_terms=["not", "limitation", "missing", "did not"]
    )
    strength = "Moderate evidence" if relevant else "No evidence"
    if relevant and not any("significant" in item.lower() or "p <" in item.lower() for item in relevant):
        strength = "Weak to moderate evidence"
    return "\n".join(
        [
            "Direct Answer",
            _direct_answer(query, relevant, limitations),
            "",
            "Relevant Evidence",
            _bullets(relevant) or "- No directly relevant evidence detected.",
            "",
            "Confidence Assessment",
            f"- {strength}. This is based on extracted statements only.",
            "",
            "Missing Information",
            _bullets(limitations) or "- The extracted text does not clearly state remaining gaps.",
            "",
            "Startup Relevance",
            "- Treat evidence as input to product, trust, risk, and validation decisions, not as proof by itself.",
        ]
    ).strip()


def _extractive_standard_synthesis(prompt: str) -> str:
    structure = _between(prompt, "DOCUMENT STRUCTURE:", "EVIDENCE:")
    evidence = _between(prompt, "EVIDENCE:", "VISUAL REFERENCES:")
    visual_refs = _after_marker(prompt, "VISUAL REFERENCES:")
    material = f"{structure}\n\n{evidence}"
    findings = select_relevant_sentences(
        material,
        limit=7,
        required_terms=["found", "increased", "significant", "conclude", "effect"],
    )
    methods = select_relevant_sentences(
        material,
        limit=6,
        required_terms=["participants", "experiment", "sample", "controls", "regression"],
    )
    limitations = select_relevant_sentences(
        material,
        limit=6,
        required_terms=["limitation", "hypothetical", "may not", "did not", "not"],
    )
    terms = keywords(material, limit=8)
    return "\n".join(
        [
            "Executive Summary",
            _paragraph(findings[:3]) or "No concise executive summary could be extracted.",
            "",
            "Research Question",
            _paragraph(select_relevant_sentences(material, limit=3, required_terms=["asks", "whether"]))
            or "Not clearly stated in extracted text.",
            "",
            "Key Findings",
            _bullets(findings) or "- No major findings detected.",
            "",
            "Methodology",
            _bullets(methods) or "- No clear methodology details detected.",
            "",
            "Evidence and Validation",
            _bullets(
                select_relevant_sentences(
                    material, limit=6, required_terms=["percent", "p <", "significant", "participants"]
                )
            )
            or "- No quantitative validation detected.",
            "",
            "Limitations",
            _bullets(limitations) or "- No explicit limitations detected; treat this as uncertain.",
            "",
            "Practical Takeaways",
            "- Use the findings as startup decision inputs, especially where they affect product trust, risk, validation, or user behavior.",
            "- Be cautious where the paper uses hypothetical settings, narrow samples, or untested long-term behavior.",
            "",
            "Terminology",
            _bullets(terms) or "- No specialized terminology detected.",
            "",
            "Recommended Original Sections to Review",
            _visual_review_lines(visual_refs),
        ]
    ).strip()


def _extractive_query_synthesis(prompt: str) -> str:
    query = _extract_query(prompt) or ""
    analysis = _between(prompt, "QUERY ANALYSIS:", "VISUAL REFERENCES:")
    visual_refs = _after_marker(prompt, "VISUAL REFERENCES:")
    relevant = select_relevant_sentences(analysis, query=query, limit=7)
    missing = select_relevant_sentences(
        analysis, query=query, limit=5, required_terms=["missing", "not", "limitation", "did not"]
    )
    return "\n".join(
        [
            "Direct Answer",
            _direct_answer(query, relevant, missing),
            "",
            "Relevant Evidence",
            _bullets(relevant) or "- No directly relevant evidence detected.",
            "",
            "Supporting Quotes",
            _bullets(relevant[:3]) or "- No short supporting excerpts detected.",
            "",
            "Confidence Assessment",
            "- Moderate if the extracted evidence includes controlled evaluation and reported significance; weaker otherwise.",
            "",
            "Missing Information",
            _bullets(missing) or "- No missing information was clearly detected in the extracted text.",
            "",
            "Startup Relevance",
            "- Use this evidence to shape product trust features, validation plans, risk controls, and claims you are willing to make.",
            "",
            "Recommended Original Sections to Review",
            _visual_review_lines(visual_refs),
        ]
    ).strip()


def _extractive_notes(prompt: str) -> str:
    selected = select_relevant_sentences(prompt, limit=12)
    return _bullets(selected) or "No extractable evidence found in the provided text."


def _after_marker(text: str, marker: str) -> str:
    if marker not in text:
        return text
    return text.split(marker, 1)[1].strip()


def _between(text: str, start_marker: str, end_marker: str) -> str:
    after = _after_marker(text, start_marker)
    if end_marker in after:
        return after.split(end_marker, 1)[0].strip()
    return after.strip()


def _bullets(items) -> str:
    cleaned = [str(item).strip(" -") for item in items if str(item).strip(" -")]
    return "\n".join(f"- {item}" for item in cleaned)


def _paragraph(items: list[str]) -> str:
    return " ".join(item.strip(" -") for item in items if item.strip(" -"))


def _direct_answer(query: str, evidence: list[str], limitations: list[str]) -> str:
    if not evidence:
        return "No. The extracted text does not provide direct evidence for the query."
    qualifier = "Partially"
    if any("significant" in item.lower() or "p <" in item.lower() for item in evidence):
        qualifier = "Yes, with important caveats"
    if limitations:
        return f"{qualifier}. The extracted evidence addresses the query, but limitations remain."
    return f"{qualifier}. The extracted evidence appears relevant to the query."


def _visual_review_lines(visual_refs: str) -> str:
    if not visual_refs.strip() or "No figure" in visual_refs:
        return "- No figure, table, or equation references detected."
    lines = []
    for line in visual_refs.splitlines():
        cleaned = line.strip(" -")
        if not cleaned:
            continue
        if cleaned.lower().startswith(("figure", "fig", "table", "equation", "eq")):
            lines.append(
                f"- {cleaned}. Review directly because visual or tabular details may not be safely compressed into text."
            )
    return "\n".join(lines) or "- No figure, table, or equation references detected."
