from __future__ import annotations

import logging
from dataclasses import dataclass

from paper_distiller.chunking import chunk_text
from paper_distiller.config import Settings
from paper_distiller.document import FigureRef, PaperDocument
from paper_distiller.models import Message, ModelProvider, build_provider
from paper_distiller.prompts import (
    DOCUMENT_UNDERSTANDING_PROMPT,
    EVIDENCE_EXTRACTION_PROMPT,
    QUERY_ANALYSIS_PROMPT,
    QUERY_SYNTHESIS_PROMPT,
    STANDARD_SYNTHESIS_PROMPT,
    STRUCTURE_EXTRACTION_PROMPT,
    SYSTEM_PROMPT,
)
from paper_distiller.text_utils import detect_sections, keywords, select_relevant_sentences

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistillationResult:
    mode: str
    text: str
    pass_outputs: dict[str, str]


class PaperDistiller:
    def __init__(self, settings: Settings, provider: ModelProvider | None = None):
        self.settings = settings
        self.provider = provider or build_provider(
            settings.provider,
            model=settings.model,
            api_key=settings.api_key,
            base_url=settings.base_url,
        )

    def distill(self, document: PaperDocument, *, query: str | None = None) -> DistillationResult:
        if not document.text.strip():
            raise ValueError("No extractable text was found in the document.")

        LOGGER.info("Starting paper distillation for %s", document.source_path.name)
        if getattr(self.provider, "name", "") == "extractive":
            return self._distill_extractive(document, query=query)

        chunks = chunk_text(
            document.text,
            max_chars=self.settings.max_chars_per_chunk,
            overlap_chars=self.settings.overlap_chars,
        )
        LOGGER.info("Created %s chunks", len(chunks))

        understanding = self._pass_document_understanding(chunks)
        structure = self._pass_structure_extraction(document, understanding)
        evidence = self._pass_evidence_extraction(understanding, structure)

        pass_outputs = {
            "pass_1_document_understanding": understanding,
            "pass_2_structure_extraction": structure,
            "pass_3_evidence_extraction": evidence,
        }

        visual_refs = _format_visual_refs(document.visual_refs)
        if query:
            query_analysis = self._pass_query_analysis(query, structure, evidence)
            final_text = self._pass_query_synthesis(query, query_analysis, visual_refs)
            pass_outputs["pass_4_query_focused_analysis"] = query_analysis
            pass_outputs["pass_5_final_synthesis"] = final_text
            return DistillationResult(mode="goal-driven", text=final_text, pass_outputs=pass_outputs)

        final_text = self._pass_standard_synthesis(structure, evidence, visual_refs)
        pass_outputs["pass_5_final_synthesis"] = final_text
        return DistillationResult(mode="standard", text=final_text, pass_outputs=pass_outputs)

    def _pass_document_understanding(self, chunks) -> str:
        outputs = []
        for chunk in chunks:
            LOGGER.info("Pass 1: understanding chunk %s", chunk.index)
            output = self._complete(
                DOCUMENT_UNDERSTANDING_PROMPT.format(
                    chunk_index=chunk.index,
                    chunk_text=chunk.text,
                )
            )
            outputs.append(f"Chunk {chunk.index}\n{output}")
        return "\n\n".join(outputs)

    def _pass_structure_extraction(self, document: PaperDocument, understanding: str) -> str:
        LOGGER.info("Pass 2: extracting structure")
        sections = "\n".join(f"- {section}" for section in detect_sections(document.text)) or "Not detected"
        visual_refs = _format_visual_refs(document.visual_refs)
        return self._complete(
            STRUCTURE_EXTRACTION_PROMPT.format(
                understanding=understanding,
                sections=sections,
                visual_refs=visual_refs,
            )
        )

    def _pass_evidence_extraction(self, understanding: str, structure: str) -> str:
        LOGGER.info("Pass 3: extracting evidence")
        return self._complete(
            EVIDENCE_EXTRACTION_PROMPT.format(understanding=understanding, structure=structure)
        )

    def _pass_query_analysis(self, query: str, structure: str, evidence: str) -> str:
        LOGGER.info("Pass 4: query-focused analysis")
        return self._complete(
            QUERY_ANALYSIS_PROMPT.format(query=query, structure=structure, evidence=evidence)
        )

    def _pass_standard_synthesis(self, structure: str, evidence: str, visual_refs: str) -> str:
        LOGGER.info("Pass 5: standard synthesis")
        return self._complete(
            STANDARD_SYNTHESIS_PROMPT.format(
                structure=structure,
                evidence=evidence,
                visual_refs=visual_refs,
            )
        )

    def _pass_query_synthesis(self, query: str, query_analysis: str, visual_refs: str) -> str:
        LOGGER.info("Pass 5: query synthesis")
        return self._complete(
            QUERY_SYNTHESIS_PROMPT.format(
                query=query,
                query_analysis=query_analysis,
                visual_refs=visual_refs,
            )
        )

    def _complete(self, prompt: str) -> str:
        return self.provider.complete(
            [Message(role="system", content=SYSTEM_PROMPT), Message(role="user", content=prompt)],
            temperature=0.1,
        ).strip()

    def _distill_extractive(
        self, document: PaperDocument, *, query: str | None = None
    ) -> DistillationResult:
        text = document.text
        sections = detect_sections(text)
        visual_refs = _format_visual_refs(document.visual_refs)

        understanding = _extractive_understanding(text)
        structure = _extractive_structure(text, sections, visual_refs)
        evidence = _extractive_evidence(text)
        pass_outputs = {
            "pass_1_document_understanding": understanding,
            "pass_2_structure_extraction": structure,
            "pass_3_evidence_extraction": evidence,
        }

        if query:
            query_analysis = _extractive_query_analysis(text, query)
            final_text = _extractive_query_output(text, query, visual_refs)
            pass_outputs["pass_4_query_focused_analysis"] = query_analysis
            pass_outputs["pass_5_final_synthesis"] = final_text
            return DistillationResult(mode="goal-driven", text=final_text, pass_outputs=pass_outputs)

        final_text = _extractive_standard_output(text, sections, visual_refs)
        pass_outputs["pass_5_final_synthesis"] = final_text
        return DistillationResult(mode="standard", text=final_text, pass_outputs=pass_outputs)


def _format_visual_refs(refs: list[FigureRef]) -> str:
    if not refs:
        return "No figure, table, or equation references detected."
    lines = []
    for ref in refs[:80]:
        page = f"page {ref.page}" if ref.page else "page unknown"
        lines.append(f"- {ref.kind} {ref.label}, {page}: {ref.context}")
    if len(refs) > 80:
        lines.append(f"- {len(refs) - 80} additional visual references omitted from prompt context.")
    return "\n".join(lines)


def _extractive_understanding(text: str) -> str:
    return "\n".join(
        [
            "Problem / Research Focus",
            _bullet_block(select_relevant_sentences(text, limit=4, required_terms=["examines", "asks"])),
            "",
            "Initial Method Signals",
            _bullet_block(
                select_relevant_sentences(
                    text,
                    limit=5,
                    required_terms=["participants", "experiment", "dataset", "sample", "controls"],
                )
            ),
            "",
            "Initial Finding Signals",
            _bullet_block(
                select_relevant_sentences(
                    text,
                    limit=6,
                    required_terms=["results", "found", "increased", "significant", "conclude"],
                )
            ),
        ]
    ).strip()


def _extractive_structure(text: str, sections: list[str], visual_refs: str) -> str:
    methodology = select_relevant_sentences(
        text,
        limit=7,
        required_terms=["participants", "randomly", "conditions", "measured", "regression"],
    )
    research = select_relevant_sentences(text, limit=3, required_terms=["asks", "whether", "question"])
    return "\n".join(
        [
            "Main Sections",
            _bullet_block(sections) or "- Not detected.",
            "",
            "Research Question",
            _bullet_block(research) or "- Not clearly stated in extracted text.",
            "",
            "Methodology Details Needed for Credibility",
            _bullet_block(methodology) or "- No clear methodology details detected.",
            "",
            "Direct Review Candidates",
            visual_refs,
        ]
    ).strip()


def _extractive_evidence(text: str) -> str:
    findings = select_relevant_sentences(
        text,
        limit=8,
        required_terms=["found", "increased", "significant", "effect", "conclude", "result"],
    )
    validation = select_relevant_sentences(
        text,
        limit=6,
        required_terms=["percent", "p <", "statistically", "participants", "regression"],
    )
    limitations = select_relevant_sentences(
        text,
        limit=7,
        required_terms=["limitations", "hypothetical", "may not", "did not", "not"],
    )
    return "\n".join(
        [
            "Claims and Findings",
            _bullet_block(findings) or "- No clear findings detected.",
            "",
            "Evidence and Validation Signals",
            _bullet_block(validation) or "- No quantitative validation detected.",
            "",
            "Limitations, Assumptions, and Threats",
            _bullet_block(limitations) or "- No explicit limitations detected.",
            "",
            "Faithfulness Notes",
            "- Statistical significance is reported only where the extracted text explicitly reports it.",
            "- Extractive mode does not independently verify claims.",
        ]
    ).strip()


def _extractive_query_analysis(text: str, query: str) -> str:
    relevant = select_relevant_sentences(text, query=query, limit=8)
    limitations = select_relevant_sentences(
        text,
        query=query,
        limit=5,
        required_terms=["limitations", "hypothetical", "may not", "did not", "not"],
    )
    return "\n".join(
        [
            "Direct Answer",
            _query_answer(relevant, limitations),
            "",
            "Relevant Evidence",
            _bullet_block(relevant) or "- No directly relevant evidence detected.",
            "",
            "Missing Information",
            _bullet_block(limitations) or "- No missing information detected.",
        ]
    ).strip()


def _extractive_standard_output(text: str, sections: list[str], visual_refs: str) -> str:
    findings = select_relevant_sentences(
        text,
        limit=7,
        required_terms=["results", "found", "increased", "significant", "conclude"],
    )
    methodology = select_relevant_sentences(
        text,
        limit=6,
        required_terms=["participants", "randomly", "conditions", "measured", "regression"],
    )
    validation = select_relevant_sentences(
        text,
        limit=6,
        required_terms=["percent", "p <", "statistically", "participants"],
    )
    limitations = select_relevant_sentences(
        text,
        limit=7,
        required_terms=["limitations", "hypothetical", "may not", "did not", "not"],
    )
    research = select_relevant_sentences(text, limit=3, required_terms=["asks", "whether"])
    return "\n".join(
        [
            "Executive Summary",
            _paragraph(findings[:3]) or "No concise executive summary could be extracted.",
            "",
            "Research Question",
            _paragraph(research) or "Not clearly stated in extracted text.",
            "",
            "Key Findings",
            _bullet_block(findings) or "- No major findings detected.",
            "",
            "Methodology",
            _bullet_block(methodology) or "- No clear methodology details detected.",
            "",
            "Evidence and Validation",
            _bullet_block(validation) or "- No quantitative validation detected.",
            "",
            "Limitations",
            _bullet_block(limitations) or "- No explicit limitations detected; treat this as uncertain.",
            "",
            "Practical Takeaways",
            "- Translate reported findings into startup decisions only where the paper's setting matches your users and product context.",
            "- Treat trust improvements cautiously when calibration, real-world behavior, or long-term outcomes are not validated.",
            "",
            "Terminology",
            _bullet_block(keywords(text, limit=10)) or "- No specialized terminology detected.",
            "",
            "Recommended Original Sections to Review",
            _visual_review_lines(visual_refs),
        ]
    ).strip()


def _extractive_query_output(text: str, query: str, visual_refs: str) -> str:
    relevant = select_relevant_sentences(text, query=query, limit=8)
    limitations = select_relevant_sentences(
        text,
        query=query,
        limit=6,
        required_terms=["limitations", "hypothetical", "may not", "did not", "not"],
    )
    return "\n".join(
        [
            "Direct Answer",
            _query_answer(relevant, limitations),
            "",
            "Relevant Evidence",
            _bullet_block(relevant) or "- No directly relevant evidence detected.",
            "",
            "Supporting Quotes",
            _bullet_block(relevant[:3]) or "- No short supporting excerpts detected.",
            "",
            "Confidence Assessment",
            _confidence_line(relevant),
            "",
            "Missing Information",
            _bullet_block(limitations) or "- No missing information was clearly detected in the extracted text.",
            "",
            "Startup Relevance",
            "- Use this evidence to shape product trust features, validation plans, risk controls, and claims you are willing to make.",
            "",
            "Recommended Original Sections to Review",
            _visual_review_lines(visual_refs),
        ]
    ).strip()


def _bullet_block(items) -> str:
    cleaned = []
    seen = set()
    for item in items:
        value = str(item).strip(" -")
        if not value or value.lower() in seen:
            continue
        cleaned.append(f"- {value}")
        seen.add(value.lower())
    return "\n".join(cleaned)


def _paragraph(items: list[str]) -> str:
    return " ".join(item.strip(" -") for item in items if item.strip(" -"))


def _query_answer(relevant: list[str], limitations: list[str]) -> str:
    if not relevant:
        return "No. The extracted text does not provide direct evidence for the query."
    prefix = "Partially."
    if any("significant" in item.lower() or "p <" in item.lower() for item in relevant):
        prefix = "Yes, with important caveats."
    if limitations:
        return f"{prefix} The paper contains relevant evidence, but it leaves limitations or missing validation."
    return f"{prefix} The paper contains relevant evidence for the query."


def _confidence_line(relevant: list[str]) -> str:
    if not relevant:
        return "No evidence. The query was not answered by the extracted text."
    if any("significant" in item.lower() or "p <" in item.lower() for item in relevant):
        return "Moderate evidence. The extracted text includes relevant quantitative or statistically reported results, but study design limits still matter."
    return "Weak to moderate evidence. Relevant claims were extracted, but the support is limited or not clearly statistical."


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
