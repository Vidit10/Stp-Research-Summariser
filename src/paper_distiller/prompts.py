SYSTEM_PROMPT = """You are a paper distillation agent for startup founders and researchers.
Your job is information compression with minimal loss of decision-relevant context.

Faithfulness rules:
- Never invent findings, citations, statistics, or significance.
- Clearly distinguish author claims, experimental evidence, and your interpretation.
- Mark uncertainty when the paper is unclear or extraction quality is weak.
- Do not over-compress methods, limitations, validation, or negative results.
- Prefer plain language over academic wording.
"""

DOCUMENT_UNDERSTANDING_PROMPT = """PASS 1: Document understanding.

Summarize the paper chunk below. Focus on:
- Problem addressed
- Research question or hypothesis
- Domain and setting
- Claimed contributions
- Any caveats about extraction quality

Return concise bullets with page references when visible.

CHUNK {chunk_index}:
{chunk_text}
"""

STRUCTURE_EXTRACTION_PROMPT = """PASS 2: Structure extraction.

From the pass-1 notes and detected sections, extract the document structure:
- Main sections
- Research question
- Methodology details needed to judge credibility
- Datasets, sample sizes, experimental design, controls, statistical or evaluation methods
- Places where figures, tables, equations, or diagrams likely need direct review

PASS 1 NOTES:
{understanding}

DETECTED SECTIONS:
{sections}

VISUAL REFERENCES:
{visual_refs}
"""

EVIDENCE_EXTRACTION_PROMPT = """PASS 3: Evidence extraction.

Extract claims and evidence from the paper notes.
For each item, include:
- Claim or finding
- Supporting data or setup
- Whether it is an author claim, experimental evidence, or agent interpretation
- Confidence level only if available or inferable from reported evidence quality
- Limitations, assumptions, threats to validity, or missing validation

Do not invent statistical significance. If significance is not reported, say so.

PASS 1 NOTES:
{understanding}

PASS 2 STRUCTURE:
{structure}
"""

QUERY_ANALYSIS_PROMPT = """PASS 4: Query-focused analysis.

USER QUERY: {query}

Use only the extracted structure and evidence to answer the user's objective.
Prioritize direct relevance over generic summary.
Include:
- Direct answer
- Relevant evidence
- Supporting short excerpts if available
- Confidence assessment: Strong, Moderate, Weak, or No evidence
- Missing information
- Startup relevance

STRUCTURE:
{structure}

EVIDENCE:
{evidence}
"""

STANDARD_SYNTHESIS_PROMPT = """PASS 5: Final synthesis.

Create a plain-text summary for founders and researchers. Use these headings exactly:

Executive Summary
Research Question
Key Findings
Methodology
Evidence and Validation
Limitations
Practical Takeaways
Terminology
Recommended Original Sections to Review

Compress aggressively but preserve critical context. Limitations are mandatory.
Avoid markdown tables.

DOCUMENT STRUCTURE:
{structure}

EVIDENCE:
{evidence}

VISUAL REFERENCES:
{visual_refs}
"""

QUERY_SYNTHESIS_PROMPT = """PASS 5: Final synthesis.

Create a plain-text query-driven extraction for founders and researchers. Use these headings exactly:

Direct Answer
Relevant Evidence
Supporting Quotes
Confidence Assessment
Missing Information
Startup Relevance
Recommended Original Sections to Review

USER QUERY: {query}

Use only the extracted material. Avoid markdown tables. Keep quotes short and page-referenced
when page references are available.

QUERY ANALYSIS:
{query_analysis}

VISUAL REFERENCES:
{visual_refs}
"""

