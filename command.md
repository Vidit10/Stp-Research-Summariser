Your task is to design and implement a Python-based agent that helps startup founders and researchers rapidly extract actionable information from academic papers without requiring them to read the entire document.

The goal is not academic review or literature analysis. The goal is information compression with minimal loss of relevant context.

## Background

The user is conducting research for a startup. During exploration, they frequently encounter academic papers, technical reports, whitepapers, validation studies, benchmarks, and related research.

Reading every paper in full is impractical because:

* Papers are often lengthy.
* Most content is irrelevant to the immediate decision being made.
* Deep reading can distract from the current research objective.
* The user primarily wants validated insights, evidence, assumptions, limitations, methodologies, and conclusions.

The agent should therefore act as an intelligent paper distillation system.

---

# High-Level Requirements

Build a Python application that accepts:

1. A paper (PDF preferred, but extensible to other formats).
2. An optional user instruction describing what they are looking for.

Examples:

* "Look for evidence supporting consumer trust."
* "Extract findings relevant to AI agent reliability."
* "Focus on limitations and negative results."
* "Tell me whether this validates my startup hypothesis."

The system should operate in two modes.

---

# Mode 1: Standard Summarization (No User Query)

If the user provides only a paper:

The agent should generate a comprehensive summary document.

Objectives:

* Preserve all important context.
* Remove redundancy.
* Translate academic language into clearer language.
* Compress information aggressively.

Target size:

* Approximately 5–15% of original length.
* Smaller when possible.
* Never sacrifice critical context merely to hit a length target.

The summary should contain:

## Executive Summary

A concise explanation of:

* What problem the paper addresses.
* Why it matters.
* Core findings.
* Practical implications.

---

## Research Question

What question is the paper trying to answer?

---

## Key Findings

List all major findings.

For each finding:

* Finding
* Evidence supporting it
* Confidence level if available

---

## Methodology

Summarize:

* Dataset
* Sample size
* Experimental design
* Controls
* Statistical methods
* Evaluation methods

Only include details necessary to evaluate credibility.

---

## Evidence and Validation

Extract:

* Quantitative results
* Benchmarks
* Improvements
* Statistical significance
* Validation methods

Avoid reproducing excessive tables.

---

## Limitations

Explicitly extract:

* Author-stated limitations
* Potential weaknesses
* Assumptions
* Threats to validity

This section is mandatory.

---

## Practical Takeaways

Convert academic findings into practical language.

Answer:

* What should a startup founder learn from this?
* What decisions might this affect?
* What should be treated cautiously?

---

## Terminology

Briefly explain important domain-specific terms introduced by the paper.

---

## Recommended Reading Sections

Some papers contain information that cannot be safely compressed.

Examples:

* Critical diagrams
* Figures
* Graphs
* Tables
* Equations
* Mathematical derivations
* Architecture diagrams
* Flowcharts

When such content exists, do NOT attempt to recreate it.

Instead generate:

"Recommended Original Sections to Review"

and specify:

* Figure number
* Table number
* Equation number
* Page number
* Reason it should be reviewed directly

Example:

"Figure 3 should be reviewed directly because it contains a multi-stage architecture diagram whose details cannot be reliably represented in text."

---

# Mode 2: Goal-Driven Extraction (User Query Provided)

If the user provides additional instructions, the agent should behave differently.

Instead of producing a generic summary first, it should prioritize extracting information relevant to the user's objective.

Example:

Paper + Query:
"Does this paper provide evidence that users trust AI-generated financial advice?"

The agent should:

## Direct Answer

Provide a concise answer first.

Example:

"Partially. The paper demonstrates increased trust under X conditions but does not test Y."

---

## Relevant Evidence

Extract only findings relevant to the query.

For each piece of evidence include:

* Claim
* Supporting data
* Experimental setup
* Limitations

---

## Supporting Quotes

Extract the most relevant excerpts from the paper.

Include page references when available.

Avoid excessive quoting.

---

## Confidence Assessment

Classify:

* Strong evidence
* Moderate evidence
* Weak evidence
* No evidence

Explain why.

---

## Missing Information

Identify:

* What the paper does not answer.
* What assumptions would still require validation.

---

## Startup Relevance

Translate findings into implications for product, business model, user behavior, growth, risk, regulation, or technical implementation.

---

# Multi-Pass Analysis Requirement

The implementation must use multiple reasoning passes.

Pass 1:
Document understanding.

Pass 2:
Structure extraction.

Pass 3:
Evidence extraction.

Pass 4:
Query-focused analysis (if query exists).

Pass 5:
Final synthesis.

Do not rely on a single-pass summary.

---

# Hallucination Prevention

The system must prioritize faithfulness over completeness.

Rules:

* Never invent findings.
* Never infer statistical significance if not reported.
* Never create citations that do not exist.
* Clearly mark uncertainty.
* Distinguish between:

  * Author claims
  * Experimental evidence
  * Agent interpretation

---

# Output Format

Generate a plain text (.txt) file.

The output should be:

* Human-readable
* Structured with headings
* Easy to skim
* Suitable for founders and researchers

Avoid markdown tables whenever possible.

---

# Technical Requirements

Implementation language:
Python

Preferred architecture:

* PDF ingestion
* Text extraction
* Chunking
* Hierarchical summarization
* Structured information extraction
* Query-aware retrieval
* Final synthesis

The design should support:

* Local LLMs
* OpenAI-compatible APIs
* Anthropic-compatible APIs

via interchangeable model providers.

---

# Stretch Goals

If feasible, also implement:

1. Figure detection.
2. Table detection.
3. Citation extraction.
4. Study quality scoring.
5. Evidence strength scoring.
6. Automatic identification of:

   * causal claims
   * correlational claims
   * assumptions
   * future work
7. Export to:

   * TXT
   * Markdown
   * JSON

The final deliverable should be production-quality Python code with clear architecture, modular components, configuration support, logging, and robust error handling.
