# Paper Distiller

Paper Distiller is a Python agent that compresses academic papers, technical reports,
whitepapers, validation studies, and benchmarks into decision-ready notes for startup
founders and researchers.

It supports two modes:

- Standard summarization when only a paper is provided.
- Goal-driven extraction when a user query is provided.

The implementation is intentionally multi-pass:

1. Document understanding
2. Structure extraction
3. Evidence extraction
4. Query-focused analysis, when a query exists
5. Final synthesis

## Install

```powershell
python -m pip install -e .
```

PDF extraction works best with optional dependencies:

```powershell
python -m pip install -e ".[pdf]"
```

Provider integrations are optional:

```powershell
python -m pip install -e ".[providers]"
```

## Quick Start

Standard summarization:

```powershell
paper-distiller papers\paper.pdf --output outputs\summary.txt
```

Goal-driven extraction:

```powershell
paper-distiller path\to\paper.pdf --query "Does this validate consumer trust in AI financial advice?" --output outputs\trust.txt
```

Without an API key, the agent uses a deterministic extractive fallback. For higher quality
synthesis, set a provider and model:

```powershell
$env:PAPER_DISTILLER_PROVIDER="openai"
$env:OPENAI_API_KEY="..."
$env:PAPER_DISTILLER_MODEL="gpt-4.1-mini"
paper-distiller paper.pdf --query "Focus on limitations and negative results."
```

OpenAI-compatible local servers are supported:

```powershell
$env:PAPER_DISTILLER_PROVIDER="openai-compatible"
$env:PAPER_DISTILLER_BASE_URL="http://localhost:11434/v1"
$env:PAPER_DISTILLER_API_KEY="ollama"
$env:PAPER_DISTILLER_MODEL="llama3.1"
```

Anthropic-compatible usage:

```powershell
$env:PAPER_DISTILLER_PROVIDER="anthropic"
$env:ANTHROPIC_API_KEY="..."
$env:PAPER_DISTILLER_MODEL="claude-3-5-sonnet-latest"
```

BharatCode usage:

```powershell
$env:PAPER_DISTILLER_PROVIDER="bharatcode"
$env:PAPER_DISTILLER_API_KEY="..."
$env:PAPER_DISTILLER_MODEL="their-model-name"
paper-distiller paper.pdf --query "Focus on startup relevance."
```

The BharatCode provider defaults to:

```text
https://bharatcode.ai/api/model/v1
```

To discover the model name from BharatCode's catalog:

```powershell
$env:PAPER_DISTILLER_PROVIDER="bharatcode"
$env:PAPER_DISTILLER_API_KEY="..."
paper-distiller --list-models
```

## Configuration

See `.env.example` for supported settings. CLI flags override environment defaults where
applicable.

## Output

The default output is plain text, built for skimming. Markdown and JSON exporters are included
for extension points, but TXT is the primary deliverable.

## Hallucination Controls

The agent prompts and fallback logic explicitly separate:

- Author claims
- Experimental evidence
- Agent interpretation

It also avoids inventing statistical significance, citations, or results. When evidence is
missing or ambiguous, the output should say so directly.
