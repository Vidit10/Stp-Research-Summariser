# Paper Distiller: AI Research Intelligence Agent

> ⭐ **If you find this tool helpful for your research or startup journey, please consider starring this repository! It helps others discover the project.**

Paper Distiller is a Python agent that compresses academic papers, technical reports, whitepapers, validation studies, and benchmarks into decision-ready notes. It is purpose-built for founders, researchers, and developers looking to rapidly extract insights and ship high-value MVPs without wading through hundreds of pages of text.

It supports two modes:

* **Standard summarization:** When only a paper is provided.
* **Goal-driven extraction:** When a user query is provided.

The implementation is intentionally multi-pass:

1. Document understanding
2. Structure extraction
3. Evidence extraction
4. Query-focused analysis, when a query exists
5. Final synthesis

---

## 🛠️ Setup: Virtual Environment

Before installing the dependencies, it is highly recommended to create an isolated Python virtual environment. This keeps your system clean and ensures package versions do not conflict.

**Windows:**

```powershell
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 📦 Install

Once your virtual environment is active, install the package:

```powershell
python -m pip install -e .
```

PDF extraction works best with optional dependencies:

```powershell
python -m pip install -e ".[pdf]"
```

Provider integrations (OpenAI, Anthropic, Gemini, etc.) are optional but recommended:

```powershell
python -m pip install -e ".[providers]"
```

---

## 🚀 Quick Start

**Standard summarization:**

```powershell
paper-distiller papers\paper.pdf --output outputs\summary.txt
```

> **💡 Check it out:** We have already run an example for you! Navigate to `outputs/summary.txt` to see how the agent distilled a complex paper (*"A Recommender System for Trip Planners"*) into an executive summary, key findings, and practical takeaways.

**Goal-driven extraction:**

```powershell
paper-distiller path\to\paper.pdf --query "Does this validate consumer trust in AI financial advice?" --output outputs\trust.txt
```

### LLM Provider Configurations

Without an API key, the agent uses a deterministic extractive fallback. For higher quality synthesis, set up one of the following providers:

**Google Gemini:**

```powershell
$env:PAPER_DISTILLER_PROVIDER="gemini"
$env:GEMINI_API_KEY="your_api_key_here"
$env:PAPER_DISTILLER_MODEL="gemini-1.5-pro-latest"
```

**OpenAI:**

```powershell
$env:PAPER_DISTILLER_PROVIDER="openai"
$env:OPENAI_API_KEY="..."
$env:PAPER_DISTILLER_MODEL="gpt-4o-mini"
```

**Anthropic:**

```powershell
$env:PAPER_DISTILLER_PROVIDER="anthropic"
$env:ANTHROPIC_API_KEY="..."
$env:PAPER_DISTILLER_MODEL="claude-3-5-sonnet-latest"
```

**Local/Ollama (OpenAI-Compatible):**

```powershell
$env:PAPER_DISTILLER_PROVIDER="openai-compatible"
$env:PAPER_DISTILLER_BASE_URL="http://localhost:11434/v1"
$env:PAPER_DISTILLER_API_KEY="ollama"
$env:PAPER_DISTILLER_MODEL="llama3.1"
```

---

## ⚙️ Configuration

See `.env.example` for supported settings. CLI flags override environment defaults where applicable.

## 📄 Output

The default output is plain text, built for skimming. Markdown and JSON exporters are included for extension points, but TXT is the primary deliverable.

## 🛡️ Hallucination Controls

The agent prompts and fallback logic explicitly separate:

* Author claims
* Experimental evidence
* Agent interpretation

It also avoids inventing statistical significance, citations, or results. When evidence is missing or ambiguous, the output should say so directly.

---

## 🤝 Connect with Me

I'm **Vidit Parikh**, passionate about building intelligent tools, exploring remote startup opportunities, and writing clean, scalable systems.

* **LinkedIn:** [Connect with me on LinkedIn](https://www.linkedin.com/in/vidit-parikh/)
* **Email:** [mc23bt010@iitdh.ac.in](mailto:mc23bt010@iitdh.ac.in)

If you use Paper Distiller to scale your research or build your next MVP, I’d love to hear about it! **Don't forget to ⭐ star the repository if you found it useful.**