from pathlib import Path

from paper_distiller.config import Settings
from paper_distiller.ingest import load_document
from paper_distiller.pipeline import PaperDistiller


def test_standard_mode_runs_on_text_fixture():
    document = load_document(Path("examples/sample_paper.txt"))
    result = PaperDistiller(Settings()).distill(document)
    assert result.mode == "standard"
    assert "Limitations" in result.text
    assert "trust" in result.text.lower()


def test_query_mode_runs_on_text_fixture():
    document = load_document(Path("examples/sample_paper.txt"))
    result = PaperDistiller(Settings()).distill(
        document,
        query="Does this paper validate consumer trust in AI-generated financial advice?",
    )
    assert result.mode == "goal-driven"
    assert "Direct Answer" in result.text or "direct answer" in result.text.lower()
    assert "financial advice" in result.text.lower()

