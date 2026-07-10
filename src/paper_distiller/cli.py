from __future__ import annotations

import argparse
import logging
from pathlib import Path

from paper_distiller.config import Settings
from paper_distiller.exporters import export_result
from paper_distiller.ingest import IngestionError, load_document
from paper_distiller.models import list_openai_compatible_models
from paper_distiller.pipeline import PaperDistiller


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper-distiller",
        description="Distill academic papers into startup-relevant decision notes.",
    )
    parser.add_argument("paper", type=Path, nargs="?", help="Path to a PDF, TXT, or Markdown paper.")
    parser.add_argument(
        "-q",
        "--query",
        help="Optional goal-driven instruction, e.g. 'Focus on limitations and negative results.'",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file path. Defaults to <paper-name>.distilled.txt.",
    )
    parser.add_argument(
        "--format",
        choices=["txt", "md", "json"],
        default="txt",
        help="Output format. TXT is the primary supported deliverable.",
    )
    parser.add_argument(
        "--provider",
        choices=["extractive", "openai", "openai-compatible", "anthropic", "bharatcode"],
        help="Model provider. Defaults to PAPER_DISTILLER_PROVIDER or extractive.",
    )
    parser.add_argument("--model", help="Model name for the selected provider.")
    parser.add_argument("--base-url", help="Base URL for OpenAI-compatible providers.")
    parser.add_argument("--api-key", help="API key. Prefer environment variables for normal use.")
    parser.add_argument("--max-chars-per-chunk", type=int, help="Maximum characters per chunk.")
    parser.add_argument("--overlap-chars", type=int, help="Overlapping characters between chunks.")
    parser.add_argument(
        "--include-passes",
        action="store_true",
        help="Append intermediate pass outputs to the output file for auditing.",
    )
    parser.add_argument("--log-level", help="Logging level, e.g. INFO or DEBUG.")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models from the selected OpenAI-compatible provider and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = Settings.from_env(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_chars_per_chunk=args.max_chars_per_chunk,
        overlap_chars=args.overlap_chars,
        log_level=args.log_level,
    )
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.list_models:
            base_url = settings.base_url
            if settings.provider == "bharatcode" and not base_url:
                base_url = "https://bharatcode.ai/api/model/v1"
            if settings.provider not in {"bharatcode", "openai-compatible"}:
                raise ValueError("--list-models is supported for bharatcode and openai-compatible.")
            if not base_url:
                raise ValueError("Set PAPER_DISTILLER_BASE_URL or use PAPER_DISTILLER_PROVIDER=bharatcode.")
            models = list_openai_compatible_models(base_url=base_url, api_key=settings.api_key)
            if not models:
                print("No models returned by provider.")
            else:
                print("Available models:")
                for model in models:
                    print(f"- {model}")
            return 0

        if args.paper is None:
            raise ValueError("paper is required unless --list-models is used.")
        document = load_document(args.paper)
        distiller = PaperDistiller(settings)
        result = distiller.distill(document, query=args.query)
        if args.include_passes:
            result = _append_passes(result)
        output_path = args.output or args.paper.with_suffix(f".distilled.{args.format}")
        written = export_result(result, output_path, fmt=args.format)
    except (IngestionError, ValueError, RuntimeError) as exc:
        parser.exit(status=1, message=f"paper-distiller: error: {exc}\n")

    print(f"Wrote {result.mode} distillation to {written}")
    return 0


def _append_passes(result):
    from dataclasses import replace

    audit = ["", "", "Intermediate Pass Audit", "======================="]
    for name, output in result.pass_outputs.items():
        audit.extend(["", name, "-" * len(name), output])
    return replace(result, text=result.text.rstrip() + "\n" + "\n".join(audit))


if __name__ == "__main__":
    raise SystemExit(main())
