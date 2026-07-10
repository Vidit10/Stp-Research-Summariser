from __future__ import annotations

import json
from pathlib import Path

from paper_distiller.pipeline import DistillationResult


def export_result(result: DistillationResult, output_path: Path, *, fmt: str = "txt") -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = fmt.lower()
    if normalized == "txt":
        output_path.write_text(result.text + "\n", encoding="utf-8")
    elif normalized in {"md", "markdown"}:
        output_path.write_text(_to_markdown(result), encoding="utf-8")
    elif normalized == "json":
        output_path.write_text(_to_json(result), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported export format: {fmt}")
    return output_path


def _to_markdown(result: DistillationResult) -> str:
    lines = [f"# Paper Distillation ({result.mode})", ""]
    for line in result.text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("-") and len(stripped) < 80:
            lines.append(f"## {stripped}")
        else:
            lines.append(line)
    return "\n".join(lines).strip() + "\n"


def _to_json(result: DistillationResult) -> str:
    return json.dumps(
        {
            "mode": result.mode,
            "text": result.text,
            "passes": result.pass_outputs,
        },
        indent=2,
        ensure_ascii=False,
    )

