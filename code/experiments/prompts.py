from __future__ import annotations

from pathlib import Path


def read_text(path: str | Path) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def format_user_prompt(template_path: str | Path, **placeholders: str | int) -> str:
    text = read_text(template_path)
    for key, value in placeholders.items():
        text = text.replace(f"{{{key}}}", str(value))
    return text


def system_prompt_path(rq_config, mode: str) -> str:
    if mode == "structured":
        return rq_config.system_prompt_structured
    if mode == "prompt":
        return rq_config.system_prompt_output
    raise ValueError(f"Unknown output mode: {mode}")
