"""Hugging Face / sentence-transformers cache in a project-local writable directory."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_SBERT_MODEL = "all-MiniLM-L6-v2"


def project_hf_cache_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "data" / "caches" / "huggingface"


def configure_hf_cache(cache_dir: Path | str | None = None) -> Path:
    """
    Force HF caches into a writable project folder.

    Overwrites machine-wide defaults (e.g. /data/caches/huggingface) that may not
    be writable for your user.
    """
    cache = Path(cache_dir) if cache_dir else project_hf_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    hub = cache / "hub"
    hub.mkdir(parents=True, exist_ok=True)

    # Assign (do not setdefault) so we override broken system paths.
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(hub)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub)
    os.environ["TRANSFORMERS_CACHE"] = str(hub)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache)
    return cache


def sbert_repo_id(model_name: str) -> str:
    if "/" in model_name:
        return model_name
    return f"sentence-transformers/{model_name}"


def sbert_local_dir(cache: Path, model_name: str) -> Path:
    return cache / "models" / sbert_repo_id(model_name).replace("/", "--")


def load_sentence_transformer(model_name: str = DEFAULT_SBERT_MODEL):
    """Download (once) and load SBERT from the project-local cache only."""
    cache = configure_hf_cache()
    hub = cache / "hub"
    local_dir = sbert_local_dir(cache, model_name)
    repo_id = sbert_repo_id(model_name)

    if not (local_dir / "config.json").is_file():
        from huggingface_hub import snapshot_download

        local_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo_id} to {local_dir} ...")
        snapshot_download(
            repo_id=repo_id,
            cache_dir=str(hub),
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(local_dir))
