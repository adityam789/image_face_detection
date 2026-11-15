"""
facesorter/search/llm/_hub.py
First-run download notice for models fetched from the Hugging Face Hub.

``huggingface_hub`` is imported lazily and every failure here is swallowed: the notice is a
courtesy, never a reason for a load to fail.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_GLOB_CHARS = set("*?[")


def is_cached(repo_id: str, filename: str | None = None) -> bool | None:
    """Whether the model is already in the local Hub cache; ``None`` if it cannot be told.

    Without ``filename`` the repo's ``config.json`` stands in for "the snapshot is present"
    (every MLX model repo has one). The lookup is a pure cache read: no network, no progress bar.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    if filename is not None and _GLOB_CHARS & set(filename):
        return None  # a pattern cannot be looked up without listing the repo
    try:
        return isinstance(try_to_load_from_cache(repo_id, filename or "config.json"), str)
    except Exception:
        return None


def announce_download(repo_id: str, size: str, filename: str | None = None) -> None:
    """Log a one-time notice when ``repo_id`` is about to be downloaded.

    Skips local paths and models that are already cached. ``size`` is a human-readable
    estimate such as ``"0.9 GB"``.
    """
    if os.path.exists(repo_id):
        return
    if is_cached(repo_id, filename) is False:
        logger.info(
            "Downloading %s (about %s) from the Hugging Face Hub. This happens once; the file "
            "is cached for later runs. Set HF_HUB_OFFLINE=1 to forbid downloads.",
            repo_id if filename is None else f"{repo_id}/{filename}",
            size,
        )
