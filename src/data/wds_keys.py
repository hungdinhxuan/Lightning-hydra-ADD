#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable WebDataset sample key helpers."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


WDS_KEY_VERSION = "safe-no-dot-v1"
_UNSAFE_KEY_CHARS = re.compile(r"[^A-Za-z0-9_-]+")
_REPEATED_UNDERSCORES = re.compile(r"_+")


def make_wds_key(relpath: str) -> str:
    """Return a WebDataset-safe key for a protocol relative path.

    WebDataset splits sample fields at dots in tar member names. Keeping dots in
    ``__key__`` can turn ``foo.bar.wav`` into key ``foo`` plus field
    ``bar.wav``. The loader then receives a grouped sample without a top-level
    ``wav`` field. Use only safe characters and append a short hash to avoid
    collisions between similarly normalized paths.
    """
    stem = Path(relpath).with_suffix("").as_posix()
    safe = _UNSAFE_KEY_CHARS.sub("_", stem)
    safe = _REPEATED_UNDERSCORES.sub("_", safe).strip("_")
    if not safe:
        safe = "sample"

    digest = hashlib.sha1(relpath.encode("utf-8")).hexdigest()[:12]
    max_prefix_len = 180
    if len(safe) > max_prefix_len:
        safe = safe[:max_prefix_len].rstrip("_")
    return f"{safe}_{digest}"
