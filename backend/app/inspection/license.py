"""Best-effort license detection from files shipped inside a model package.

"Downloadable from Hugging Face" is not a license. Anything not positively
identified stays ``unknown`` and is flagged until a human sets it.
"""
from __future__ import annotations

from pathlib import Path

_MAX = 200_000
_RULES: list[tuple[str, str, tuple[str, ...]]] = [
    ("Coqui Public Model License (non-commercial)", "noncommercial", ("coqui public model", "cpml")),
    ("CC BY-NC (non-commercial)", "noncommercial", ("cc-by-nc", "cc by-nc", "noncommercial", "non-commercial")),
    ("OpenRAIL", "conditional", ("openrail",)),
    ("GPL", "conditional", ("gnu general public license", "gpl-3", "agpl")),
    ("Apache-2.0", "permissive", ("apache license", "apache-2.0", "apache 2.0")),
    ("MIT", "permissive", ("mit license", "license: mit")),
    ("CC BY 4.0", "permissive", ("cc-by-4.0", "cc by 4.0", "creative commons attribution 4.0")),
    ("CC0", "permissive", ("cc0",)),
]


def detect_license(root: Path, files: list[dict]) -> tuple[str, str, str]:
    """Return (license name, status, source file). Status: permissive | conditional | noncommercial | unknown."""
    candidates = [
        f["path"]
        for f in files
        if Path(f["path"]).name.lower().startswith(("license", "licence", "readme", "model_card"))
    ]
    for rel in sorted(candidates, key=lambda p: not Path(p).name.lower().startswith("lic")):
        try:
            text = (root / rel).read_text(encoding="utf-8", errors="replace")[:_MAX].lower()
        except OSError:
            continue
        for name, status, needles in _RULES:
            if any(n in text for n in needles):
                return name, status, rel
    return "unknown", "unknown", ""
