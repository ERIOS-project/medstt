"""Import hand-built term lists: a directory of ``*.txt`` files, one term per line."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from medstt.nomenclatures.importers.base import Concept, clean_text

SOURCE = "local"


def iter_local_concepts(path: Path) -> Iterator[Concept]:
    """Yield one concept per nonempty line of ``path`` or of each ``*.txt`` file in it.

    Files are read in name order. The file name without its extension is the concept
    category; the code is that name followed by the line number. Lines starting with
    ``#`` are comments.
    """
    if not path.exists():
        raise FileNotFoundError(f"local term list not found: {path}")
    files = sorted(path.glob("*.txt")) if path.is_dir() else [path]
    for file in files:
        with file.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                label = clean_text(line)
                if not label or label.startswith("#"):
                    continue
                yield Concept(
                    code=f"{file.stem}:{number}",
                    source=SOURCE,
                    label=label,
                    category=file.stem,
                    extra={"file": file.name, "line": number},
                )
