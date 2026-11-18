"""Parse the ``nomenclatures`` section of ``medstt.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from medstt.nomenclatures.importers import IMPORTERS, LOCAL_FORMAT

_DOWNLOAD_KEYS = {"format", "url", "file"}
_LOCAL_KEYS = {"format", "path"}
#: Directory read by the ``local`` format when ``path`` is not set.
DEFAULT_LOCAL_PATH = "local"


@dataclass(frozen=True)
class NomenclatureSpec:
    """One source nomenclature: its importer format and where its data comes from."""

    name: str
    format: str
    #: Download URL; empty for the ``local`` format.
    url: str = ""
    #: File to import when the downloaded archive holds several candidates.
    file: str | None = None
    #: File or directory of ``*.txt`` files read by the ``local`` format.
    path: str | None = None

    @property
    def is_local(self) -> bool:
        """Whether the data is read from ``path`` instead of downloaded."""
        return self.format == LOCAL_FORMAT

    @classmethod
    def parse(cls, name: str, raw: Any) -> NomenclatureSpec:
        """Validate one ``nomenclatures.<name>`` mapping."""
        if not isinstance(raw, dict):
            raise ValueError(f"nomenclatures.{name} must be a mapping with at least `format`.")
        fmt = raw.get("format")
        local = fmt == LOCAL_FORMAT
        unknown = set(raw) - (_LOCAL_KEYS if local else _DOWNLOAD_KEYS)
        if unknown:
            raise ValueError(f"nomenclatures.{name}: unknown keys {sorted(unknown)}.")
        if local:
            return cls(name=name, format=fmt, path=str(raw.get("path") or DEFAULT_LOCAL_PATH))
        url = raw.get("url")
        if not fmt or not url:
            raise ValueError(f"nomenclatures.{name} requires `format` and `url`.")
        if fmt not in IMPORTERS:
            raise ValueError(
                f"nomenclatures.{name}: unknown format {fmt!r}; expected one of "
                f"{sorted(IMPORTERS)}."
            )
        return cls(name=name, format=fmt, url=str(url), file=raw.get("file") or None)


def parse_nomenclatures(section: dict[str, Any]) -> list[NomenclatureSpec]:
    """Return the configured nomenclatures in file order."""
    return [NomenclatureSpec.parse(name, raw) for name, raw in section.items()]
